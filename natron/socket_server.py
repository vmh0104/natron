"""
Alternative Socket Server for MQL5 Integration
MQL5 sockets work better with raw TCP sockets than HTTP
"""
import socket
import threading
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

from feature_engine import FeatureEngine
from sequence_creator import SequenceCreator
from model import create_model
from label_generator import LabelGenerator


class NatronSocketServer:
    """TCP Socket server for MQL5 integration."""
    
    def __init__(self, model_path: str = 'model/natron_v2.pt', config_path: str = 'config.yaml'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        import yaml
        if Path(config_path).exists():
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {}
        
        # Initialize components
        self.feature_engine = FeatureEngine()
        self.sequence_creator = SequenceCreator(sequence_length=96)
        self.label_generator = LabelGenerator()
        
        # Create and load model
        model_cfg = self.config.get('model', {})
        model_cfg.setdefault('d_model', 128)
        model_cfg.setdefault('nhead', 8)
        model_cfg.setdefault('num_layers', 6)
        model_cfg.setdefault('dim_feedforward', 512)
        model_cfg.setdefault('dropout', 0.1)
        model_cfg.setdefault('num_features', 100)
        model_cfg.setdefault('max_seq_len', 96)
        
        self.model = create_model(model_cfg)
        
        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from {model_path}")
        
        self.model.eval()
        self.model.to(self.device)
        
        # Load feature columns
        feature_cols_path = Path('model/feature_columns.pkl')
        if feature_cols_path.exists():
            with open(feature_cols_path, 'rb') as f:
                self.feature_columns = pickle.load(f)
        else:
            self.feature_columns = None
        
        print("Model loaded successfully")
    
    def predict(self, candles: list) -> dict:
        """Predict from candles."""
        if len(candles) < 96:
            return {'error': f'Need at least 96 candles, got {len(candles)}'}
        
        # Convert to DataFrame
        df = pd.DataFrame(candles[-96:])
        
        # Generate features
        df_features = self.feature_engine.generate_features(df)
        
        # Get feature columns if not set
        if self.feature_columns is None:
            base_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
            self.feature_columns = [col for col in df_features.columns if col not in base_cols]
            Path('model').mkdir(exist_ok=True)
            with open('model/feature_columns.pkl', 'wb') as f:
                pickle.dump(self.feature_columns, f)
        
        # Create sequence
        X_scaled = self.sequence_creator.transform_new_data(df_features)
        X_sequence = X_scaled[-96:].reshape(1, 96, -1)
        
        # Predict
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_sequence).to(self.device)
            outputs = self.model(X_tensor)
            
            buy_prob = outputs['buy'].cpu().item()
            sell_prob = outputs['sell'].cpu().item()
            direction_probs = outputs['direction'].cpu().numpy()[0]
            regime_probs = outputs['regime'].cpu().numpy()[0]
            
            direction_up = direction_probs[1]
            regime_id = int(np.argmax(regime_probs))
            regime_name = self.label_generator.get_regime_name(regime_id)
            
            confidence = max(buy_prob, sell_prob, direction_probs.max(), regime_probs.max())
        
        return {
            'buy_prob': float(buy_prob),
            'sell_prob': float(sell_prob),
            'direction_up': float(direction_up),
            'direction_down': float(direction_probs[0]),
            'regime': regime_name,
            'regime_id': regime_id,
            'confidence': float(confidence)
        }
    
    def handle_client(self, client_socket, address):
        """Handle client connection."""
        print(f"Client connected: {address}")
        
        try:
            buffer = ""
            while True:
                data = client_socket.recv(4096).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                
                # Try to parse JSON (MQL5 sends JSON)
                try:
                    # Look for complete JSON object
                    if buffer.strip().startswith('{'):
                        # Find matching closing brace
                        brace_count = 0
                        json_end = -1
                        for i, char in enumerate(buffer):
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    json_end = i + 1
                                    break
                        
                        if json_end > 0:
                            json_str = buffer[:json_end]
                            buffer = buffer[json_end:]
                            
                            request = json.loads(json_str)
                            
                            if 'candles' in request:
                                # Predict
                                response = self.predict(request['candles'])
                                
                                # Send response
                                response_json = json.dumps(response) + '\n'
                                client_socket.send(response_json.encode('utf-8'))
                            else:
                                error_response = json.dumps({'error': 'Missing "candles" field'}) + '\n'
                                client_socket.send(error_response.encode('utf-8'))
                except json.JSONDecodeError:
                    # Incomplete JSON, wait for more data
                    continue
                except Exception as e:
                    error_response = json.dumps({'error': str(e)}) + '\n'
                    client_socket.send(error_response.encode('utf-8'))
        
        except Exception as e:
            print(f"Error handling client {address}: {e}")
        finally:
            client_socket.close()
            print(f"Client disconnected: {address}")
    
    def start(self, host: str = '0.0.0.0', port: int = 8888):
        """Start socket server."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((host, port))
        server_socket.listen(5)
        
        print(f"Natron Socket Server listening on {host}:{port}")
        
        try:
            while True:
                client_socket, address = server_socket.accept()
                client_thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            server_socket.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron Socket Server')
    parser.add_argument('--model', type=str, default='model/natron_v2.pt', help='Model path')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config path')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host')
    parser.add_argument('--port', type=int, default=8888, help='Port')
    
    args = parser.parse_args()
    
    server = NatronSocketServer(args.model, args.config)
    server.start(args.host, args.port)
