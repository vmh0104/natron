"""
Socket Server for MQL5 Integration
Bidirectional TCP/JSON communication with MetaTrader 5 Expert Advisor
"""

import socket
import json
import threading
import torch
import numpy as np
import pandas as pd
import yaml
import os
import pickle
from model_natron import NatronTransformer
from feature_engine import FeatureEngine
from label_generator import LabelGenerator


class NatronSocketServer:
    """
    TCP Socket server for real-time trading inference.
    Communicates with MQL5 EA via JSON messages.
    """
    
    def __init__(self, model_path: str, config_path: str, host: str = '127.0.0.1', port: int = 8888):
        self.host = host
        self.port = port
        self.model_path = model_path
        self.config_path = config_path
        
        # Load model and components
        self._load_model()
        
        # Socket
        self.socket = None
        self.running = False
        
    def _load_model(self):
        """Load model and scaler"""
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device(
            self.config['training']['device'] if torch.cuda.is_available() else 'cpu'
        )
        
        # Initialize model
        self.model = NatronTransformer(
            feature_dim=self.config['model']['feature_dim'],
            d_model=self.config['model']['d_model'],
            nhead=self.config['model']['nhead'],
            num_layers=self.config['model']['num_layers'],
            dim_feedforward=self.config['model']['dim_feedforward'],
            dropout=self.config['model']['dropout'],
            activation=self.config['model']['activation']
        ).to(self.device)
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load scaler
        scaler_path = os.path.join(self.config['paths']['model_dir'], 'scaler.pkl')
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        else:
            self.scaler = None
        
        # Initialize feature engine
        self.feature_engine = FeatureEngine()
        self.label_generator = LabelGenerator(self.config)
        
        print(f"Model loaded from {self.model_path}")
        print(f"Using device: {self.device}")
    
    def _prepare_sequence(self, candles: list) -> torch.Tensor:
        """Prepare input sequence from OHLCV candles"""
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        
        # Ensure we have exactly sequence_length candles
        sequence_length = self.config['model']['sequence_length']
        if len(df) < sequence_length:
            # Pad with last candle if needed
            last_candle = df.iloc[-1]
            while len(df) < sequence_length:
                df = pd.concat([df, pd.DataFrame([last_candle])], ignore_index=True)
        
        # Take last sequence_length candles
        df = df.tail(sequence_length).reset_index(drop=True)
        
        # Generate features
        features_df = self.feature_engine.generate_all_features(df)
        
        # Normalize
        if self.scaler is not None:
            features = self.scaler.transform(features_df.values)
        else:
            features = features_df.values
        
        # Convert to tensor
        sequence = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        return sequence
    
    def _process_prediction_request(self, data: dict) -> dict:
        """Process prediction request and return response"""
        try:
            candles = data.get('candles', [])
            
            if len(candles) < self.config['model']['sequence_length']:
                return {
                    'status': 'error',
                    'message': f'Need at least {self.config["model"]["sequence_length"]} candles'
                }
            
            # Prepare sequence
            sequence = self._prepare_sequence(candles)
            
            # Predict
            with torch.no_grad():
                predictions = self.model(sequence)
            
            # Extract predictions
            buy_prob = predictions['buy'].item()
            sell_prob = predictions['sell'].item()
            
            # Direction
            direction_logits = predictions['direction']
            direction_probs = torch.exp(direction_logits)
            direction_up = direction_probs[0][1].item()
            
            # Regime
            regime_logits = predictions['regime']
            regime_probs = torch.exp(regime_logits)
            regime_id = regime_probs[0].argmax().item()
            regime_name = self.label_generator.get_regime_name(regime_id)
            
            # Confidence
            confidence = (buy_prob + sell_prob + direction_up + regime_probs[0].max().item()) / 4.0
            
            response = {
                'status': 'success',
                'buy_prob': round(buy_prob, 4),
                'sell_prob': round(sell_prob, 4),
                'direction_up': round(direction_up, 4),
                'regime': regime_name,
                'regime_id': int(regime_id),
                'confidence': round(confidence, 4),
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            return response
        
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def _handle_client(self, client_socket, address):
        """Handle client connection"""
        print(f"Client connected: {address}")
        
        buffer = b''
        
        try:
            while self.running:
                # Receive data
                data = client_socket.recv(4096)
                if not data:
                    break
                
                buffer += data
                
                # Try to parse JSON messages (messages are newline-delimited)
                while b'\n' in buffer:
                    line, buffer = buffer.split(b'\n', 1)
                    
                    try:
                        message = json.loads(line.decode('utf-8'))
                        
                        # Process message
                        msg_type = message.get('type', '')
                        
                        if msg_type == 'predict':
                            response = self._process_prediction_request(message)
                            response['type'] = 'prediction_response'
                        
                        elif msg_type == 'ping':
                            response = {'type': 'pong', 'status': 'ok'}
                        
                        else:
                            response = {
                                'type': 'error',
                                'message': f'Unknown message type: {msg_type}'
                            }
                        
                        # Send response
                        response_json = json.dumps(response) + '\n'
                        client_socket.send(response_json.encode('utf-8'))
                    
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e}")
                        error_response = {
                            'type': 'error',
                            'message': f'Invalid JSON: {str(e)}'
                        }
                        client_socket.send((json.dumps(error_response) + '\n').encode('utf-8'))
        
        except Exception as e:
            print(f"Error handling client {address}: {e}")
        
        finally:
            client_socket.close()
            print(f"Client disconnected: {address}")
    
    def start(self):
        """Start the socket server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)
        self.running = True
        
        print(f"Socket server listening on {self.host}:{self.port}")
        
        try:
            while self.running:
                client_socket, address = self.socket.accept()
                # Handle each client in a separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
        
        except KeyboardInterrupt:
            print("\nShutting down server...")
            self.stop()
    
    def stop(self):
        """Stop the socket server"""
        self.running = False
        if self.socket:
            self.socket.close()
        print("Server stopped")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/natron_v2.pt',
                       help='Path to trained model')
    parser.add_argument('--config_path', type=str, default='./config.yaml',
                       help='Path to config file')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=8888,
                       help='Port to bind to')
    
    args = parser.parse_args()
    
    # Create and start server
    server = NatronSocketServer(
        model_path=args.model_path,
        config_path=args.config_path,
        host=args.host,
        port=args.port
    )
    
    try:
        server.start()
    except KeyboardInterrupt:
        server.stop()


if __name__ == '__main__':
    main()
