"""
Socket Server for MQL5 Integration
Receives OHLCV data from MetaTrader 5 and returns trading signals
"""
import socket
import json
import threading
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from config import Config
from feature_engine import FeatureEngine
from model import NatronTransformer
from label_generator import LabelGenerator


class MQL5SocketServer:
    """Socket server for MQL5 Expert Advisor communication"""
    
    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        if not torch.cuda.is_available() and self.device.type == 'cuda':
            self.device = torch.device('cpu')
        
        self.model = None
        self.feature_engine = FeatureEngine()
        self.label_generator = LabelGenerator()
        self.load_model()
        
        self.socket = None
        self.running = False
    
    def load_model(self):
        """Load trained model"""
        self.model = NatronTransformer(
            num_features=Config.NUM_FEATURES,
            d_model=Config.D_MODEL,
            n_heads=Config.N_HEADS,
            n_layers=Config.N_LAYERS,
            d_ff=Config.D_FF,
            dropout=Config.DROPOUT,
            sequence_length=Config.SEQUENCE_LENGTH
        ).to(self.device)
        
        if Config.MODEL_PATH.exists():
            self.model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=self.device))
            print(f"✓ Loaded model from {Config.MODEL_PATH}")
        else:
            print(f"⚠ Warning: Model not found at {Config.MODEL_PATH}")
        
        self.model.eval()
    
    def process_request(self, data: dict) -> dict:
        """
        Process trading request from MQL5
        
        Expected format:
        {
            "action": "predict",
            "candles": [
                {"time": "...", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 1000},
                ...
            ]
        }
        
        Returns:
        {
            "status": "success",
            "buy_prob": 0.71,
            "sell_prob": 0.24,
            "direction_up": 0.69,
            "regime": "BULL_WEAK",
            "confidence": 0.82
        }
        """
        try:
            action = data.get('action', 'predict')
            
            if action == 'predict':
                candles = data.get('candles', [])
                
                if len(candles) < Config.SEQUENCE_LENGTH:
                    return {
                        'status': 'error',
                        'message': f'Need at least {Config.SEQUENCE_LENGTH} candles, got {len(candles)}'
                    }
                
                # Convert to DataFrame
                df = pd.DataFrame(candles)
                
                # Take last 96 candles
                df = df.tail(Config.SEQUENCE_LENGTH).reset_index(drop=True)
                
                # Generate features
                features_df = self.feature_engine.generate_features(df)
                
                # Convert to tensor
                features_array = features_df.values.astype(np.float32)
                features_tensor = torch.FloatTensor(features_array).unsqueeze(0).to(self.device)
                
                # Predict
                with torch.no_grad():
                    predictions = self.model(features_tensor)
                
                # Extract probabilities
                buy_prob = predictions['buy'][0].item()
                sell_prob = predictions['sell'][0].item()
                direction_probs = torch.softmax(predictions['direction'][0], dim=0)
                direction_up_prob = direction_probs[1].item()
                regime_probs = torch.softmax(predictions['regime'][0], dim=0)
                regime_idx = regime_probs.argmax().item()
                regime_name = self.label_generator.regime_names[regime_idx]
                regime_confidence = regime_probs[regime_idx].item()
                
                # Overall confidence
                confidence = (buy_prob if buy_prob > sell_prob else sell_prob + 
                            direction_probs.max().item() + regime_confidence) / 3
                
                return {
                    'status': 'success',
                    'buy_prob': round(buy_prob, 4),
                    'sell_prob': round(sell_prob, 4),
                    'direction_up': round(direction_up_prob, 4),
                    'regime': regime_name,
                    'regime_prob': round(regime_confidence, 4),
                    'confidence': round(confidence, 4)
                }
            
            elif action == 'health':
                return {'status': 'success', 'model_loaded': self.model is not None}
            
            else:
                return {'status': 'error', 'message': f'Unknown action: {action}'}
        
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def handle_client(self, client_socket, address):
        """Handle client connection"""
        print(f"Client connected: {address}")
        
        try:
            while self.running:
                # Receive data
                data = client_socket.recv(8192)
                if not data:
                    break
                
                # Parse JSON
                try:
                    request = json.loads(data.decode('utf-8'))
                except json.JSONDecodeError:
                    response = {'status': 'error', 'message': 'Invalid JSON'}
                    client_socket.send(json.dumps(response).encode('utf-8'))
                    continue
                
                # Process request
                response = self.process_request(request)
                
                # Send response
                response_json = json.dumps(response)
                client_socket.send(response_json.encode('utf-8'))
        
        except Exception as e:
            print(f"Error handling client {address}: {e}")
        
        finally:
            client_socket.close()
            print(f"Client disconnected: {address}")
    
    def start(self):
        """Start socket server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.socket.bind((Config.SOCKET_HOST, Config.SOCKET_PORT))
            self.socket.listen(5)
            self.running = True
            
            print(f"="*70)
            print(f"NATRON MQL5 SOCKET SERVER")
            print(f"="*70)
            print(f"Listening on {Config.SOCKET_HOST}:{Config.SOCKET_PORT}")
            print(f"Model loaded: {self.model is not None}")
            print(f"Ready to receive requests from MQL5 Expert Advisor...")
            print(f"="*70)
            
            while self.running:
                try:
                    client_socket, address = self.socket.accept()
                    # Handle in separate thread
                    thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address)
                    )
                    thread.daemon = True
                    thread.start()
                except Exception as e:
                    if self.running:
                        print(f"Error accepting connection: {e}")
        
        except KeyboardInterrupt:
            print("\nShutting down server...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop socket server"""
        self.running = False
        if self.socket:
            self.socket.close()
        print("Server stopped")


def main():
    """Main entry point"""
    server = MQL5SocketServer()
    server.start()


if __name__ == '__main__':
    main()
