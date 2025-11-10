"""
Natron Socket Server - Bridge between MQL5 EA and Python model
"""
import socket
import json
import threading
import numpy as np
import pandas as pd
import torch
from typing import Dict, Optional
import requests
import os

from feature_engine import FeatureEngine
from model import create_model


class NatronSocketServer:
    """Socket server for MQL5 integration"""
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 8888,
        api_url: str = 'http://localhost:5000/predict',
        model_path: Optional[str] = None
    ):
        self.host = host
        self.port = port
        self.api_url = api_url
        self.model_path = model_path
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.feature_engine = FeatureEngine()
        self.sequence_length = 96
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self._load_model(model_path)
    
    def _load_model(self, model_path: str, num_features: int = 100):
        """Load model for direct inference"""
        self.model = create_model(
            num_features=num_features,
            d_model=256,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            pretrain=False
        )
        
        checkpoint = torch.load(model_path, map_location=self.device)
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded from {model_path}")
    
    def _predict_local(self, candles: list) -> Dict:
        """Predict using local model"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        
        # Extract features
        features_df = self.feature_engine.extract_all_features(df)
        features = features_df.values.astype(np.float32)
        
        # Take last sequence_length
        if len(features) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} candles")
        
        features = features[-self.sequence_length:]
        
        # Predict
        X = torch.FloatTensor(features).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(X, mode='supervised')
            
            buy_prob = outputs['buy'].item()
            sell_prob = outputs['sell'].item()
            direction_logits = outputs['direction'][0]
            direction_probs = torch.softmax(direction_logits, dim=0)
            direction_up = direction_probs[1].item()
            
            regime_logits = outputs['regime'][0]
            regime_probs = torch.softmax(regime_logits, dim=0)
            regime_idx = regime_probs.argmax().item()
            regime_names = [
                'BULL_STRONG', 'BULL_WEAK', 'RANGE',
                'BEAR_WEAK', 'BEAR_STRONG', 'VOLATILE'
            ]
            regime = regime_names[regime_idx]
            regime_confidence = regime_probs[regime_idx].item()
        
        confidence = (buy_prob + (1 - sell_prob) + direction_up + regime_confidence) / 4
        
        return {
            'buy_prob': round(buy_prob, 4),
            'sell_prob': round(sell_prob, 4),
            'direction_up': round(direction_up, 4),
            'regime': regime,
            'confidence': round(confidence, 4)
        }
    
    def _predict_api(self, candles: list) -> Dict:
        """Predict using Flask API"""
        response = requests.post(
            self.api_url,
            json={'candles': candles},
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    
    def _handle_client(self, client_socket, address):
        """Handle client connection"""
        print(f"Connection from {address}")
        
        try:
            while True:
                # Receive data
                data = client_socket.recv(4096)
                if not data:
                    break
                
                # Parse JSON
                try:
                    request = json.loads(data.decode('utf-8'))
                except json.JSONDecodeError:
                    error_response = {'error': 'Invalid JSON'}
                    client_socket.send(json.dumps(error_response).encode('utf-8'))
                    continue
                
                # Handle request
                if request.get('action') == 'predict':
                    candles = request.get('candles', [])
                    
                    try:
                        # Use local model if available, otherwise API
                        if self.model is not None:
                            result = self._predict_local(candles)
                        else:
                            result = self._predict_api(candles)
                        
                        response = {'status': 'success', 'result': result}
                    except Exception as e:
                        response = {'status': 'error', 'error': str(e)}
                
                elif request.get('action') == 'health':
                    response = {
                        'status': 'healthy',
                        'model_loaded': self.model is not None
                    }
                
                else:
                    response = {'status': 'error', 'error': 'Unknown action'}
                
                # Send response
                client_socket.send(json.dumps(response).encode('utf-8'))
        
        except Exception as e:
            print(f"Error handling client {address}: {e}")
        finally:
            client_socket.close()
            print(f"Connection closed: {address}")
    
    def start(self):
        """Start socket server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        
        print(f"Socket server listening on {self.host}:{self.port}")
        
        try:
            while True:
                client_socket, address = server_socket.accept()
                # Handle each client in a separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
        except KeyboardInterrupt:
            print("\nShutting down socket server...")
        finally:
            server_socket.close()


if __name__ == '__main__':
    import argparse
    import os
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost', help='Host')
    parser.add_argument('--port', type=int, default=8888, help='Port')
    parser.add_argument('--api_url', type=str, default='http://localhost:5000/predict', 
                       help='Flask API URL')
    parser.add_argument('--model', type=str, default=None, 
                       help='Direct model path (optional, uses API if not provided)')
    
    args = parser.parse_args()
    
    server = NatronSocketServer(
        host=args.host,
        port=args.port,
        api_url=args.api_url,
        model_path=args.model
    )
    
    server.start()
