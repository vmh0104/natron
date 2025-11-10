"""
Socket Server for MQL5 Expert Advisor Communication
"""
import socket
import json
import threading
import numpy as np
import pandas as pd
import torch
from model import NatronModel
from feature_engine import FeatureEngine
import os
from typing import Dict, List


class MQL5SocketServer:
    """Socket server for real-time MQL5 integration"""
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 8888,
        model_path: str = 'models/natron_v2.pt'
    ):
        self.host = host
        self.port = port
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self.model = None
        self.feature_engine = FeatureEngine()
        self.load_model()
        
        # Buffer for candle data
        self.candle_buffer = []
        self.buffer_lock = threading.Lock()
    
    def load_model(self):
        """Load the trained Natron model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        self.model = NatronModel(
            input_dim=100,
            d_model=256,
            nhead=8,
            num_layers=6
        ).to(self.device)
        
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Model loaded from {self.model_path}")
    
    def prepare_sequence(self, ohlcv_data: List[Dict]) -> np.ndarray:
        """Prepare input sequence from OHLCV data"""
        df = pd.DataFrame(ohlcv_data)
        
        if len(df) < 96:
            # Pad with zeros if needed
            padding_needed = 96 - len(df)
            padding_df = pd.DataFrame({
                'time': [df['time'].iloc[0]] * padding_needed,
                'open': [df['open'].iloc[0]] * padding_needed,
                'high': [df['high'].iloc[0]] * padding_needed,
                'low': [df['low'].iloc[0]] * padding_needed,
                'close': [df['close'].iloc[0]] * padding_needed,
                'volume': [0] * padding_needed
            })
            df = pd.concat([padding_df, df], ignore_index=True)
        
        df = df.tail(96).reset_index(drop=True)
        
        # Generate features
        df_features = self.feature_engine.generate_features(df)
        feature_columns = self.feature_engine.get_feature_columns(df_features)
        feature_matrix = df_features[feature_columns].values.astype(np.float32)
        
        # Normalize
        feature_mean = np.nanmean(feature_matrix, axis=0, keepdims=True)
        feature_std = np.nanstd(feature_matrix, axis=0, keepdims=True) + 1e-8
        feature_matrix = (feature_matrix - feature_mean) / feature_std
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        
        return feature_matrix
    
    def predict(self, candles: List[Dict]) -> Dict:
        """Make prediction on candle data"""
        sequence = self.prepare_sequence(candles)
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_tensor)
        
        buy_prob = float(outputs['buy'].item())
        sell_prob = float(outputs['sell'].item())
        direction_probs = outputs['direction'].cpu().numpy()[0]
        regime_probs = outputs['regime'].cpu().numpy()[0]
        regime_id = int(np.argmax(regime_probs))
        
        regime_names = [
            'BULL_STRONG', 'BULL_WEAK', 'RANGE',
            'BEAR_WEAK', 'BEAR_STRONG', 'VOLATILE'
        ]
        
        return {
            'buy_prob': round(buy_prob, 4),
            'sell_prob': round(sell_prob, 4),
            'direction_up': round(float(direction_probs[1]), 4),
            'direction_down': round(float(direction_probs[0]), 4),
            'regime': regime_names[regime_id],
            'regime_id': regime_id,
            'confidence': round(max(buy_prob, sell_prob, direction_probs.max(), regime_probs.max()), 4)
        }
    
    def handle_client(self, client_socket: socket.socket, address: tuple):
        """Handle client connection"""
        print(f"Client connected: {address}")
        
        try:
            while True:
                # Receive data
                data = client_socket.recv(8192)
                if not data:
                    break
                
                try:
                    # Parse JSON message
                    message = json.loads(data.decode('utf-8'))
                    msg_type = message.get('type', '')
                    
                    if msg_type == 'PREDICT':
                        # Predict on candle data
                        candles = message.get('candles', [])
                        
                        if len(candles) < 96:
                            response = {
                                'type': 'ERROR',
                                'message': f'Need at least 96 candles, got {len(candles)}'
                            }
                        else:
                            prediction = self.predict(candles)
                            response = {
                                'type': 'PREDICTION',
                                'data': prediction
                            }
                        
                        # Send response
                        response_json = json.dumps(response)
                        client_socket.send(response_json.encode('utf-8'))
                    
                    elif msg_type == 'ADD_CANDLE':
                        # Add candle to buffer
                        candle = message.get('candle', {})
                        with self.buffer_lock:
                            self.candle_buffer.append(candle)
                            # Keep only last 96 candles
                            if len(self.candle_buffer) > 96:
                                self.candle_buffer.pop(0)
                        
                        # If buffer has 96 candles, predict
                        if len(self.candle_buffer) >= 96:
                            prediction = self.predict(self.candle_buffer)
                            response = {
                                'type': 'PREDICTION',
                                'data': prediction
                            }
                            response_json = json.dumps(response)
                            client_socket.send(response_json.encode('utf-8'))
                        else:
                            response = {
                                'type': 'ACK',
                                'message': f'Candle added. Buffer size: {len(self.candle_buffer)}'
                            }
                            response_json = json.dumps(response)
                            client_socket.send(response_json.encode('utf-8'))
                    
                    elif msg_type == 'GET_PREDICTION':
                        # Get prediction from current buffer
                        with self.buffer_lock:
                            if len(self.candle_buffer) >= 96:
                                prediction = self.predict(self.candle_buffer)
                                response = {
                                    'type': 'PREDICTION',
                                    'data': prediction
                                }
                            else:
                                response = {
                                    'type': 'ERROR',
                                    'message': f'Buffer has only {len(self.candle_buffer)} candles, need 96'
                                }
                        
                        response_json = json.dumps(response)
                        client_socket.send(response_json.encode('utf-8'))
                    
                    elif msg_type == 'PING':
                        # Health check
                        response = {
                            'type': 'PONG',
                            'status': 'alive'
                        }
                        response_json = json.dumps(response)
                        client_socket.send(response_json.encode('utf-8'))
                    
                    else:
                        response = {
                            'type': 'ERROR',
                            'message': f'Unknown message type: {msg_type}'
                        }
                        response_json = json.dumps(response)
                        client_socket.send(response_json.encode('utf-8'))
                
                except json.JSONDecodeError:
                    response = {
                        'type': 'ERROR',
                        'message': 'Invalid JSON'
                    }
                    response_json = json.dumps(response)
                    client_socket.send(response_json.encode('utf-8'))
                
                except Exception as e:
                    response = {
                        'type': 'ERROR',
                        'message': str(e)
                    }
                    response_json = json.dumps(response)
                    client_socket.send(response_json.encode('utf-8'))
        
        except Exception as e:
            print(f"Error handling client {address}: {e}")
        
        finally:
            client_socket.close()
            print(f"Client disconnected: {address}")
    
    def start(self):
        """Start the socket server"""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.host, self.port))
        server_socket.listen(5)
        
        print(f"Socket server listening on {self.host}:{self.port}")
        print("Waiting for MQL5 Expert Advisor connections...")
        
        try:
            while True:
                client_socket, address = server_socket.accept()
                # Handle each client in a separate thread
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
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='localhost',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=8888,
                       help='Port to bind to')
    parser.add_argument('--model_path', type=str, default='models/natron_v2.pt',
                       help='Path to trained model')
    
    args = parser.parse_args()
    
    server = MQL5SocketServer(
        host=args.host,
        port=args.port,
        model_path=args.model_path
    )
    
    server.start()
