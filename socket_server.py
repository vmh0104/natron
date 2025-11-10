"""
Natron Socket Server - TCP/JSON Server for MQL5 Integration

Handles bidirectional communication with MetaTrader 5 Expert Advisor.
Protocol: JSON over TCP
"""

import socket
import json
import threading
import time
import torch
import numpy as np
import pandas as pd
from typing import Optional, Dict
import logging

from feature_engine import FeatureEngine
from model_natron import create_natron_model
from api_server import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NatronSocketServer:
    """TCP socket server for MQL5 communication."""
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 8888,
        model_path: str = './checkpoints/natron_v2.pt',
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.host = host
        self.port = port
        self.device = torch.device(device)
        self.model_path = model_path
        
        self.model = None
        self.feature_engine = FeatureEngine()
        self.regime_names = ['BULL_STRONG', 'BULL_WEAK', 'RANGE', 'BEAR_WEAK', 'BEAR_STRONG', 'VOLATILE']
        
        self.socket = None
        self.running = False
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load trained model."""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device)
            self.model = create_natron_model(
                input_dim=100,
                d_model=256,
                nhead=8,
                num_layers=6,
                dim_feedforward=1024,
                dropout=0.1,
                max_seq_len=96
            ).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            logger.info(f'Model loaded from {self.model_path}')
        except Exception as e:
            logger.error(f'Failed to load model: {e}')
            raise
    
    def _process_request(self, data: Dict) -> Dict:
        """
        Process incoming request from MQL5.
        
        Expected format:
        {
            "type": "predict",
            "candles": [...]
        }
        
        Returns:
        {
            "type": "prediction",
            "buy_prob": 0.71,
            "sell_prob": 0.24,
            ...
        }
        """
        try:
            req_type = data.get('type', '')
            
            if req_type == 'predict':
                candles = data.get('candles', [])
                
                if len(candles) < 96:
                    return {
                        'type': 'error',
                        'message': f'Need at least 96 candles, got {len(candles)}'
                    }
                
                # Convert to DataFrame
                df = pd.DataFrame(candles)
                
                # Generate features
                features_df = self.feature_engine.fit_transform(df)
                features_array = features_df.values.astype(np.float32)
                features_tensor = torch.FloatTensor(features_array).unsqueeze(0).to(self.device)
                
                # Predict
                with torch.no_grad():
                    predictions = self.model.predict(features_tensor)
                
                # Format response
                buy_prob = predictions['buy'].item()
                sell_prob = predictions['sell'].item()
                
                direction_logprobs = predictions['direction']
                direction_probs = torch.exp(direction_logprobs)
                direction_up_prob = direction_probs[0][1].item()
                
                regime_logprobs = predictions['regime']
                regime_probs = torch.exp(regime_logprobs)
                regime_id = regime_probs.argmax().item()
                regime_name = self.regime_names[regime_id] if regime_id < len(self.regime_names) else 'UNKNOWN'
                
                return {
                    'type': 'prediction',
                    'buy_prob': round(buy_prob, 4),
                    'sell_prob': round(sell_prob, 4),
                    'direction_up': round(direction_up_prob, 4),
                    'regime': regime_name,
                    'regime_id': int(regime_id),
                    'timestamp': time.time()
                }
            
            elif req_type == 'ping':
                return {'type': 'pong', 'timestamp': time.time()}
            
            else:
                return {'type': 'error', 'message': f'Unknown request type: {req_type}'}
        
        except Exception as e:
            logger.error(f'Error processing request: {e}')
            return {'type': 'error', 'message': str(e)}
    
    def _handle_client(self, client_socket: socket.socket, address: tuple):
        """Handle client connection."""
        logger.info(f'Client connected: {address}')
        
        try:
            buffer = b''
            while self.running:
                # Receive data
                data = client_socket.recv(4096)
                if not data:
                    break
                
                buffer += data
                
                # Try to parse JSON (messages may be split)
                try:
                    # Look for complete JSON objects
                    while buffer:
                        # Try to find JSON boundaries
                        try:
                            # Attempt to decode
                            message_str = buffer.decode('utf-8')
                            
                            # Try to parse JSON
                            # Simple approach: assume one JSON per message
                            try:
                                data_dict = json.loads(message_str.strip())
                                buffer = b''  # Clear buffer after successful parse
                                
                                # Process request
                                response = self._process_request(data_dict)
                                
                                # Send response
                                response_json = json.dumps(response) + '\n'
                                client_socket.send(response_json.encode('utf-8'))
                                
                            except json.JSONDecodeError:
                                # Incomplete JSON, wait for more data
                                break
                        
                        except UnicodeDecodeError:
                            # Invalid UTF-8, skip
                            buffer = b''
                            break
                
                except Exception as e:
                    logger.error(f'Error handling message: {e}')
                    error_response = {'type': 'error', 'message': str(e)}
                    client_socket.send(json.dumps(error_response).encode('utf-8'))
        
        except Exception as e:
            logger.error(f'Error in client handler: {e}')
        
        finally:
            client_socket.close()
            logger.info(f'Client disconnected: {address}')
    
    def start(self):
        """Start the socket server."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)
        self.running = True
        
        logger.info(f'Socket server listening on {self.host}:{self.port}')
        
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
            logger.info('Shutting down server...')
        finally:
            self.stop()
    
    def stop(self):
        """Stop the server."""
        self.running = False
        if self.socket:
            self.socket.close()
        logger.info('Server stopped')


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron Socket Server for MQL5')
    parser.add_argument('--host', type=str, default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8888, help='Port to bind to')
    parser.add_argument('--model_path', type=str, default='./checkpoints/natron_v2.pt', help='Path to model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    
    args = parser.parse_args()
    
    server = NatronSocketServer(
        host=args.host,
        port=args.port,
        model_path=args.model_path,
        device=args.device
    )
    
    server.start()


if __name__ == '__main__':
    main()
