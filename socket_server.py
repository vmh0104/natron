"""
Socket Server for MQL5 Integration
Bidirectional TCP/JSON communication with MetaTrader 5 Expert Advisor
"""

import socket
import json
import threading
import time
import torch
import numpy as np
import pandas as pd
from datetime import datetime
import logging

from model_natron import NatronTransformer
from feature_engine import FeatureEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('natron_socket.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class NatronSocketServer:
    """
    TCP Socket Server for real-time trading with MQL5 EA
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 8888,
        model_path: str = './models/natron_v2.pt',
        sequence_length: int = 96
    ):
        self.host = host
        self.port = port
        self.sequence_length = sequence_length
        self.model = None
        self.feature_engine = None
        self.feature_columns = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load model
        self._load_model(model_path)
        
        # Socket
        self.socket = None
        self.running = False
        self.client_socket = None
        self.client_address = None
        
        # Buffer for incoming candles
        self.candle_buffer = []
    
    def _load_model(self, model_path: str):
        """Load trained model"""
        try:
            logger.info(f"Loading model from {model_path}...")
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get model configuration
            num_features = checkpoint.get('num_features', 100)
            feature_columns = checkpoint.get('feature_columns', None)
            
            if feature_columns is None:
                logger.warning("feature_columns not found, inferring from dummy data...")
                # Infer feature columns
                dummy_df = pd.DataFrame({
                    'time': pd.date_range('2024-01-01', periods=200, freq='15min'),
                    'open': np.random.randn(200),
                    'high': np.random.randn(200),
                    'low': np.random.randn(200),
                    'close': np.random.randn(200),
                    'volume': np.random.randn(200)
                })
                from dataset_loader import SequenceCreator
                creator = SequenceCreator()
                _, _, _ = creator.create_dataset(dummy_df)
                feature_columns = creator.feature_columns
                num_features = len(feature_columns)
            
            self.feature_columns = feature_columns
            
            # Load model
            self.model = NatronTransformer(
                num_features=num_features,
                d_model=256,
                nhead=8,
                num_layers=6,
                dim_feedforward=1024,
                dropout=0.1,
                sequence_length=self.sequence_length
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.feature_engine = FeatureEngine()
            
            logger.info("Model loaded successfully!")
            logger.info(f"Device: {self.device}, Features: {num_features}")
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _process_prediction_request(self, candles: list) -> dict:
        """Process candles and return prediction"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            
            # Ensure we have sequence_length candles
            if len(df) < self.sequence_length:
                return {'error': f'Need at least {self.sequence_length} candles'}
            
            df = df.tail(self.sequence_length).reset_index(drop=True)
            
            # Generate features
            features_df = self.feature_engine.fit_transform(df)
            feature_matrix = features_df[self.feature_columns].values.astype(np.float32)
            
            # Convert to tensor
            X = torch.FloatTensor(feature_matrix).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                result = self.model.predict(X)
            
            result['timestamp'] = datetime.now().isoformat()
            return result
        
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {'error': str(e)}
    
    def _handle_client(self, client_socket, address):
        """Handle client connection"""
        logger.info(f"Client connected: {address}")
        self.client_socket = client_socket
        self.client_address = address
        
        buffer = ""
        
        try:
            while self.running:
                # Receive data
                data = client_socket.recv(4096).decode('utf-8')
                
                if not data:
                    break
                
                buffer += data
                
                # Process complete JSON messages (separated by newlines)
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    
                    if not line.strip():
                        continue
                    
                    try:
                        # Parse JSON message
                        message = json.loads(line)
                        msg_type = message.get('type', '')
                        
                        if msg_type == 'predict':
                            # Prediction request
                            candles = message.get('candles', [])
                            result = self._process_prediction_request(candles)
                            
                            # Send response
                            response = {
                                'type': 'prediction',
                                'data': result,
                                'timestamp': datetime.now().isoformat()
                            }
                            client_socket.send((json.dumps(response) + '\n').encode('utf-8'))
                        
                        elif msg_type == 'ping':
                            # Heartbeat
                            response = {
                                'type': 'pong',
                                'timestamp': datetime.now().isoformat()
                            }
                            client_socket.send((json.dumps(response) + '\n').encode('utf-8'))
                        
                        elif msg_type == 'status':
                            # Status request
                            response = {
                                'type': 'status',
                                'status': 'ready',
                                'model_loaded': self.model is not None,
                                'device': self.device,
                                'buffer_size': len(self.candle_buffer)
                            }
                            client_socket.send((json.dumps(response) + '\n').encode('utf-8'))
                        
                        else:
                            logger.warning(f"Unknown message type: {msg_type}")
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON decode error: {e}, line: {line[:100]}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        error_response = {
                            'type': 'error',
                            'message': str(e)
                        }
                        client_socket.send((json.dumps(error_response) + '\n').encode('utf-8'))
        
        except Exception as e:
            logger.error(f"Client connection error: {e}")
        finally:
            logger.info(f"Client disconnected: {address}")
            client_socket.close()
            self.client_socket = None
    
    def start(self):
        """Start socket server"""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.host, self.port))
        self.socket.listen(5)
        self.running = True
        
        logger.info(f"Natron Socket Server started on {self.host}:{self.port}")
        logger.info("Waiting for MQL5 EA connection...")
        
        try:
            while self.running:
                client_socket, address = self.socket.accept()
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket, address)
                )
                client_thread.daemon = True
                client_thread.start()
        
        except KeyboardInterrupt:
            logger.info("Shutting down server...")
        finally:
            self.stop()
    
    def stop(self):
        """Stop socket server"""
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        if self.socket:
            self.socket.close()
        logger.info("Server stopped")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron Socket Server for MQL5')
    parser.add_argument('--host', type=str, default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=8888, help='Server port')
    parser.add_argument('--model', type=str, default='./models/natron_v2.pt', help='Model path')
    parser.add_argument('--sequence_length', type=int, default=96, help='Sequence length')
    
    args = parser.parse_args()
    
    server = NatronSocketServer(
        host=args.host,
        port=args.port,
        model_path=args.model,
        sequence_length=args.sequence_length
    )
    
    try:
        server.start()
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise


if __name__ == '__main__':
    main()
