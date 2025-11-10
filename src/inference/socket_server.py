"""
Natron V2 Socket Server
Real-time TCP socket server for MQL5 communication
"""

import socket
import json
import torch
import numpy as np
import yaml
import sys
import os
import threading
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.natron_transformer import NatronTransformer
from src.data.dataset_loader import DataProcessor
from src.labels.label_generator import LabelGenerator


class NatronSocketServer:
    """
    Socket server for real-time communication with MQL5 EA.
    
    Protocol:
    - Client sends JSON request with 96 OHLCV candles
    - Server responds with JSON prediction
    """
    
    def __init__(self, config_path: str = 'config/config.yaml', 
                 model_path: str = 'model/natron_v2.pt'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.host = '0.0.0.0'
        self.port = self.config['inference']['socket_port']
        
        # Set device
        self.device = torch.device(self.config['deployment']['device'] 
                                   if torch.cuda.is_available() 
                                   else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = NatronTransformer(self.config).to(self.device)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            print(f"✓ Model loaded from {model_path}")
        else:
            print(f"⚠ Model not found at {model_path}")
        
        # Data processor
        self.data_processor = DataProcessor(self.config)
        scaler_path = 'model/scaler.pkl'
        if os.path.exists(scaler_path):
            self.data_processor.load_scaler(scaler_path)
            print(f"✓ Scaler loaded from {scaler_path}")
        
        # Label generator
        self.label_generator = LabelGenerator()
        
        # Socket
        self.server_socket = None
        self.running = False
        
        # Statistics
        self.total_requests = 0
        self.total_errors = 0
        self.start_time = datetime.now()
    
    def start(self):
        """Start the socket server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        
        self.running = True
        
        print("=" * 80)
        print(f"Natron Socket Server started on {self.host}:{self.port}")
        print("=" * 80)
        print("Waiting for connections from MQL5 EA...")
        
        try:
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    print(f"\n[{datetime.now()}] Connection from {address}")
                    
                    # Handle client in a new thread
                    client_thread = threading.Thread(
                        target=self.handle_client,
                        args=(client_socket, address)
                    )
                    client_thread.start()
                
                except KeyboardInterrupt:
                    print("\nShutting down server...")
                    break
                except Exception as e:
                    print(f"Error accepting connection: {e}")
        
        finally:
            self.stop()
    
    def handle_client(self, client_socket, address):
        """
        Handle individual client connection.
        """
        try:
            # Receive data (max 1MB)
            data = b''
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                
                # Check for end of message
                if b'\n' in chunk or len(data) > 1000000:
                    break
            
            if not data:
                return
            
            # Parse JSON request
            request = json.loads(data.decode('utf-8'))
            
            # Process prediction
            response = self.predict(request)
            
            # Send response
            response_json = json.dumps(response) + '\n'
            client_socket.sendall(response_json.encode('utf-8'))
            
            self.total_requests += 1
            
            print(f"[{datetime.now()}] Prediction sent to {address}")
            print(f"  Buy: {response.get('buy_prob', 0):.3f}, "
                  f"Sell: {response.get('sell_prob', 0):.3f}, "
                  f"Regime: {response.get('regime', 'N/A')}")
        
        except json.JSONDecodeError as e:
            error_response = {'error': 'Invalid JSON', 'details': str(e)}
            client_socket.sendall(json.dumps(error_response).encode('utf-8'))
            self.total_errors += 1
        
        except Exception as e:
            error_response = {'error': 'Prediction failed', 'details': str(e)}
            client_socket.sendall(json.dumps(error_response).encode('utf-8'))
            self.total_errors += 1
            print(f"Error: {e}")
        
        finally:
            client_socket.close()
    
    def predict(self, request: dict) -> dict:
        """
        Make prediction from request.
        
        Expected request format:
        {
            "symbol": "EURUSD",
            "timeframe": "M15",
            "data": [[time, open, high, low, close, volume], ...]  # 96 candles
        }
        """
        if 'data' not in request:
            return {'error': 'Missing "data" field'}
        
        ohlcv = np.array(request['data'])
        
        if ohlcv.shape[0] != 96:
            return {'error': f'Expected 96 candles, got {ohlcv.shape[0]}'}
        
        # Preprocess
        input_tensor = self.data_processor.preprocess_inference_data(ohlcv)
        input_tensor = input_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model.predict(input_tensor)
        
        # Format response
        response = {
            'symbol': request.get('symbol', 'UNKNOWN'),
            'timeframe': request.get('timeframe', 'UNKNOWN'),
            'timestamp': datetime.now().isoformat(),
            'buy_prob': float(predictions['buy_prob'][0].cpu().numpy()),
            'sell_prob': float(predictions['sell_prob'][0].cpu().numpy()),
            'direction_up': float(predictions['direction_prob'][0].cpu().numpy()),
            'direction_class': int(predictions['direction_class'][0].cpu().numpy()),
            'regime': self.label_generator.get_regime_name(
                int(predictions['regime_class'][0].cpu().numpy())
            ),
            'regime_class': int(predictions['regime_class'][0].cpu().numpy()),
            'confidence': float(predictions['confidence'][0].cpu().numpy())
        }
        
        return response
    
    def stop(self):
        """Stop the server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        
        uptime = (datetime.now() - self.start_time).total_seconds()
        print("\n" + "=" * 80)
        print("Server Statistics:")
        print(f"  Total Requests: {self.total_requests}")
        print(f"  Total Errors: {self.total_errors}")
        print(f"  Uptime: {uptime:.2f} seconds")
        print(f"  Avg Request Rate: {self.total_requests / uptime:.2f} req/s")
        print("=" * 80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron V2 Socket Server')
    parser.add_argument('--config', type=str, default='config/config.yaml', 
                       help='Config path')
    parser.add_argument('--model', type=str, default='model/natron_v2.pt', 
                       help='Model path')
    
    args = parser.parse_args()
    
    server = NatronSocketServer(args.config, args.model)
    
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop()


if __name__ == '__main__':
    main()
