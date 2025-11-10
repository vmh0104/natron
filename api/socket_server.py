"""
Natron Transformer - Socket Server
Real-time TCP socket server for MQL5 integration
"""

import socket
import json
import threading
import time
import pandas as pd
import numpy as np
from typing import Dict
import yaml
import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from inference import NatronInference


class NatronSocketServer:
    """
    TCP Socket server for real-time communication with MQL5
    Protocol: JSON over TCP
    """
    
    def __init__(
        self,
        config_path: str = 'config.yaml',
        model_path: str = None
    ):
        """
        Args:
            config_path: Path to configuration file
            model_path: Path to model weights
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Socket config
        self.host = self.config['socket']['host']
        self.port = self.config['socket']['port']
        self.max_connections = self.config['socket']['max_connections']
        self.buffer_size = self.config['socket']['buffer_size']
        self.heartbeat_interval = self.config['socket']['heartbeat_interval']
        
        # Initialize inference engine
        if model_path is None:
            model_path = self.config['inference']['model_path']
        
        device = 'cuda' if self.config['inference']['device'] == 'cuda' else 'cpu'
        self.inference_engine = NatronInference(model_path, config_path, device)
        
        # Server state
        self.server_socket = None
        self.running = False
        self.clients = []
        self.request_count = 0
        
        print(f"🔧 Socket Server initialized")
        print(f"   Host: {self.host}")
        print(f"   Port: {self.port}")
        print(f"   Max connections: {self.max_connections}")
    
    def start(self):
        """Start the socket server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        try:
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(self.max_connections)
            self.running = True
            
            print(f"\n🚀 Socket Server started")
            print(f"   Listening on {self.host}:{self.port}")
            print(f"   Waiting for MQL5 connections...")
            
            # Start heartbeat thread
            heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
            heartbeat_thread.start()
            
            # Accept connections
            while self.running:
                try:
                    client_socket, address = self.server_socket.accept()
                    print(f"\n📡 New connection from {address}")
                    
                    # Handle client in separate thread
                    client_thread = threading.Thread(
                        target=self._handle_client,
                        args=(client_socket, address),
                        daemon=True
                    )
                    client_thread.start()
                    
                except Exception as e:
                    if self.running:
                        print(f"❌ Error accepting connection: {e}")
        
        except Exception as e:
            print(f"❌ Failed to start server: {e}")
        
        finally:
            self.stop()
    
    def _handle_client(self, client_socket: socket.socket, address: tuple):
        """
        Handle client connection
        
        Args:
            client_socket: Client socket
            address: Client address
        """
        self.clients.append((client_socket, address))
        
        try:
            while self.running:
                # Receive data
                data = client_socket.recv(self.buffer_size)
                
                if not data:
                    break
                
                # Parse request
                try:
                    request = json.loads(data.decode('utf-8'))
                    
                    # Process request
                    response = self._process_request(request)
                    
                    # Send response
                    response_json = json.dumps(response)
                    client_socket.send(response_json.encode('utf-8'))
                    
                    self.request_count += 1
                
                except json.JSONDecodeError as e:
                    error_response = {
                        'success': False,
                        'error': f'Invalid JSON: {str(e)}'
                    }
                    client_socket.send(json.dumps(error_response).encode('utf-8'))
                
                except Exception as e:
                    error_response = {
                        'success': False,
                        'error': f'Processing error: {str(e)}'
                    }
                    client_socket.send(json.dumps(error_response).encode('utf-8'))
        
        except Exception as e:
            print(f"❌ Client handler error: {e}")
        
        finally:
            print(f"🔌 Client disconnected: {address}")
            self.clients.remove((client_socket, address))
            client_socket.close()
    
    def _process_request(self, request: Dict) -> Dict:
        """
        Process client request
        
        Request format:
        {
            "type": "predict",
            "data": [
                {"time": "...", "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 5000},
                ...
            ]
        }
        
        Response format:
        {
            "success": true,
            "prediction": {...},
            "request_id": 123,
            "timestamp": "..."
        }
        
        Args:
            request: Request dictionary
            
        Returns:
            Response dictionary
        """
        request_type = request.get('type', 'unknown')
        
        if request_type == 'predict':
            # Prediction request
            data = request.get('data', [])
            
            if len(data) < self.inference_engine.sequence_length:
                return {
                    'success': False,
                    'error': f'Need at least {self.inference_engine.sequence_length} candles'
                }
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Predict
            prediction = self.inference_engine.predict(df)
            
            return {
                'success': True,
                'prediction': prediction,
                'request_id': self.request_count,
                'timestamp': time.time()
            }
        
        elif request_type == 'ping':
            # Health check
            return {
                'success': True,
                'type': 'pong',
                'timestamp': time.time()
            }
        
        elif request_type == 'status':
            # Server status
            return {
                'success': True,
                'status': {
                    'running': self.running,
                    'clients': len(self.clients),
                    'requests': self.request_count,
                    'model': 'Natron Transformer v2',
                    'device': self.inference_engine.device
                },
                'timestamp': time.time()
            }
        
        else:
            return {
                'success': False,
                'error': f'Unknown request type: {request_type}'
            }
    
    def _heartbeat_loop(self):
        """Send periodic heartbeat to clients"""
        while self.running:
            time.sleep(self.heartbeat_interval)
            
            # Send heartbeat to all clients
            for client_socket, address in self.clients[:]:  # Copy list to avoid modification during iteration
                try:
                    heartbeat = {
                        'type': 'heartbeat',
                        'timestamp': time.time()
                    }
                    client_socket.send(json.dumps(heartbeat).encode('utf-8'))
                except:
                    pass  # Client may have disconnected
    
    def stop(self):
        """Stop the server"""
        print("\n🛑 Stopping socket server...")
        self.running = False
        
        # Close all client connections
        for client_socket, address in self.clients:
            try:
                client_socket.close()
            except:
                pass
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        print("✅ Socket server stopped")


def main():
    """Main function to run the socket server"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron Socket Server')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--model', type=str, default=None, help='Model file path')
    parser.add_argument('--host', type=str, default=None, help='Host address')
    parser.add_argument('--port', type=int, default=None, help='Port number')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override host/port if provided
    if args.host:
        config['socket']['host'] = args.host
    if args.port:
        config['socket']['port'] = args.port
    
    # Save modified config
    with open(args.config, 'w') as f:
        yaml.dump(config, f)
    
    # Create and start server
    try:
        server = NatronSocketServer(args.config, args.model)
        server.start()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")


if __name__ == '__main__':
    main()
