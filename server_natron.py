"""
Natron Inference Server
Flask API + Socket server for realtime MQL5 integration.
"""

import os
import json
import socket
import threading
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from flask_cors import CORS
import yaml
from typing import Dict, Optional

from feature_engine import FeatureEngine
from dataset_loader import SequenceCreator
from model_natron import NatronTransformer


class NatronInferenceServer:
    """Inference server for Natron Transformer"""
    
    def __init__(self, config_path: str = 'config.yaml', model_path: str = 'model/natron_v2.pt'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model
        self.model = NatronTransformer(
            input_dim=self.config['model']['input_dim'],
            sequence_length=self.config['data']['sequence_length'],
            d_model=self.config['model']['d_model'],
            nhead=self.config['model']['nhead'],
            num_layers=self.config['model']['num_layers'],
            dim_feedforward=self.config['model']['dim_feedforward'],
            dropout=0.0  # No dropout during inference
        ).to(self.device)
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Loaded model from {model_path}")
        
        # Load scaler
        self.sequence_creator = SequenceCreator(
            sequence_length=self.config['data']['sequence_length'],
            feature_dim=self.config['model']['input_dim']
        )
        self.sequence_creator.load_scaler('model/scaler.pkl')
        
        # Feature engine
        self.feature_engine = FeatureEngine()
        
        # Regime names
        self.regime_names = {
            0: 'BULL_STRONG',
            1: 'BULL_WEAK',
            2: 'RANGE',
            3: 'BEAR_WEAK',
            4: 'BEAR_STRONG',
            5: 'VOLATILE'
        }
    
    def predict(self, candles: list) -> Dict:
        """
        Predict from last 96 OHLCV candles.
        
        Args:
            candles: List of dicts with ['time', 'open', 'high', 'low', 'close', 'volume']
        
        Returns:
            Dict with predictions
        """
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        
        # Ensure we have enough candles
        if len(df) < self.config['data']['sequence_length']:
            raise ValueError(f"Need at least {self.config['data']['sequence_length']} candles, got {len(df)}")
        
        # Take last N candles
        df = df.tail(self.config['data']['sequence_length'])
        
        # Generate features
        features_df = self.feature_engine.generate_all_features(df)
        
        # Scale features
        feature_values = features_df.values.astype(np.float32)
        feature_scaled = self.sequence_creator.transform_features(feature_values)
        
        # Create sequence (add batch dimension)
        sequence = torch.FloatTensor(feature_scaled).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            predictions = self.model(sequence)
        
        # Extract predictions
        buy_prob = predictions['buy'].item()
        sell_prob = predictions['sell'].item()
        direction_logits = predictions['direction']
        regime_logits = predictions['regime']
        
        direction_up_prob = torch.exp(direction_logits[0, 1]).item()
        regime_idx = regime_logits[0].argmax().item()
        regime_name = self.regime_names[regime_idx]
        regime_confidence = torch.exp(regime_logits[0, regime_idx]).item()
        
        # Overall confidence (average of probabilities)
        confidence = (buy_prob + sell_prob + direction_up_prob + regime_confidence) / 4
        
        return {
            'buy_prob': float(buy_prob),
            'sell_prob': float(sell_prob),
            'direction_up': float(direction_up_prob),
            'regime': regime_name,
            'regime_id': int(regime_idx),
            'confidence': float(confidence)
        }


# Flask app
app = Flask(__name__)
CORS(app)

# Global server instance
server_instance: Optional[NatronInferenceServer] = None


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """Flask endpoint for predictions"""
    try:
        data = request.json
        
        if 'candles' not in data:
            return jsonify({'error': 'Missing "candles" field'}), 400
        
        candles = data['candles']
        
        if not isinstance(candles, list) or len(candles) == 0:
            return jsonify({'error': 'Invalid candles format'}), 400
        
        predictions = server_instance.predict(candles)
        return jsonify(predictions)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'device': str(server_instance.device)})


def socket_server_thread(server_instance: NatronInferenceServer, host: str = 'localhost', port: int = 8888):
    """Socket server for MQL5 communication"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(5)
    
    print(f"Socket server listening on {host}:{port}")
    
    while True:
        try:
            client_socket, address = sock.accept()
            print(f"Connected to MQL5 client: {address}")
            
            # Handle client in separate thread
            threading.Thread(
                target=handle_mql5_client,
                args=(client_socket, server_instance),
                daemon=True
            ).start()
        
        except Exception as e:
            print(f"Socket server error: {e}")


def handle_mql5_client(client_socket: socket.socket, server_instance: NatronInferenceServer):
    """Handle MQL5 client requests"""
    try:
        while True:
            # Receive data
            data = client_socket.recv(8192)
            if not data:
                break
            
            try:
                # Parse JSON
                request_data = json.loads(data.decode('utf-8'))
                
                if request_data.get('action') == 'predict':
                    candles = request_data.get('candles', [])
                    predictions = server_instance.predict(candles)
                    
                    # Send response
                    response = json.dumps(predictions).encode('utf-8')
                    client_socket.send(response)
                
                elif request_data.get('action') == 'ping':
                    client_socket.send(json.dumps({'status': 'pong'}).encode('utf-8'))
            
            except json.JSONDecodeError:
                client_socket.send(json.dumps({'error': 'Invalid JSON'}).encode('utf-8'))
            except Exception as e:
                client_socket.send(json.dumps({'error': str(e)}).encode('utf-8'))
    
    except Exception as e:
        print(f"Client handler error: {e}")
    finally:
        client_socket.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron Inference Server')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    parser.add_argument('--model', type=str, default='model/natron_v2.pt', help='Model path')
    parser.add_argument('--flask-port', type=int, default=5000, help='Flask port')
    parser.add_argument('--socket-port', type=int, default=8888, help='Socket port')
    parser.add_argument('--socket-host', type=str, default='localhost', help='Socket host')
    args = parser.parse_args()
    
    global server_instance
    server_instance = NatronInferenceServer(config_path=args.config, model_path=args.model)
    
    # Start socket server in background
    socket_thread = threading.Thread(
        target=socket_server_thread,
        args=(server_instance, args.socket_host, args.socket_port),
        daemon=True
    )
    socket_thread.start()
    
    # Start Flask app
    print(f"Flask API server starting on port {args.flask_port}")
    print(f"Socket server starting on {args.socket_host}:{args.socket_port}")
    app.run(host='0.0.0.0', port=args.flask_port, threaded=True)


if __name__ == '__main__':
    main()
