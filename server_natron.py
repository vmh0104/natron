"""
Natron API Server - Flask + Socket Server
Real-time inference for trading signals

Endpoints:
- POST /predict: Get trading signal from 96 OHLCV candles
- GET /health: Health check
- GET /model_info: Model information

Socket Server:
- TCP socket on port 9999 for MQL5 communication
"""

import torch
import numpy as np
import pandas as pd
import pickle
import json
import yaml
import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import socket
import threading
import time
from typing import Dict, List
import sys

# Add src to path
sys.path.append('src')

from model_natron import NatronTransformer
from dataset_loader import create_inference_sequence
from feature_engine import FeatureEngine
from label_generator import LabelGenerator


app = Flask(__name__)
CORS(app)

# Global variables
model = None
scaler = None
feature_engine = None
config = None
device = None
label_generator = None

# Regime mapping
REGIME_MAP = {
    0: "BULL_STRONG",
    1: "BULL_WEAK",
    2: "RANGE",
    3: "BEAR_WEAK",
    4: "BEAR_STRONG",
    5: "VOLATILE"
}


def load_model():
    """Load trained model and preprocessing objects"""
    global model, scaler, feature_engine, config, device, label_generator
    
    print("🔧 Loading Natron model...")
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Load scaler
    scaler_path = 'models/scaler.pkl'
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ Scaler loaded from {scaler_path}")
    else:
        print(f"⚠️  Warning: Scaler not found at {scaler_path}")
        scaler = None
    
    # Initialize feature engine
    feature_engine = FeatureEngine(verbose=False)
    label_generator = LabelGenerator(config.get('labels', {}))
    
    # Load model
    model_path = config['api']['model_path']
    if not os.path.exists(model_path):
        print(f"❌ Model not found at {model_path}")
        return False
    
    checkpoint = torch.load(model_path, map_location=device)
    
    model = NatronTransformer(
        num_features=config['data']['feature_count'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        activation=config['model']['activation'],
        buy_head_dims=config['model']['buy_head_dims'],
        sell_head_dims=config['model']['sell_head_dims'],
        direction_head_dims=config['model']['direction_head_dims'],
        regime_head_dims=config['model']['regime_head_dims'],
        sequence_length=config['data']['sequence_length']
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"✅ Model loaded from {model_path}")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return True


def predict_from_ohlcv(ohlcv_data: List[Dict]) -> Dict:
    """
    Generate prediction from OHLCV data
    
    Args:
        ohlcv_data: List of dicts with keys: time, open, high, low, close, volume
        
    Returns:
        Prediction dictionary
    """
    if model is None or scaler is None:
        return {'error': 'Model not loaded'}
    
    if len(ohlcv_data) < config['data']['sequence_length']:
        return {'error': f'Need at least {config["data"]["sequence_length"]} candles'}
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame(ohlcv_data)
        
        # Create inference sequence
        sequence = create_inference_sequence(
            df,
            feature_engine,
            scaler,
            sequence_length=config['data']['sequence_length']
        )
        
        # Move to device
        sequence = sequence.to(device)
        
        # Get predictions
        with torch.no_grad():
            predictions = model.get_predictions(sequence)
        
        # Extract values
        buy_prob = float(predictions['buy_prob'][0])
        sell_prob = float(predictions['sell_prob'][0])
        direction_prob = predictions['direction_prob'][0].cpu().numpy()
        regime_prob = predictions['regime_prob'][0].cpu().numpy()
        
        direction_up = float(direction_prob[1])
        regime_id = int(predictions['regime_pred'][0])
        regime_name = REGIME_MAP[regime_id]
        
        # Calculate confidence (max probability)
        confidence = float(max(buy_prob, sell_prob, direction_up, 1 - direction_up))
        
        result = {
            'buy_prob': round(buy_prob, 4),
            'sell_prob': round(sell_prob, 4),
            'direction_up': round(direction_up, 4),
            'direction_down': round(1 - direction_up, 4),
            'regime': regime_name,
            'regime_id': regime_id,
            'regime_probs': {
                REGIME_MAP[i]: round(float(regime_prob[i]), 4)
                for i in range(len(regime_prob))
            },
            'confidence': round(confidence, 4),
            'timestamp': time.time()
        }
        
        return result
        
    except Exception as e:
        return {'error': str(e)}


# Flask Routes
@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device),
        'timestamp': time.time()
    })


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model': 'Natron Transformer V2',
        'sequence_length': config['data']['sequence_length'],
        'num_features': config['data']['feature_count'],
        'tasks': ['buy', 'sell', 'direction', 'regime'],
        'regime_classes': REGIME_MAP,
        'parameters': sum(p.numel() for p in model.parameters())
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    
    Request body:
    {
        "data": [
            {"time": "2023-01-01 00:00", "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 1000},
            ...
        ]
    }
    
    Response:
    {
        "buy_prob": 0.71,
        "sell_prob": 0.24,
        "direction_up": 0.69,
        "regime": "BULL_WEAK",
        "confidence": 0.82
    }
    """
    try:
        data = request.json
        
        if 'data' not in data:
            return jsonify({'error': 'Missing "data" field'}), 400
        
        ohlcv_data = data['data']
        
        # Generate prediction
        result = predict_from_ohlcv(ohlcv_data)
        
        # Log if enabled
        if config['api'].get('log_predictions', False) and 'error' not in result:
            log_path = os.path.join(config['logging']['save_dir'], 'predictions.log')
            with open(log_path, 'a') as f:
                f.write(json.dumps(result) + '\n')
        
        if 'error' in result:
            return jsonify(result), 400
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Socket Server for MQL5
class SocketServer:
    """TCP Socket server for MQL5 communication"""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 9999):
        self.host = host
        self.port = port
        self.server_socket = None
        self.running = False
        
    def start(self):
        """Start socket server"""
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.running = True
        
        print(f"🔌 Socket server listening on {self.host}:{self.port}")
        
        while self.running:
            try:
                client_socket, address = self.server_socket.accept()
                print(f"📞 Connection from {address}")
                
                # Handle in new thread
                thread = threading.Thread(
                    target=self.handle_client,
                    args=(client_socket,)
                )
                thread.start()
                
            except Exception as e:
                if self.running:
                    print(f"❌ Socket error: {e}")
    
    def handle_client(self, client_socket: socket.socket):
        """Handle client connection"""
        try:
            # Receive data
            data = b''
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                data += chunk
                if len(chunk) < 4096:
                    break
            
            if not data:
                return
            
            # Parse JSON
            request = json.loads(data.decode('utf-8'))
            
            # Generate prediction
            if 'data' in request:
                result = predict_from_ohlcv(request['data'])
            else:
                result = {'error': 'Missing data field'}
            
            # Send response
            response = json.dumps(result).encode('utf-8')
            client_socket.sendall(response)
            
        except Exception as e:
            error_response = json.dumps({'error': str(e)}).encode('utf-8')
            client_socket.sendall(error_response)
            
        finally:
            client_socket.close()
    
    def stop(self):
        """Stop socket server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()


def start_socket_server():
    """Start socket server in background thread"""
    if config is None:
        return
    
    socket_server = SocketServer(
        host='0.0.0.0',
        port=config['mql5']['socket_port']
    )
    
    thread = threading.Thread(target=socket_server.start)
    thread.daemon = True
    thread.start()


if __name__ == '__main__':
    print("="*60)
    print("🚀 Natron API Server Starting...")
    print("="*60)
    
    # Load model
    if not load_model():
        print("❌ Failed to load model. Exiting.")
        exit(1)
    
    # Start socket server
    start_socket_server()
    
    # Start Flask app
    host = config['api']['host']
    port = config['api']['port']
    
    print(f"\n✅ Server ready!")
    print(f"   Flask API: http://{host}:{port}")
    print(f"   Socket Server: {host}:{config['mql5']['socket_port']}")
    print(f"\n📡 Endpoints:")
    print(f"   GET  /health")
    print(f"   GET  /model_info")
    print(f"   POST /predict")
    print("\n" + "="*60 + "\n")
    
    app.run(
        host=host,
        port=port,
        debug=False,
        threaded=True
    )
