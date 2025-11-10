"""
Natron API Server - Flask REST API and Socket Server
Author: Natron AI System
"""

import torch
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import yaml
import socket
import threading
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.natron_transformer import NatronTransformer
from src.features.feature_engine import FeatureEngine


class NatronInferenceServer:
    """
    Inference server for Natron Transformer.
    Provides REST API and TCP socket interface.
    """
    
    def __init__(self, config_path: str = "config/config.yaml", model_path: str = None):
        """
        Args:
            config_path: Path to configuration
            model_path: Path to trained model checkpoint
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.server_config = self.config['server']
        self.model_config = self.config['model']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {self.device}")
        
        # Load model
        if model_path is None:
            model_path = self.config['supervised']['checkpoint_path'].replace('.pt', '_best.pt')
        
        self.model = self._load_model(model_path)
        
        # Feature engine
        self.feature_engine = FeatureEngine(verbose=False)
        self.scaling_params = None
        
        # Regime names
        self.regime_names = {
            0: 'BULL_STRONG',
            1: 'BULL_WEAK',
            2: 'RANGE',
            3: 'BEAR_WEAK',
            4: 'BEAR_STRONG',
            5: 'VOLATILE'
        }
        
        print("✅ Natron Inference Server initialized")
    
    def _load_model(self, model_path: str) -> NatronTransformer:
        """Load trained model"""
        print(f"📥 Loading model from {model_path}...")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get num_features from config or checkpoint
        num_features = self.config['features']['n_features']
        
        model = NatronTransformer(
            num_features=num_features,
            d_model=self.model_config['d_model'],
            nhead=self.model_config['nhead'],
            num_encoder_layers=self.model_config['num_encoder_layers'],
            dim_feedforward=self.model_config['dim_feedforward'],
            dropout=self.model_config['dropout'],
            activation=self.model_config['activation']
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print(f"✅ Model loaded successfully")
        return model
    
    def preprocess_ohlcv(self, ohlcv_data: pd.DataFrame) -> np.ndarray:
        """
        Preprocess OHLCV data to features.
        
        Args:
            ohlcv_data: DataFrame with columns [time, open, high, low, close, volume]
                        Should have at least 96 rows for sequence
        
        Returns:
            Feature array ready for model (96, num_features)
        """
        # Generate features
        features = self.feature_engine.generate_all_features(ohlcv_data)
        
        # Normalize (use stored params if available)
        if self.scaling_params is not None:
            scaler = self.scaling_params['scaler']
            features_normalized = pd.DataFrame(
                scaler.transform(features),
                columns=features.columns
            )
        else:
            features_normalized, self.scaling_params = self.feature_engine.normalize_features(features)
        
        # Get last 96 candles
        sequence = features_normalized.values[-96:]
        
        return sequence
    
    @torch.no_grad()
    def predict(self, sequence: np.ndarray) -> dict:
        """
        Make prediction on a sequence.
        
        Args:
            sequence: Feature sequence (96, num_features)
        
        Returns:
            Dictionary with predictions
        """
        # Convert to tensor
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
        
        # Forward pass
        outputs = self.model(sequence_tensor)
        
        # Get predictions
        buy_prob = outputs['buy_prob'].item()
        sell_prob = outputs['sell_prob'].item()
        
        direction_probs = torch.softmax(outputs['direction_logits'], dim=1)
        direction_up = direction_probs[0, 1].item()
        
        regime_probs = torch.softmax(outputs['regime_logits'], dim=1)
        regime_id = regime_probs.argmax(dim=1).item()
        regime_name = self.regime_names[regime_id]
        regime_confidence = regime_probs[0, regime_id].item()
        
        # Overall confidence (average of all probabilities)
        confidence = (buy_prob + (1 - sell_prob) + direction_up + regime_confidence) / 4
        
        result = {
            'buy_prob': round(buy_prob, 4),
            'sell_prob': round(sell_prob, 4),
            'direction_up': round(direction_up, 4),
            'direction_down': round(1 - direction_up, 4),
            'regime': regime_name,
            'regime_id': int(regime_id),
            'regime_confidence': round(regime_confidence, 4),
            'confidence': round(confidence, 4),
            'signal': self._generate_signal(buy_prob, sell_prob, direction_up)
        }
        
        return result
    
    def _generate_signal(self, buy_prob: float, sell_prob: float, direction_up: float) -> str:
        """
        Generate trading signal from predictions.
        
        Args:
            buy_prob: Buy probability
            sell_prob: Sell probability
            direction_up: Direction up probability
        
        Returns:
            Signal: 'BUY', 'SELL', 'HOLD'
        """
        # Conservative signal generation
        if buy_prob > 0.6 and direction_up > 0.55 and sell_prob < 0.3:
            return 'BUY'
        elif sell_prob > 0.6 and direction_up < 0.45 and buy_prob < 0.3:
            return 'SELL'
        else:
            return 'HOLD'


# Flask App
app = Flask(__name__)
CORS(app)

# Global inference server
inference_server = None


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': inference_server is not None})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.
    
    Expects JSON with OHLCV data:
    {
        "data": [
            {"time": "2023-01-01 00:00:00", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 1000},
            ...
        ]
    }
    
    Returns:
    {
        "buy_prob": 0.71,
        "sell_prob": 0.24,
        "direction_up": 0.69,
        "regime": "BULL_WEAK",
        "confidence": 0.82,
        "signal": "BUY"
    }
    """
    try:
        # Get data from request
        data = request.get_json()
        
        if 'data' not in data:
            return jsonify({'error': 'Missing "data" field'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data['data'])
        
        # Validate columns
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                return jsonify({'error': f'Missing column: {col}'}), 400
        
        # Check minimum length
        if len(df) < 96:
            return jsonify({'error': f'Insufficient data. Need at least 96 candles, got {len(df)}'}), 400
        
        # Preprocess
        sequence = inference_server.preprocess_ohlcv(df)
        
        # Predict
        prediction = inference_server.predict(sequence)
        
        return jsonify(prediction)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_csv', methods=['POST'])
def predict_from_csv():
    """
    Prediction endpoint from CSV file.
    Expects form-data with file upload.
    """
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Read CSV
        df = pd.read_csv(file)
        
        # Validate
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                return jsonify({'error': f'Missing column: {col}'}), 400
        
        if len(df) < 96:
            return jsonify({'error': f'Insufficient data. Need at least 96 candles'}), 400
        
        # Preprocess
        sequence = inference_server.preprocess_ohlcv(df)
        
        # Predict
        prediction = inference_server.predict(sequence)
        
        return jsonify(prediction)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


class SocketServer:
    """
    TCP Socket server for MQL5 integration.
    Allows MetaTrader to send OHLCV data and receive predictions.
    """
    
    def __init__(self, inference_server: NatronInferenceServer, host: str = '0.0.0.0', port: int = 9090):
        self.inference_server = inference_server
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
                print(f"📞 Client connected: {address}")
                
                # Handle client in a new thread
                client_thread = threading.Thread(target=self.handle_client, args=(client_socket,))
                client_thread.start()
            
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
                if b'\n' in data:  # End of message
                    break
            
            if not data:
                return
            
            # Parse JSON
            message = json.loads(data.decode('utf-8'))
            
            # Convert to DataFrame
            df = pd.DataFrame(message['data'])
            
            # Preprocess and predict
            sequence = self.inference_server.preprocess_ohlcv(df)
            prediction = self.inference_server.predict(sequence)
            
            # Send response
            response = json.dumps(prediction) + '\n'
            client_socket.sendall(response.encode('utf-8'))
            
            print(f"✅ Prediction sent: {prediction['signal']}")
        
        except Exception as e:
            print(f"❌ Error handling client: {e}")
            error_response = json.dumps({'error': str(e)}) + '\n'
            client_socket.sendall(error_response.encode('utf-8'))
        
        finally:
            client_socket.close()
    
    def stop(self):
        """Stop socket server"""
        self.running = False
        if self.server_socket:
            self.server_socket.close()


def main():
    """Main server function"""
    print("=" * 80)
    print("🚀 NATRON INFERENCE SERVER")
    print("=" * 80)
    
    global inference_server
    
    # Initialize inference server
    inference_server = NatronInferenceServer(
        config_path="config/config.yaml"
    )
    
    # Start socket server in background
    socket_server = SocketServer(
        inference_server=inference_server,
        host=inference_server.server_config['host'],
        port=inference_server.server_config['socket_port']
    )
    
    socket_thread = threading.Thread(target=socket_server.start, daemon=True)
    socket_thread.start()
    
    # Start Flask app
    print(f"\n🌐 Starting Flask API server...")
    print(f"   REST API: http://{inference_server.server_config['host']}:{inference_server.server_config['port']}")
    print(f"   Socket: {inference_server.server_config['host']}:{inference_server.server_config['socket_port']}")
    print(f"\n📡 Endpoints:")
    print(f"   GET  /health")
    print(f"   POST /predict")
    print(f"   POST /predict_csv")
    
    app.run(
        host=inference_server.server_config['host'],
        port=inference_server.server_config['port'],
        debug=False,
        threaded=True
    )


if __name__ == "__main__":
    main()
