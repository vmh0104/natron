"""
Natron V2 Flask API Server
RESTful API for model inference
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import yaml
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.natron_transformer import NatronTransformer
from src.data.dataset_loader import DataProcessor
from src.labels.label_generator import LabelGenerator


app = Flask(__name__)
CORS(app)

# Global variables
model = None
config = None
data_processor = None
label_generator = None
device = None


def load_model(config_path: str = 'config/config.yaml', 
               model_path: str = 'model/natron_v2.pt'):
    """
    Load trained model and configuration.
    """
    global model, config, data_processor, label_generator, device
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(config['deployment']['device'] 
                         if torch.cuda.is_available() 
                         else 'cpu')
    
    print(f"Using device: {device}")
    
    # Initialize model
    model = NatronTransformer(config).to(device)
    
    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"✓ Model loaded from {model_path}")
    else:
        print(f"⚠ Model not found at {model_path}. Using untrained model.")
    
    # Initialize data processor
    data_processor = DataProcessor(config)
    
    # Load scaler if exists
    scaler_path = 'model/scaler.pkl'
    if os.path.exists(scaler_path):
        data_processor.load_scaler(scaler_path)
        print(f"✓ Scaler loaded from {scaler_path}")
    else:
        print(f"⚠ Scaler not found at {scaler_path}")
    
    # Initialize label generator
    label_generator = LabelGenerator()
    
    print("✓ Server ready!")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device)
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint.
    
    Expects JSON with OHLCV data:
    {
        "data": [
            [timestamp, open, high, low, close, volume],
            ... (96 candles total)
        ]
    }
    
    Returns:
    {
        "buy_prob": float,
        "sell_prob": float,
        "direction_up": float,
        "direction_class": int (0=down, 1=up),
        "regime": str,
        "regime_class": int,
        "confidence": float
    }
    """
    try:
        # Parse request
        data = request.get_json()
        
        if 'data' not in data:
            return jsonify({'error': 'Missing "data" field'}), 400
        
        ohlcv = np.array(data['data'])
        
        if ohlcv.shape[0] != 96:
            return jsonify({'error': f'Expected 96 candles, got {ohlcv.shape[0]}'}), 400
        
        if ohlcv.shape[1] != 6:
            return jsonify({'error': f'Expected 6 columns [time, O, H, L, C, V], got {ohlcv.shape[1]}'}), 400
        
        # Preprocess
        input_tensor = data_processor.preprocess_inference_data(ohlcv)
        input_tensor = input_tensor.to(device)
        
        # Predict
        with torch.no_grad():
            predictions = model.predict(input_tensor)
        
        # Convert to JSON-serializable format
        response = {
            'buy_prob': float(predictions['buy_prob'][0].cpu().numpy()),
            'sell_prob': float(predictions['sell_prob'][0].cpu().numpy()),
            'direction_up': float(predictions['direction_prob'][0].cpu().numpy()),
            'direction_class': int(predictions['direction_class'][0].cpu().numpy()),
            'regime': label_generator.get_regime_name(
                int(predictions['regime_class'][0].cpu().numpy())
            ),
            'regime_class': int(predictions['regime_class'][0].cpu().numpy()),
            'confidence': float(predictions['confidence'][0].cpu().numpy())
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """
    Batch prediction endpoint.
    
    Expects JSON with multiple sequences:
    {
        "sequences": [
            [[t, o, h, l, c, v], ...],  # 96 candles
            [[t, o, h, l, c, v], ...],  # 96 candles
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        
        if 'sequences' not in data:
            return jsonify({'error': 'Missing "sequences" field'}), 400
        
        sequences = data['sequences']
        results = []
        
        for seq in sequences:
            ohlcv = np.array(seq)
            
            # Preprocess
            input_tensor = data_processor.preprocess_inference_data(ohlcv)
            input_tensor = input_tensor.to(device)
            
            # Predict
            with torch.no_grad():
                predictions = model.predict(input_tensor)
            
            result = {
                'buy_prob': float(predictions['buy_prob'][0].cpu().numpy()),
                'sell_prob': float(predictions['sell_prob'][0].cpu().numpy()),
                'direction_up': float(predictions['direction_prob'][0].cpu().numpy()),
                'regime': label_generator.get_regime_name(
                    int(predictions['regime_class'][0].cpu().numpy())
                ),
                'confidence': float(predictions['confidence'][0].cpu().numpy())
            }
            
            results.append(result)
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_type': 'NatronTransformer',
        'd_model': config['model']['d_model'],
        'num_layers': config['model']['num_encoder_layers'],
        'num_heads': config['model']['nhead'],
        'sequence_length': config['model']['max_seq_length'],
        'num_features': config['features']['num_features']
    })


def start_server(host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
    """Start Flask server"""
    load_model()
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron V2 Flask API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Config path')
    parser.add_argument('--model', type=str, default='model/natron_v2.pt', help='Model path')
    
    args = parser.parse_args()
    
    load_model(args.config, args.model)
    app.run(host=args.host, port=args.port, debug=args.debug)
