"""
Natron Transformer - Flask API Server
REST API for model inference
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from typing import Dict
import yaml
import os
import sys
from pathlib import Path

# Add paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from inference import NatronInference

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for cross-origin requests

# Global inference engine
inference_engine = None
config = None


def init_app(config_path: str = 'config.yaml', model_path: str = None):
    """Initialize the application"""
    global inference_engine, config
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get model path
    if model_path is None:
        model_path = config['inference']['model_path']
    
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found: {model_path}")
        print("   Please train a model first using train_natron.py")
        return False
    
    # Initialize inference engine
    print(f"🚀 Initializing Natron API Server...")
    device = 'cuda' if config['inference']['device'] == 'cuda' else 'cpu'
    inference_engine = NatronInference(model_path, config_path, device)
    
    print(f"✅ API Server initialized")
    return True


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    if inference_engine is None:
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded'
        }), 503
    
    return jsonify({
        'status': 'healthy',
        'model': 'Natron Transformer v2',
        'device': inference_engine.device
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    
    Request body (JSON):
    {
        "data": [
            {"time": "2024-01-01 00:00", "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 5000},
            ...
            (at least 96 candles)
        ]
    }
    
    Response (JSON):
    {
        "success": true,
        "prediction": {
            "buy_prob": 0.71,
            "sell_prob": 0.24,
            "direction_up": 0.69,
            "direction_down": 0.31,
            "regime": "BULL_WEAK",
            "regime_id": 1,
            "confidence": 0.82,
            "signal": "BUY",
            "timestamp": "2024-01-01 00:00"
        }
    }
    """
    if inference_engine is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 503
    
    try:
        # Parse request
        data = request.get_json()
        
        if 'data' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "data" field in request'
            }), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(data['data'])
        
        # Validate columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                return jsonify({
                    'success': False,
                    'error': f'Missing required column: {col}'
                }), 400
        
        # Check length
        if len(df) < inference_engine.sequence_length:
            return jsonify({
                'success': False,
                'error': f'Need at least {inference_engine.sequence_length} candles, got {len(df)}'
            }), 400
        
        # Predict
        prediction = inference_engine.predict(df)
        
        return jsonify({
            'success': True,
            'prediction': prediction
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    
    Request body (JSON):
    {
        "batches": [
            [
                {"time": "...", "open": 100, "high": 101, "low": 99, "close": 100.5, "volume": 5000},
                ...
            ],
            [
                ...
            ]
        ]
    }
    
    Response (JSON):
    {
        "success": true,
        "predictions": [
            {...},
            {...}
        ]
    }
    """
    if inference_engine is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 503
    
    try:
        # Parse request
        data = request.get_json()
        
        if 'batches' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing "batches" field in request'
            }), 400
        
        # Convert to DataFrames
        dfs = [pd.DataFrame(batch) for batch in data['batches']]
        
        # Predict
        predictions = inference_engine.predict_batch(dfs)
        
        return jsonify({
            'success': True,
            'predictions': predictions
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if inference_engine is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 503
    
    return jsonify({
        'success': True,
        'info': {
            'model_name': 'Natron Transformer v2',
            'sequence_length': inference_engine.sequence_length,
            'num_features': inference_engine.config['model']['input_dim'],
            'device': inference_engine.device,
            'confidence_threshold': inference_engine.confidence_threshold,
            'regime_classes': inference_engine.label_generator.regime_names
        }
    })


def main():
    """Main function to run the server"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron API Server')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--model', type=str, default=None, help='Model file path')
    parser.add_argument('--host', type=str, default=None, help='Host address')
    parser.add_argument('--port', type=int, default=None, help='Port number')
    
    args = parser.parse_args()
    
    # Initialize app
    success = init_app(args.config, args.model)
    
    if not success:
        print("❌ Failed to initialize API server")
        return
    
    # Get host and port from config or args
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    host = args.host or cfg['api']['host']
    port = args.port or cfg['api']['port']
    
    print(f"\n🌐 Starting Natron API Server...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"\n📡 Endpoints:")
    print(f"   GET  /health       - Health check")
    print(f"   POST /predict      - Single prediction")
    print(f"   POST /predict_batch - Batch prediction")
    print(f"   GET  /model_info   - Model information")
    print(f"\n✅ Server ready!")
    
    # Run server
    app.run(
        host=host,
        port=port,
        debug=False,
        threaded=True
    )


if __name__ == '__main__':
    main()
