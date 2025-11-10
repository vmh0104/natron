"""
Flask API Server for Natron Transformer Inference
Provides REST API endpoint for trading predictions
"""

import torch
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from datetime import datetime

from model_natron import NatronTransformer
from feature_engine import FeatureEngine
from dataset_loader import SequenceCreator


app = Flask(__name__)
CORS(app)

# Global variables
model = None
feature_engine = None
feature_columns = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sequence_length = 96


def load_model(model_path: str, num_features: int, feature_columns: list):
    """Load trained Natron model"""
    global model, feature_engine
    
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = NatronTransformer(
        num_features=num_features,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        sequence_length=sequence_length
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    feature_engine = FeatureEngine()
    
    print("Model loaded successfully!")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': device
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint
    
    Expected JSON:
    {
        "candles": [
            {
                "time": "2024-01-01 00:00:00",
                "open": 1.1000,
                "high": 1.1050,
                "low": 1.0990,
                "close": 1.1030,
                "volume": 1000
            },
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
        "regime_probs": [0.1, 0.4, 0.2, 0.1, 0.1, 0.1]
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if 'candles' not in data:
            return jsonify({'error': 'Missing "candles" field'}), 400
        
        candles = data['candles']
        
        # Validate input
        if len(candles) < sequence_length:
            return jsonify({
                'error': f'Need at least {sequence_length} candles, got {len(candles)}'
            }), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return jsonify({
                'error': f'Missing required columns. Need: {required_cols}'
            }), 400
        
        # Use last sequence_length candles
        df = df.tail(sequence_length).reset_index(drop=True)
        
        # Generate features
        features_df = feature_engine.fit_transform(df)
        feature_matrix = features_df[feature_columns].values.astype(np.float32)
        
        # Convert to tensor
        X = torch.FloatTensor(feature_matrix).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            result = model.predict(X)
        
        # Add timestamp
        result['timestamp'] = datetime.now().isoformat()
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    
    Expected JSON:
    {
        "sequences": [
            [candle1, candle2, ..., candle96],
            ...
        ]
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if 'sequences' not in data:
            return jsonify({'error': 'Missing "sequences" field'}), 400
        
        sequences = data['sequences']
        results = []
        
        for candles in sequences:
            if len(candles) < sequence_length:
                continue
            
            df = pd.DataFrame(candles)
            features_df = feature_engine.fit_transform(df)
            feature_matrix = features_df[feature_columns].values.astype(np.float32)
            X = torch.FloatTensor(feature_matrix).unsqueeze(0).to(device)
            
            with torch.no_grad():
                result = model.predict(X)
            
            results.append(result)
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron API Server')
    parser.add_argument('--model', type=str, default='./models/natron_v2.pt', help='Model path')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    
    args = parser.parse_args()
    
    # Load model configuration
    if os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location=device)
        feature_columns = checkpoint.get('feature_columns', None)
        num_features = checkpoint.get('num_features', 100)
        
        if feature_columns is None:
            print("Warning: feature_columns not found in checkpoint. Using default.")
            # Try to infer from dataset
            from dataset_loader import SequenceCreator
            import pandas as pd
            # Create dummy dataset to get feature columns
            dummy_df = pd.DataFrame({
                'time': pd.date_range('2024-01-01', periods=200, freq='15min'),
                'open': np.random.randn(200),
                'high': np.random.randn(200),
                'low': np.random.randn(200),
                'close': np.random.randn(200),
                'volume': np.random.randn(200)
            })
            creator = SequenceCreator()
            _, _, _ = creator.create_dataset(dummy_df)
            feature_columns = creator.feature_columns
            num_features = len(feature_columns)
        
        load_model(args.model, num_features, feature_columns)
    else:
        print(f"Warning: Model file {args.model} not found. Server will start but predictions will fail.")
    
    print(f"\nStarting Natron API Server on {args.host}:{args.port}")
    print(f"Device: {device}")
    app.run(host=args.host, port=args.port, debug=False)
