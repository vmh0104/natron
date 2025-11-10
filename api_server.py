"""
Natron Flask API Server - Inference endpoint
"""
from flask import Flask, request, jsonify
import torch
import numpy as np
import pandas as pd
from typing import Dict
import os

from feature_engine import FeatureEngine
from model import create_model
from sequence_creator import SequenceCreator


app = Flask(__name__)

# Global variables
model = None
feature_engine = None
device = 'cuda' if torch.cuda.is_available() else 'cpu'
sequence_length = 96
regime_names = [
    'BULL_STRONG', 'BULL_WEAK', 'RANGE', 
    'BEAR_WEAK', 'BEAR_STRONG', 'VOLATILE'
]


def load_model(model_path: str, num_features: int = 100):
    """Load trained model"""
    global model
    
    model = create_model(
        num_features=num_features,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        pretrain=False
    )
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    print(f"Model loaded from {model_path}")


def prepare_features(ohlcv_data: pd.DataFrame) -> np.ndarray:
    """Extract features from OHLCV data"""
    global feature_engine
    
    if feature_engine is None:
        feature_engine = FeatureEngine()
    
    features_df = feature_engine.extract_all_features(ohlcv_data)
    return features_df.values.astype(np.float32)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint
    
    Expects JSON with:
    {
        "candles": [
            {"time": "...", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 1000},
            ...
        ]
    }
    
    Returns:
    {
        "buy_prob": 0.71,
        "sell_prob": 0.24,
        "direction_up": 0.69,
        "regime": "BULL_WEAK",
        "confidence": 0.82
    }
    """
    global model
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        if 'candles' not in data:
            return jsonify({'error': 'Missing "candles" key'}), 400
        
        candles = data['candles']
        
        if len(candles) < sequence_length:
            return jsonify({
                'error': f'Need at least {sequence_length} candles, got {len(candles)}'
            }), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                return jsonify({'error': f'Missing column: {col}'}), 400
        
        # Take last sequence_length candles
        df = df.tail(sequence_length).reset_index(drop=True)
        
        # Extract features
        features = prepare_features(df)
        
        # Create sequence
        X = torch.FloatTensor(features).unsqueeze(0).to(device)  # [1, seq_len, num_features]
        
        # Predict
        with torch.no_grad():
            outputs = model(X, mode='supervised')
            
            buy_prob = outputs['buy'].item()
            sell_prob = outputs['sell'].item()
            direction_logits = outputs['direction'][0]
            direction_probs = torch.softmax(direction_logits, dim=0)
            direction_up = direction_probs[1].item()
            
            regime_logits = outputs['regime'][0]
            regime_probs = torch.softmax(regime_logits, dim=0)
            regime_idx = regime_probs.argmax().item()
            regime = regime_names[regime_idx]
            regime_confidence = regime_probs[regime_idx].item()
        
        # Overall confidence
        confidence = (buy_prob + (1 - sell_prob) + direction_up + regime_confidence) / 4
        
        response = {
            'buy_prob': round(buy_prob, 4),
            'sell_prob': round(sell_prob, 4),
            'direction_up': round(direction_up, 4),
            'regime': regime,
            'confidence': round(confidence, 4)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': device
    })


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/natron_v2.pt', help='Model path')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host')
    parser.add_argument('--num_features', type=int, default=100, help='Number of features')
    
    args = parser.parse_args()
    
    # Load model
    if os.path.exists(args.model):
        load_model(args.model, args.num_features)
    else:
        print(f"Warning: Model file {args.model} not found. Start training first.")
    
    # Initialize feature engine
    feature_engine = FeatureEngine()
    
    print(f"Starting API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
