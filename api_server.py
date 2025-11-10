"""
Natron Flask API Server - Inference Endpoint

Provides /predict endpoint that receives last 96 OHLCV candles
and returns JSON trading signal.
"""

import torch
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import json
import os
from typing import Dict

from feature_engine import FeatureEngine
from model_natron import create_natron_model
from label_generator import LabelGenerator


app = Flask(__name__)

# Global variables
model = None
feature_engine = None
label_generator = None
device = None
regime_names = ['BULL_STRONG', 'BULL_WEAK', 'RANGE', 'BEAR_WEAK', 'BEAR_STRONG', 'VOLATILE']


def load_model(model_path: str, device: torch.device):
    """Load trained Natron model."""
    global model, feature_engine
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get feature names from checkpoint or use default
    feature_names = checkpoint.get('feature_names', None)
    
    # Create model
    model = create_natron_model(
        input_dim=100,  # Default, adjust if needed
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=96
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    feature_engine = FeatureEngine()
    
    print(f'Model loaded from {model_path}')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict trading signal from last 96 OHLCV candles.
    
    Expected JSON:
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
    try:
        data = request.get_json()
        
        if 'candles' not in data:
            return jsonify({'error': 'Missing "candles" field'}), 400
        
        candles = data['candles']
        
        if len(candles) < 96:
            return jsonify({'error': f'Need at least 96 candles, got {len(candles)}'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return jsonify({'error': f'Missing required columns: {required_cols}'}), 400
        
        # Take last 96 candles
        df = df.tail(96).reset_index(drop=True)
        
        # Generate features
        features_df = feature_engine.fit_transform(df)
        
        # Convert to tensor
        features_array = features_df.values.astype(np.float32)
        features_tensor = torch.FloatTensor(features_array).unsqueeze(0).to(device)  # (1, 96, 100)
        
        # Predict
        with torch.no_grad():
            predictions = model.predict(features_tensor)
        
        # Extract predictions
        buy_prob = predictions['buy'].item()
        sell_prob = predictions['sell'].item()
        
        # Direction (convert from log-softmax to probability)
        direction_logprobs = predictions['direction']
        direction_probs = torch.exp(direction_logprobs)
        direction_up_prob = direction_probs[0][1].item()
        
        # Regime
        regime_logprobs = predictions['regime']
        regime_probs = torch.exp(regime_logprobs)
        regime_id = regime_probs.argmax().item()
        regime_name = regime_names[regime_id] if regime_id < len(regime_names) else 'UNKNOWN'
        regime_confidence = regime_probs.max().item()
        
        # Overall confidence (average of task confidences)
        confidence = (buy_prob + sell_prob + direction_up_prob + regime_confidence) / 4
        
        response = {
            'buy_prob': round(buy_prob, 4),
            'sell_prob': round(sell_prob, 4),
            'direction_up': round(direction_up_prob, 4),
            'regime': regime_name,
            'regime_id': int(regime_id),
            'confidence': round(confidence, 4)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron API Server')
    parser.add_argument('--model_path', type=str, default='./checkpoints/natron_v2.pt', help='Path to model checkpoint')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Load model
    if os.path.exists(args.model_path):
        load_model(args.model_path, device)
    else:
        print(f'Warning: Model not found at {args.model_path}. Please train a model first.')
    
    # Initialize label generator
    label_generator = LabelGenerator()
    
    print(f'Starting API server on {args.host}:{args.port}')
    app.run(host=args.host, port=args.port, debug=False)
