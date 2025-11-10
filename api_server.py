"""
Flask API Server for Natron Model Inference
"""
from flask import Flask, request, jsonify
import torch
import numpy as np
import pandas as pd
from model import NatronModel
from feature_engine import FeatureEngine
import os
from typing import Dict, List


app = Flask(__name__)

# Global model and feature engine
model = None
feature_engine = FeatureEngine()
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(model_path: str = 'models/natron_v2.pt'):
    """Load the trained Natron model"""
    global model
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Initialize model (adjust parameters if needed)
    model = NatronModel(
        input_dim=100,
        d_model=256,
        nhead=8,
        num_layers=6
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded from {model_path}")


def prepare_sequence(ohlcv_data: List[Dict]) -> np.ndarray:
    """
    Prepare input sequence from OHLCV data
    
    Args:
        ohlcv_data: List of dicts with keys: time, open, high, low, close, volume
        
    Returns:
        Feature array of shape (96, 100)
    """
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv_data)
    
    # Ensure we have exactly 96 candles
    if len(df) < 96:
        raise ValueError(f"Need at least 96 candles, got {len(df)}")
    
    # Take last 96 candles
    df = df.tail(96).reset_index(drop=True)
    
    # Generate features
    df_features = feature_engine.generate_features(df)
    
    # Get feature columns (exclude original OHLCV)
    feature_columns = feature_engine.get_feature_columns(df_features)
    
    # Extract feature matrix
    feature_matrix = df_features[feature_columns].values.astype(np.float32)
    
    # Normalize (using saved statistics if available, otherwise compute on-the-fly)
    feature_mean = np.nanmean(feature_matrix, axis=0, keepdims=True)
    feature_std = np.nanstd(feature_matrix, axis=0, keepdims=True) + 1e-8
    feature_matrix = (feature_matrix - feature_mean) / feature_std
    
    # Replace NaN/inf
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feature_matrix


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict trading signals from OHLCV data
    
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
        "direction_down": 0.31,
        "regime": "BULL_WEAK",
        "regime_probs": [0.1, 0.4, 0.2, 0.1, 0.1, 0.1],
        "confidence": 0.82
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        
        if 'candles' not in data:
            return jsonify({'error': 'Missing "candles" key'}), 400
        
        candles = data['candles']
        
        if len(candles) < 96:
            return jsonify({'error': f'Need at least 96 candles, got {len(candles)}'}), 400
        
        # Prepare sequence
        sequence = prepare_sequence(candles)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = model(input_tensor)
        
        # Extract predictions
        buy_prob = float(outputs['buy'].item())
        sell_prob = float(outputs['sell'].item())
        direction_probs = outputs['direction'].cpu().numpy()[0]
        regime_probs = outputs['regime'].cpu().numpy()[0]
        
        # Get regime name
        regime_id = int(np.argmax(regime_probs))
        regime_names = [
            'BULL_STRONG', 'BULL_WEAK', 'RANGE',
            'BEAR_WEAK', 'BEAR_STRONG', 'VOLATILE'
        ]
        regime_name = regime_names[regime_id]
        
        # Calculate confidence (max probability across tasks)
        confidence = float(max(buy_prob, sell_prob, direction_probs.max(), regime_probs.max()))
        
        response = {
            'buy_prob': round(buy_prob, 4),
            'sell_prob': round(sell_prob, 4),
            'direction_up': round(float(direction_probs[1]), 4),
            'direction_down': round(float(direction_probs[0]), 4),
            'regime': regime_name,
            'regime_id': regime_id,
            'regime_probs': [round(float(p), 4) for p in regime_probs],
            'confidence': round(confidence, 4)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict on multiple sequences
    
    Expected JSON:
    {
        "sequences": [
            [{"time": "...", "open": 1.0, ...}, ...],
            ...
        ]
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.get_json()
        sequences = data.get('sequences', [])
        
        results = []
        for candles in sequences:
            sequence = prepare_sequence(candles)
            input_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            
            with torch.no_grad():
                outputs = model(input_tensor)
            
            buy_prob = float(outputs['buy'].item())
            sell_prob = float(outputs['sell'].item())
            direction_probs = outputs['direction'].cpu().numpy()[0]
            regime_probs = outputs['regime'].cpu().numpy()[0]
            regime_id = int(np.argmax(regime_probs))
            
            regime_names = [
                'BULL_STRONG', 'BULL_WEAK', 'RANGE',
                'BEAR_WEAK', 'BEAR_STRONG', 'VOLATILE'
            ]
            
            results.append({
                'buy_prob': round(buy_prob, 4),
                'sell_prob': round(sell_prob, 4),
                'direction_up': round(float(direction_probs[1]), 4),
                'regime': regime_names[regime_id]
            })
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/natron_v2.pt',
                       help='Path to trained model')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to bind to')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    load_model(args.model_path)
    
    # Start server
    print(f"Starting API server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)
