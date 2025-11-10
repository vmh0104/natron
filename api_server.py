"""
Flask API Server for Natron Transformer Inference
Provides REST API endpoint for trading predictions
"""

from flask import Flask, request, jsonify
import torch
import numpy as np
import pandas as pd
import yaml
import os
import pickle
from model_natron import NatronTransformer
from feature_engine import FeatureEngine
from label_generator import LabelGenerator

app = Flask(__name__)

# Global variables
model = None
feature_engine = None
scaler = None
config = None
device = None


def load_model(model_path: str, config_path: str):
    """Load trained model and scaler"""
    global model, feature_engine, scaler, config, device
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    
    # Initialize model
    model = NatronTransformer(
        feature_dim=config['model']['feature_dim'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        activation=config['model']['activation']
    ).to(device)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load scaler
    scaler_path = os.path.join(config['paths']['model_dir'], 'scaler.pkl')
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
    
    # Initialize feature engine
    feature_engine = FeatureEngine()
    
    print(f"Model loaded from {model_path}")
    print(f"Using device: {device}")


def prepare_sequence(ohlcv_data: list) -> torch.Tensor:
    """
    Prepare input sequence from OHLCV data.
    
    Args:
        ohlcv_data: List of dicts with ['time', 'open', 'high', 'low', 'close', 'volume']
        
    Returns:
        sequence: (1, seq_len, feature_dim) tensor
    """
    # Convert to DataFrame
    df = pd.DataFrame(ohlcv_data)
    
    # Ensure we have exactly sequence_length candles
    sequence_length = config['model']['sequence_length']
    if len(df) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} candles, got {len(df)}")
    
    # Take last sequence_length candles
    df = df.tail(sequence_length).reset_index(drop=True)
    
    # Generate features
    features_df = feature_engine.generate_all_features(df)
    
    # Normalize
    if scaler is not None:
        features = scaler.transform(features_df.values)
    else:
        features = features_df.values
    
    # Convert to tensor
    sequence = torch.FloatTensor(features).unsqueeze(0).to(device)  # (1, seq_len, feature_dim)
    
    return sequence


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict trading signals from OHLCV data.
    
    Request body:
    {
        "candles": [
            {"time": "...", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 1000},
            ...
        ]
    }
    
    Response:
    {
        "buy_prob": 0.71,
        "sell_prob": 0.24,
        "direction_up": 0.69,
        "regime": "BULL_WEAK",
        "regime_id": 1,
        "confidence": 0.82
    }
    """
    try:
        data = request.get_json()
        
        if 'candles' not in data:
            return jsonify({'error': 'Missing "candles" field'}), 400
        
        candles = data['candles']
        
        if len(candles) < config['model']['sequence_length']:
            return jsonify({
                'error': f'Need at least {config["model"]["sequence_length"]} candles'
            }), 400
        
        # Prepare sequence
        sequence = prepare_sequence(candles)
        
        # Predict
        with torch.no_grad():
            predictions = model(sequence)
        
        # Extract predictions
        buy_prob = predictions['buy'].item()
        sell_prob = predictions['sell'].item()
        
        # Direction (convert log probabilities to probabilities)
        direction_logits = predictions['direction']
        direction_probs = torch.exp(direction_logits)
        direction_up = direction_probs[0][1].item()
        
        # Regime
        regime_logits = predictions['regime']
        regime_probs = torch.exp(regime_logits)
        regime_id = regime_probs[0].argmax().item()
        regime_confidence = regime_probs[0].max().item()
        
        # Get regime name
        label_generator = LabelGenerator(config)
        regime_name = label_generator.get_regime_name(regime_id)
        
        # Overall confidence (average of task confidences)
        confidence = (buy_prob + sell_prob + direction_up + regime_confidence) / 4.0
        
        response = {
            'buy_prob': round(buy_prob, 4),
            'sell_prob': round(sell_prob, 4),
            'direction_up': round(direction_up, 4),
            'direction_down': round(1 - direction_up, 4),
            'regime': regime_name,
            'regime_id': int(regime_id),
            'regime_probs': {
                'BULL_STRONG': round(regime_probs[0][0].item(), 4),
                'BULL_WEAK': round(regime_probs[0][1].item(), 4),
                'RANGE': round(regime_probs[0][2].item(), 4),
                'BEAR_WEAK': round(regime_probs[0][3].item(), 4),
                'BEAR_STRONG': round(regime_probs[0][4].item(), 4),
                'VOLATILE': round(regime_probs[0][5].item(), 4),
            },
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
        'device': str(device) if device else None
    })


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/natron_v2.pt',
                       help='Path to trained model')
    parser.add_argument('--config_path', type=str, default='./config.yaml',
                       help='Path to config file')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000,
                       help='Port to bind to')
    
    args = parser.parse_args()
    
    # Load model
    load_model(args.model_path, args.config_path)
    
    # Run server
    app.run(host=args.host, port=args.port, debug=False)
