"""
Flask API Server for Natron Transformer Inference
"""
import os
import torch
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from pathlib import Path

from config import Config
from feature_engine import FeatureEngine
from model import NatronTransformer

app = Flask(__name__)

# Global model and feature engine
model = None
feature_engine = None
device = None
label_generator = None


def load_model():
    """Load trained model"""
    global model, feature_engine, device, label_generator
    
    device = torch.device(Config.DEVICE)
    if not torch.cuda.is_available() and device.type == 'cuda':
        device = torch.device('cpu')
    
    # Initialize model
    model = NatronTransformer(
        num_features=Config.NUM_FEATURES,
        d_model=Config.D_MODEL,
        n_heads=Config.N_HEADS,
        n_layers=Config.N_LAYERS,
        d_ff=Config.D_FF,
        dropout=Config.DROPOUT,
        sequence_length=Config.SEQUENCE_LENGTH
    ).to(device)
    
    # Load weights
    if Config.MODEL_PATH.exists():
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"Loaded model from {Config.MODEL_PATH}")
    else:
        print(f"Warning: Model not found at {Config.MODEL_PATH}")
    
    model.eval()
    
    # Initialize feature engine
    feature_engine = FeatureEngine()
    
    # Regime names
    from label_generator import LabelGenerator
    label_generator = LabelGenerator()
    
    print("Model loaded and ready for inference")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict trading signals from last 96 OHLCV candles
    
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
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        
        if 'candles' not in data:
            return jsonify({'error': 'Missing "candles" key'}), 400
        
        candles = data['candles']
        
        if len(candles) < Config.SEQUENCE_LENGTH:
            return jsonify({
                'error': f'Need at least {Config.SEQUENCE_LENGTH} candles, got {len(candles)}'
            }), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(candles)
        
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            return jsonify({'error': f'Missing columns: {missing}'}), 400
        
        # Take last 96 candles
        df = df.tail(Config.SEQUENCE_LENGTH).reset_index(drop=True)
        
        # Generate features
        features_df = feature_engine.generate_features(df)
        
        # Convert to tensor
        features_array = features_df.values.astype(np.float32)
        features_tensor = torch.FloatTensor(features_array).unsqueeze(0).to(device)  # (1, 96, 100)
        
        # Predict
        with torch.no_grad():
            predictions = model(features_tensor)
        
        # Extract probabilities
        buy_prob = predictions['buy'][0].item()
        sell_prob = predictions['sell'][0].item()
        direction_probs = torch.softmax(predictions['direction'][0], dim=0)
        direction_up_prob = direction_probs[1].item()
        regime_probs = torch.softmax(predictions['regime'][0], dim=0)
        regime_idx = regime_probs.argmax().item()
        regime_name = label_generator.regime_names[regime_idx]
        regime_confidence = regime_probs[regime_idx].item()
        
        # Overall confidence (average of max probabilities)
        confidence = (buy_prob if buy_prob > sell_prob else sell_prob + 
                     direction_probs.max().item() + regime_confidence) / 3
        
        response = {
            'buy_prob': round(buy_prob, 4),
            'sell_prob': round(sell_prob, 4),
            'direction_up': round(direction_up_prob, 4),
            'regime': regime_name,
            'regime_prob': round(regime_confidence, 4),
            'confidence': round(confidence, 4)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict for multiple sequences
    
    Expected JSON:
    {
        "sequences": [
            [{"time": "...", "open": 1.0, ...}, ...],  # 96 candles
            ...
        ]
    }
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        data = request.json
        sequences = data.get('sequences', [])
        
        results = []
        for candles in sequences:
            if len(candles) < Config.SEQUENCE_LENGTH:
                results.append({'error': f'Need {Config.SEQUENCE_LENGTH} candles'})
                continue
            
            df = pd.DataFrame(candles).tail(Config.SEQUENCE_LENGTH)
            features_df = feature_engine.generate_features(df)
            features_array = features_df.values.astype(np.float32)
            features_tensor = torch.FloatTensor(features_array).unsqueeze(0).to(device)
            
            with torch.no_grad():
                predictions = model(features_tensor)
            
            buy_prob = predictions['buy'][0].item()
            sell_prob = predictions['sell'][0].item()
            direction_probs = torch.softmax(predictions['direction'][0], dim=0)
            regime_probs = torch.softmax(predictions['regime'][0], dim=0)
            regime_idx = regime_probs.argmax().item()
            
            results.append({
                'buy_prob': round(buy_prob, 4),
                'sell_prob': round(sell_prob, 4),
                'direction_up': round(direction_probs[1].item(), 4),
                'regime': label_generator.regime_names[regime_idx]
            })
        
        return jsonify({'results': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("Loading Natron Transformer model...")
    load_model()
    print(f"Starting API server on {Config.API_HOST}:{Config.API_PORT}")
    app.run(host=Config.API_HOST, port=Config.API_PORT, debug=False)
