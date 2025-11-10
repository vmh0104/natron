"""
Natron Flask API Server - Inference endpoint for trading signals
"""
import torch
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import os
from pathlib import Path
import pickle

from feature_engine import FeatureEngine
from sequence_creator import SequenceCreator
from model import create_model
from label_generator import LabelGenerator


app = Flask(__name__)

# Global variables
model = None
feature_engine = None
sequence_creator = None
label_generator = LabelGenerator()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_columns = None
model_config = None


def load_model(model_path: str = 'model/natron_v2.pt', config_path: str = 'config.yaml'):
    """Load trained model and preprocessing components."""
    global model, feature_engine, sequence_creator, feature_columns, model_config
    
    # Load config
    import yaml
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            model_config = yaml.safe_load(f)
    else:
        model_config = {}
    
    # Initialize components
    feature_engine = FeatureEngine()
    sequence_creator = SequenceCreator(sequence_length=96)
    
    # Create model
    model_cfg = model_config.get('model', {})
    model_cfg.setdefault('d_model', 128)
    model_cfg.setdefault('nhead', 8)
    model_cfg.setdefault('num_layers', 6)
    model_cfg.setdefault('dim_feedforward', 512)
    model_cfg.setdefault('dropout', 0.1)
    model_cfg.setdefault('num_features', 100)
    model_cfg.setdefault('max_seq_len', 96)
    
    model = create_model(model_cfg)
    
    # Load weights
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.eval()
    model.to(device)
    
    # Load feature columns (if saved)
    feature_cols_path = 'model/feature_columns.pkl'
    if os.path.exists(feature_cols_path):
        with open(feature_cols_path, 'rb') as f:
            feature_columns = pickle.load(f)
    else:
        # Default: will be inferred from data
        feature_columns = None
    
    print("Model loaded successfully")


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'device': str(device)})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict trading signals from last 96 OHLCV candles.
    
    Request body (JSON):
    {
        "candles": [
            {"time": "2024-01-01 00:00:00", "open": 1.1000, "high": 1.1050, 
             "low": 1.0990, "close": 1.1040, "volume": 1000},
            ...
        ]
    }
    
    Response (JSON):
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
        data = request.json
        
        if 'candles' not in data:
            return jsonify({'error': 'Missing "candles" field'}), 400
        
        candles = data['candles']
        
        if len(candles) < 96:
            return jsonify({'error': f'Need at least 96 candles, got {len(candles)}'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(candles[-96:])  # Take last 96
        
        # Ensure required columns
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                return jsonify({'error': f'Missing column: {col}'}), 400
        
        # Generate features
        df_features = feature_engine.generate_features(df)
        
        # Get feature columns if not set
        global feature_columns
        if feature_columns is None:
            base_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
            feature_columns = [col for col in df_features.columns if col not in base_cols]
            # Save for future use
            os.makedirs('model', exist_ok=True)
            with open('model/feature_columns.pkl', 'wb') as f:
                pickle.dump(feature_columns, f)
        
        # Create sequence
        X_scaled = sequence_creator.transform_new_data(df_features)
        X_sequence = X_scaled[-96:].reshape(1, 96, -1)  # (1, 96, num_features)
        
        # Predict
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_sequence).to(device)
            outputs = model(X_tensor)
            
            buy_prob = outputs['buy'].cpu().item()
            sell_prob = outputs['sell'].cpu().item()
            direction_probs = outputs['direction'].cpu().numpy()[0]
            regime_probs = outputs['regime'].cpu().numpy()[0]
            
            direction_up = direction_probs[1]
            regime_id = int(np.argmax(regime_probs))
            regime_name = label_generator.get_regime_name(regime_id)
            
            # Confidence: max probability across tasks
            confidence = max(buy_prob, sell_prob, direction_probs.max(), regime_probs.max())
        
        response = {
            'buy_prob': float(buy_prob),
            'sell_prob': float(sell_prob),
            'direction_up': float(direction_up),
            'direction_down': float(direction_probs[0]),
            'regime': regime_name,
            'regime_id': regime_id,
            'regime_probs': {label_generator.get_regime_name(i): float(regime_probs[i]) 
                            for i in range(6)},
            'confidence': float(confidence)
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict for multiple sequences (for backtesting).
    
    Request body (JSON):
    {
        "sequences": [
            [{"time": "...", "open": ..., ...}, ...],  # 96 candles
            ...
        ]
    }
    """
    try:
        data = request.json
        
        if 'sequences' not in data:
            return jsonify({'error': 'Missing "sequences" field'}), 400
        
        sequences = data['sequences']
        results = []
        
        for candles in sequences:
            if len(candles) < 96:
                continue
            
            df = pd.DataFrame(candles[-96:])
            df_features = feature_engine.generate_features(df)
            X_scaled = sequence_creator.transform_new_data(df_features)
            X_sequence = X_scaled[-96:].reshape(1, 96, -1)
            
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X_sequence).to(device)
                outputs = model(X_tensor)
                
                buy_prob = outputs['buy'].cpu().item()
                sell_prob = outputs['sell'].cpu().item()
                direction_probs = outputs['direction'].cpu().numpy()[0]
                regime_probs = outputs['regime'].cpu().numpy()[0]
                
                results.append({
                    'buy_prob': float(buy_prob),
                    'sell_prob': float(sell_prob),
                    'direction_up': float(direction_probs[1]),
                    'regime': label_generator.get_regime_name(int(np.argmax(regime_probs)))
                })
        
        return jsonify({'predictions': results})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Load model on startup
    model_path = os.getenv('MODEL_PATH', 'model/natron_v2.pt')
    config_path = os.getenv('CONFIG_PATH', 'config.yaml')
    
    print("Loading Natron model...")
    load_model(model_path, config_path)
    
    # Run server
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    
    print(f"Starting Natron API server on {host}:{port}")
    app.run(host=host, port=port, debug=False)
