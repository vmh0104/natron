"""
Natron Inference Script - Standalone inference for testing
"""
import torch
import pandas as pd
import numpy as np
import argparse
import json

from feature_engine import FeatureEngine
from model import create_model
from sequence_creator import SequenceCreator


def load_model(model_path: str, num_features: int = 100, device: str = 'cuda'):
    """Load trained model"""
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
    return model


def predict_from_csv(
    model_path: str,
    csv_path: str,
    num_features: int = 100,
    sequence_length: int = 96
):
    """Predict from CSV file"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path, num_features, device)
    
    # Load data
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    
    # Extract features
    print("Extracting features...")
    feature_engine = FeatureEngine()
    features_df = feature_engine.extract_all_features(df)
    
    # Take last sequence_length candles
    if len(features_df) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} candles, got {len(features_df)}")
    
    features = features_df.tail(sequence_length).values.astype(np.float32)
    
    # Predict
    print("Running prediction...")
    X = torch.FloatTensor(features).unsqueeze(0).to(device)
    
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
        regime_names = [
            'BULL_STRONG', 'BULL_WEAK', 'RANGE',
            'BEAR_WEAK', 'BEAR_STRONG', 'VOLATILE'
        ]
        regime = regime_names[regime_idx]
        regime_confidence = regime_probs[regime_idx].item()
    
    confidence = (buy_prob + (1 - sell_prob) + direction_up + regime_confidence) / 4
    
    result = {
        'buy_prob': round(buy_prob, 4),
        'sell_prob': round(sell_prob, 4),
        'direction_up': round(direction_up, 4),
        'regime': regime,
        'regime_confidence': round(regime_confidence, 4),
        'confidence': round(confidence, 4)
    }
    
    print("\n" + "="*50)
    print("PREDICTION RESULTS")
    print("="*50)
    print(json.dumps(result, indent=2))
    print("="*50)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Natron Inference')
    parser.add_argument('--model', type=str, required=True, help='Model path')
    parser.add_argument('--data', type=str, required=True, help='CSV data path')
    parser.add_argument('--num_features', type=int, default=100, help='Number of features')
    parser.add_argument('--sequence_length', type=int, default=96, help='Sequence length')
    
    args = parser.parse_args()
    
    predict_from_csv(
        args.model,
        args.data,
        args.num_features,
        args.sequence_length
    )


if __name__ == '__main__':
    main()
