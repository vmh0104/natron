#!/usr/bin/env python3
"""
Natron Inference Script
Test the trained model on new data

Usage:
    python inference.py --csv data/test_data.csv
    python inference.py --realtime  # Get predictions every N seconds
"""

import torch
import pandas as pd
import numpy as np
import yaml
import argparse
import sys
import pickle
import time
from typing import Dict

sys.path.append('src')

from model_natron import NatronTransformer
from feature_engine import FeatureEngine
from label_generator import LabelGenerator
from dataset_loader import create_inference_sequence


REGIME_MAP = {
    0: "BULL_STRONG",
    1: "BULL_WEAK",
    2: "RANGE",
    3: "BEAR_WEAK",
    4: "BEAR_STRONG",
    5: "VOLATILE"
}


def load_model_and_scaler(config_path: str = 'config.yaml'):
    """Load trained model and preprocessing objects"""
    print("🔧 Loading model and scaler...")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"📱 Using device: {device}")
    
    # Load scaler
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    # Load model
    model = NatronTransformer(
        num_features=config['data']['feature_count'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        activation=config['model']['activation'],
        buy_head_dims=config['model']['buy_head_dims'],
        sell_head_dims=config['model']['sell_head_dims'],
        direction_head_dims=config['model']['direction_head_dims'],
        regime_head_dims=config['model']['regime_head_dims'],
        sequence_length=config['data']['sequence_length']
    ).to(device)
    
    checkpoint = torch.load('models/natron_v2.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ Model and scaler loaded")
    
    return model, scaler, config, device


def predict_single(
    model,
    df: pd.DataFrame,
    feature_engine: FeatureEngine,
    scaler,
    device,
    sequence_length: int = 96
) -> Dict:
    """Generate prediction for a single sequence"""
    
    # Create inference sequence
    sequence = create_inference_sequence(
        df,
        feature_engine,
        scaler,
        sequence_length=sequence_length
    )
    
    sequence = sequence.to(device)
    
    # Get predictions
    with torch.no_grad():
        predictions = model.get_predictions(sequence)
    
    # Extract values
    buy_prob = float(predictions['buy_prob'][0])
    sell_prob = float(predictions['sell_prob'][0])
    direction_prob = predictions['direction_prob'][0].cpu().numpy()
    regime_prob = predictions['regime_prob'][0].cpu().numpy()
    
    direction_up = float(direction_prob[1])
    regime_id = int(predictions['regime_pred'][0])
    regime_name = REGIME_MAP[regime_id]
    
    confidence = max(buy_prob, sell_prob, direction_up, 1 - direction_up)
    
    return {
        'buy_prob': buy_prob,
        'sell_prob': sell_prob,
        'direction_up': direction_up,
        'direction_down': 1 - direction_up,
        'regime': regime_name,
        'regime_id': regime_id,
        'regime_probs': {REGIME_MAP[i]: float(regime_prob[i]) for i in range(len(regime_prob))},
        'confidence': confidence
    }


def predict_from_csv(csv_path: str):
    """Run inference on CSV file"""
    print(f"\n📂 Loading data from {csv_path}...")
    
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} rows")
    
    # Load model
    model, scaler, config, device = load_model_and_scaler()
    
    # Feature engine
    feature_engine = FeatureEngine(verbose=False)
    
    # Generate prediction
    print("\n🔮 Generating prediction...")
    result = predict_single(
        model,
        df,
        feature_engine,
        scaler,
        device,
        sequence_length=config['data']['sequence_length']
    )
    
    # Print results
    print("\n" + "="*60)
    print("🎯 NATRON PREDICTION")
    print("="*60)
    print(f"\n📊 Trading Signals:")
    print(f"   BUY Probability:  {result['buy_prob']*100:.2f}%")
    print(f"   SELL Probability: {result['sell_prob']*100:.2f}%")
    print(f"\n📈 Direction:")
    print(f"   UP:   {result['direction_up']*100:.2f}%")
    print(f"   DOWN: {result['direction_down']*100:.2f}%")
    print(f"\n🌐 Market Regime: {result['regime']}")
    print(f"   Confidence: {result['confidence']*100:.2f}%")
    print(f"\n📊 Regime Probabilities:")
    for regime, prob in result['regime_probs'].items():
        bar = "█" * int(prob * 30)
        print(f"   {regime:15s} {prob*100:5.1f}% {bar}")
    
    print("\n" + "="*60)
    
    # Trading recommendation
    print("\n💡 Trading Recommendation:")
    if result['buy_prob'] > 0.65:
        print("   🟢 STRONG BUY SIGNAL")
    elif result['buy_prob'] > 0.55:
        print("   🟡 MODERATE BUY SIGNAL")
    elif result['sell_prob'] > 0.65:
        print("   🔴 STRONG SELL SIGNAL")
    elif result['sell_prob'] > 0.55:
        print("   🟠 MODERATE SELL SIGNAL")
    else:
        print("   ⚪ NEUTRAL - No clear signal")
    
    print("\n" + "="*60 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Natron Inference')
    parser.add_argument('--csv', type=str, help='Path to CSV file with OHLCV data')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file')
    
    args = parser.parse_args()
    
    if args.csv:
        predict_from_csv(args.csv)
    else:
        print("❌ Please provide --csv argument with path to OHLCV data")
        print("   Example: python inference.py --csv data/test_data.csv")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
