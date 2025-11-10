"""
Test inference script - Verify model works end-to-end
"""
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from config import Config
from feature_engine import FeatureEngine
from label_generator import LabelGenerator
from model import NatronTransformer


def test_inference():
    """Test model inference with sample data"""
    print("="*70)
    print("NATRON TRANSFORMER - INFERENCE TEST")
    print("="*70)
    
    # Check if model exists
    if not Config.MODEL_PATH.exists():
        print(f"⚠ Model not found at {Config.MODEL_PATH}")
        print("Please train the model first using: python train.py")
        return
    
    # Setup device
    device = torch.device(Config.DEVICE)
    if not torch.cuda.is_available() and device.type == 'cuda':
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = NatronTransformer(
        num_features=Config.NUM_FEATURES,
        d_model=Config.D_MODEL,
        n_heads=Config.N_HEADS,
        n_layers=Config.N_LAYERS,
        d_ff=Config.D_FF,
        dropout=Config.DROPOUT,
        sequence_length=Config.SEQUENCE_LENGTH
    ).to(device)
    
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()
    print("✓ Model loaded")
    
    # Initialize feature engine
    feature_engine = FeatureEngine()
    label_generator = LabelGenerator()
    
    # Create sample data
    print("\nGenerating sample data...")
    np.random.seed(42)
    base_price = 100.0
    prices = [base_price]
    for i in range(Config.SEQUENCE_LENGTH):
        change = np.random.normal(0.01, 0.5)
        prices.append(max(prices[-1] + change, 1.0))
    
    # Create OHLCV DataFrame
    data = []
    for i in range(Config.SEQUENCE_LENGTH):
        close = prices[i]
        volatility = np.random.uniform(0.1, 0.5)
        high = close + np.random.uniform(0, volatility)
        low = close - np.random.uniform(0, volatility)
        open_price = prices[i-1] if i > 0 else close
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        data.append({
            'time': f'2024-01-01 {i:02d}:00:00',
            'open': round(open_price, 5),
            'high': round(high, 5),
            'low': round(low, 5),
            'close': round(close, 5),
            'volume': int(np.random.uniform(1000, 10000))
        })
    
    df = pd.DataFrame(data)
    print(f"✓ Created {len(df)} sample candles")
    
    # Generate features
    print("Generating features...")
    features_df = feature_engine.generate_features(df)
    print(f"✓ Generated {features_df.shape[1]} features")
    
    # Convert to tensor
    features_array = features_df.values.astype(np.float32)
    features_tensor = torch.FloatTensor(features_array).unsqueeze(0).to(device)
    
    # Inference
    print("\nRunning inference...")
    with torch.no_grad():
        predictions = model(features_tensor)
    
    # Extract results
    buy_prob = predictions['buy'][0].item()
    sell_prob = predictions['sell'][0].item()
    direction_probs = torch.softmax(predictions['direction'][0], dim=0)
    direction_up_prob = direction_probs[1].item()
    regime_probs = torch.softmax(predictions['regime'][0], dim=0)
    regime_idx = regime_probs.argmax().item()
    regime_name = label_generator.regime_names[regime_idx]
    regime_confidence = regime_probs[regime_idx].item()
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"Buy Probability:     {buy_prob:.4f}")
    print(f"Sell Probability:    {sell_prob:.4f}")
    print(f"Direction Up:        {direction_up_prob:.4f}")
    print(f"Regime:              {regime_name} (confidence: {regime_confidence:.4f})")
    print(f"\nRegime Probabilities:")
    for i, name in enumerate(label_generator.regime_names):
        prob = regime_probs[i].item()
        marker = " ←" if i == regime_idx else ""
        print(f"  {name:15s}: {prob:.4f}{marker}")
    print("="*70)
    
    print("\n✓ Inference test completed successfully!")


if __name__ == "__main__":
    test_inference()
