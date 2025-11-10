"""
Test script to verify pipeline components
"""
import numpy as np
import pandas as pd
import torch
from feature_engine import FeatureEngine
from label_generator import LabelGenerator
from sequence_creator import SequenceCreator
from model import NatronModel


def test_feature_engine():
    """Test feature generation"""
    print("Testing FeatureEngine...")
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=200, freq='15min')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(200) * 0.5)
    
    df = pd.DataFrame({
        'time': dates,
        'open': prices + np.random.randn(200) * 0.1,
        'high': prices + np.abs(np.random.randn(200) * 0.2),
        'low': prices - np.abs(np.random.randn(200) * 0.2),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 200)
    })
    
    # Generate features
    engine = FeatureEngine()
    df_features = engine.generate_features(df)
    feature_cols = engine.get_feature_columns(df_features)
    
    print(f"✓ Generated {len(feature_cols)} features")
    print(f"  Sample features: {feature_cols[:5]}")
    
    return df_features, feature_cols


def test_label_generator(df_features):
    """Test label generation"""
    print("\nTesting LabelGenerator...")
    
    generator = LabelGenerator()
    df_labeled = generator.generate_labels(df_features)
    
    print(f"✓ Generated labels:")
    print(f"  Buy signals: {df_labeled['buy'].sum()}")
    print(f"  Sell signals: {df_labeled['sell'].sum()}")
    print(f"  Direction distribution: {df_labeled['direction'].value_counts().to_dict()}")
    print(f"  Regime distribution: {df_labeled['regime'].value_counts().to_dict()}")
    
    return df_labeled


def test_sequence_creator(df_labeled, feature_cols):
    """Test sequence creation"""
    print("\nTesting SequenceCreator...")
    
    creator = SequenceCreator(sequence_length=96)
    X, y = creator.create_sequences(
        df_labeled,
        feature_cols,
        label_columns=['buy', 'sell', 'direction', 'regime']
    )
    
    print(f"✓ Created sequences:")
    print(f"  X shape: {X.shape}")
    print(f"  Labels: {list(y.keys())}")
    for key, arr in y.items():
        print(f"    {key}: shape {arr.shape}, dtype {arr.dtype}")
    
    return X, y


def test_model(X_sample):
    """Test model forward pass"""
    print("\nTesting NatronModel...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Using device: {device}")
    
    model = NatronModel(
        input_dim=X_sample.shape[-1],
        d_model=256,
        nhead=8,
        num_layers=2  # Reduced for testing
    ).to(device)
    
    # Test forward pass
    X_tensor = torch.FloatTensor(X_sample[:1]).to(device)
    outputs = model(X_tensor)
    
    print(f"✓ Model forward pass successful:")
    print(f"  Buy prob: {outputs['buy'].item():.4f}")
    print(f"  Sell prob: {outputs['sell'].item():.4f}")
    print(f"  Direction probs: {outputs['direction'].cpu().numpy()[0]}")
    print(f"  Regime probs: {outputs['regime'].cpu().numpy()[0]}")
    
    return model


def main():
    """Run all tests"""
    print("="*50)
    print("NATRON PIPELINE TEST")
    print("="*50)
    
    try:
        # Test feature engine
        df_features, feature_cols = test_feature_engine()
        
        # Test label generator
        df_labeled = test_label_generator(df_features)
        
        # Test sequence creator
        X, y = test_sequence_creator(df_labeled, feature_cols)
        
        # Test model
        model = test_model(X)
        
        print("\n" + "="*50)
        print("✓ ALL TESTS PASSED")
        print("="*50)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
