"""
Test script for Natron components
"""

import numpy as np
import pandas as pd
import torch
from feature_engine import FeatureEngine
from label_generator import LabelGenerator
from dataset_loader import SequenceCreator
from model_natron import NatronTransformer


def test_feature_engine():
    """Test feature generation"""
    print("Testing FeatureEngine...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=200, freq='15min')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'time': dates,
        'open': 1.1000 + np.cumsum(np.random.randn(200) * 0.0001),
        'high': 1.1000 + np.cumsum(np.random.randn(200) * 0.0001) + 0.0010,
        'low': 1.1000 + np.cumsum(np.random.randn(200) * 0.0001) - 0.0005,
        'close': 1.1000 + np.cumsum(np.random.randn(200) * 0.0001) + 0.0005,
        'volume': np.random.randint(1000, 10000, 200)
    })
    
    engine = FeatureEngine()
    features_df = engine.generate_all_features(df)
    
    print(f"  ✓ Generated {features_df.shape[1]} features")
    print(f"  ✓ Feature shape: {features_df.shape}")
    assert features_df.shape[1] >= 90, "Should generate at least 90 features"
    
    return df, features_df


def test_label_generator(df, features_df):
    """Test label generation"""
    print("\nTesting LabelGenerator...")
    
    generator = LabelGenerator()
    labels_df = generator.generate_all_labels(df, features_df)
    
    print(f"  ✓ Generated labels: {labels_df.columns.tolist()}")
    print(f"  ✓ Buy signals: {labels_df['buy'].sum()}/{len(labels_df)} ({labels_df['buy'].mean()*100:.2f}%)")
    print(f"  ✓ Sell signals: {labels_df['sell'].sum()}/{len(labels_df)} ({labels_df['sell'].mean()*100:.2f}%)")
    print(f"  ✓ Regime distribution: {labels_df['regime'].value_counts().to_dict()}")
    
    assert 'buy' in labels_df.columns
    assert 'sell' in labels_df.columns
    assert 'direction' in labels_df.columns
    assert 'regime' in labels_df.columns
    
    return labels_df


def test_sequence_creator(features_df, labels_df):
    """Test sequence creation"""
    print("\nTesting SequenceCreator...")
    
    creator = SequenceCreator(sequence_length=96, feature_dim=features_df.shape[1])
    X, y_buy, y_sell, y_direction, y_regime = creator.create_sequences(features_df, labels_df)
    
    print(f"  ✓ Created {len(X)} sequences")
    print(f"  ✓ X shape: {X.shape}")
    print(f"  ✓ y_buy shape: {y_buy.shape}")
    print(f"  ✓ y_sell shape: {y_sell.shape}")
    print(f"  ✓ y_direction shape: {y_direction.shape}")
    print(f"  ✓ y_regime shape: {y_regime.shape}")
    
    assert X.shape[1] == 96, "Sequence length should be 96"
    assert X.shape[2] == features_df.shape[1], "Feature dimension should match"
    
    return X, y_buy, y_sell, y_direction, y_regime


def test_model(X):
    """Test model forward pass"""
    print("\nTesting NatronTransformer...")
    
    model = NatronTransformer(
        input_dim=X.shape[2],
        sequence_length=X.shape[1],
        d_model=128,
        nhead=8,
        num_layers=2,  # Smaller for testing
        dim_feedforward=512,
        dropout=0.1
    )
    
    # Test forward pass
    batch = torch.FloatTensor(X[:2])  # Take 2 samples
    predictions = model(batch)
    
    print(f"  ✓ Buy predictions shape: {predictions['buy'].shape}")
    print(f"  ✓ Sell predictions shape: {predictions['sell'].shape}")
    print(f"  ✓ Direction predictions shape: {predictions['direction'].shape}")
    print(f"  ✓ Regime predictions shape: {predictions['regime'].shape}")
    
    assert predictions['buy'].shape[0] == 2
    assert predictions['sell'].shape[0] == 2
    assert predictions['direction'].shape == (2, 2)
    assert predictions['regime'].shape == (2, 6)
    
    print(f"  ✓ Sample buy prob: {predictions['buy'][0].item():.4f}")
    print(f"  ✓ Sample sell prob: {predictions['sell'][0].item():.4f}")
    
    return model


def main():
    print("=" * 50)
    print("Natron Component Tests")
    print("=" * 50)
    
    try:
        # Test feature engine
        df, features_df = test_feature_engine()
        
        # Test label generator
        labels_df = test_label_generator(df, features_df)
        
        # Test sequence creator
        X, y_buy, y_sell, y_direction, y_regime = test_sequence_creator(features_df, labels_df)
        
        # Test model
        model = test_model(X)
        
        print("\n" + "=" * 50)
        print("✓ All tests passed!")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
