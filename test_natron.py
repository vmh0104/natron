"""
Test script for Natron components
"""
import numpy as np
import pandas as pd
import sys
import os

# Add natron to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from natron.feature_engine import FeatureEngine
from natron.label_generator import LabelGenerator
from natron.sequence_creator import SequenceCreator


def generate_sample_data(n=200):
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    
    # Generate random walk prices
    base_price = 1.1000
    returns = np.random.randn(n) * 0.001
    prices = base_price + np.cumsum(returns)
    
    # Generate OHLCV
    data = []
    for i in range(n):
        open_price = prices[i]
        close_price = prices[i] + np.random.randn() * 0.0005
        high_price = max(open_price, close_price) + abs(np.random.randn() * 0.0003)
        low_price = min(open_price, close_price) - abs(np.random.randn() * 0.0003)
        volume = np.random.randint(500, 2000)
        
        time_str = pd.Timestamp('2024-01-01') + pd.Timedelta(minutes=15*i)
        
        data.append({
            'time': time_str.strftime('%Y-%m-%d %H:%M:%S'),
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    return pd.DataFrame(data)


def test_feature_engine():
    """Test feature engineering."""
    print("Testing FeatureEngine...")
    df = generate_sample_data(200)
    
    engine = FeatureEngine()
    df_features = engine.generate_features(df)
    feature_cols = engine.get_feature_columns(df_features)
    
    print(f"  ✓ Generated {len(feature_cols)} features")
    print(f"  ✓ Feature columns: {feature_cols[:5]}... (showing first 5)")
    
    assert len(feature_cols) >= 90, f"Expected at least 90 features, got {len(feature_cols)}"
    return df_features


def test_label_generator(df_features):
    """Test label generation."""
    print("\nTesting LabelGenerator...")
    
    generator = LabelGenerator()
    df_labeled = generator.generate_labels(df_features)
    
    print(f"  ✓ Buy signals: {df_labeled['buy'].sum()}")
    print(f"  ✓ Sell signals: {df_labeled['sell'].sum()}")
    print(f"  ✓ Direction up: {(df_labeled['direction']==1).sum()}")
    print(f"  ✓ Regime distribution:")
    for i in range(6):
        count = (df_labeled['regime'] == i).sum()
        name = generator.get_regime_name(i)
        print(f"    - {name}: {count}")
    
    assert 'buy' in df_labeled.columns
    assert 'sell' in df_labeled.columns
    assert 'direction' in df_labeled.columns
    assert 'regime' in df_labeled.columns
    
    return df_labeled


def test_sequence_creator(df_labeled):
    """Test sequence creation."""
    print("\nTesting SequenceCreator...")
    
    creator = SequenceCreator(sequence_length=96)
    
    # Get feature columns
    base_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 
                'buy', 'sell', 'direction', 'regime']
    feature_cols = [col for col in df_labeled.columns if col not in base_cols]
    
    X, y = creator.create_sequences(df_labeled, feature_cols)
    
    print(f"  ✓ Created {len(X)} sequences")
    print(f"  ✓ X shape: {X.shape}")
    print(f"  ✓ y keys: {list(y.keys())}")
    print(f"  ✓ y shapes: {[v.shape for v in y.values()]}")
    
    assert X.shape[0] > 0, "No sequences created"
    assert X.shape[1] == 96, f"Expected sequence length 96, got {X.shape[1]}"
    assert X.shape[2] == len(feature_cols), "Feature dimension mismatch"
    
    return X, y


def test_model():
    """Test model creation."""
    print("\nTesting Model...")
    
    try:
        import torch
        from natron.model import create_model
        
        config = {
            'd_model': 64,  # Smaller for testing
            'nhead': 4,
            'num_layers': 2,
            'dim_feedforward': 256,
            'dropout': 0.1,
            'num_features': 100,
            'max_seq_len': 96
        }
        
        model = create_model(config)
        num_params = sum(p.numel() for p in model.parameters())
        
        print(f"  ✓ Model created with {num_params:,} parameters")
        
        # Test forward pass
        x = torch.randn(2, 96, 100)  # batch_size=2, seq_len=96, features=100
        outputs = model(x)
        
        print(f"  ✓ Forward pass successful")
        print(f"  ✓ Outputs: {list(outputs.keys())}")
        print(f"  ✓ Buy prob shape: {outputs['buy'].shape}")
        print(f"  ✓ Direction shape: {outputs['direction'].shape}")
        print(f"  ✓ Regime shape: {outputs['regime'].shape}")
        
        assert outputs['buy'].shape[0] == 2
        assert outputs['direction'].shape == (2, 2)
        assert outputs['regime'].shape == (2, 6)
        
    except ImportError:
        print("  ⚠ PyTorch not available, skipping model test")
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        raise


if __name__ == '__main__':
    print("=" * 60)
    print("Natron Component Tests")
    print("=" * 60)
    
    try:
        # Test feature engineering
        df_features = test_feature_engine()
        
        # Test label generation
        df_labeled = test_label_generator(df_features)
        
        # Test sequence creation
        X, y = test_sequence_creator(df_labeled)
        
        # Test model
        test_model()
        
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
