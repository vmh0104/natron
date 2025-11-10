#!/usr/bin/env python3
"""
Natron System Test Script
Tests all components to verify installation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported"""
    print("\n🧪 Testing imports...")
    
    try:
        from src.features.feature_engine import FeatureEngine
        print("   ✅ FeatureEngine")
    except Exception as e:
        print(f"   ❌ FeatureEngine: {e}")
        return False
    
    try:
        from src.labels.label_generator import LabelGenerator
        print("   ✅ LabelGenerator")
    except Exception as e:
        print(f"   ❌ LabelGenerator: {e}")
        return False
    
    try:
        from src.data.dataset_loader import NatronDataModule
        print("   ✅ NatronDataModule")
    except Exception as e:
        print(f"   ❌ NatronDataModule: {e}")
        return False
    
    try:
        from src.models.natron_transformer import NatronTransformer
        print("   ✅ NatronTransformer")
    except Exception as e:
        print(f"   ❌ NatronTransformer: {e}")
        return False
    
    try:
        from src.models.losses import MultiTaskLoss
        print("   ✅ MultiTaskLoss")
    except Exception as e:
        print(f"   ❌ MultiTaskLoss: {e}")
        return False
    
    return True


def test_pytorch():
    """Test PyTorch and CUDA"""
    print("\n🧪 Testing PyTorch...")
    
    try:
        import torch
        print(f"   ✅ PyTorch version: {torch.__version__}")
        print(f"   ✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"   ✅ CUDA version: {torch.version.cuda}")
            print(f"   ✅ GPU: {torch.cuda.get_device_name(0)}")
        
        return True
    except Exception as e:
        print(f"   ❌ PyTorch error: {e}")
        return False


def test_feature_engine():
    """Test feature generation"""
    print("\n🧪 Testing Feature Engine...")
    
    try:
        import numpy as np
        import pandas as pd
        from src.features.feature_engine import FeatureEngine
        
        # Generate sample data
        n = 1000
        df = pd.DataFrame({
            'time': pd.date_range('2023-01-01', periods=n, freq='15min'),
            'open': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'high': 101 + np.cumsum(np.random.randn(n) * 0.1),
            'low': 99 + np.cumsum(np.random.randn(n) * 0.1),
            'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'volume': np.random.randint(1000, 10000, n)
        })
        
        engine = FeatureEngine(verbose=False)
        features = engine.generate_all_features(df)
        
        print(f"   ✅ Generated {features.shape[1]} features")
        print(f"   ✅ Shape: {features.shape}")
        
        return True
    except Exception as e:
        print(f"   ❌ Feature Engine error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_label_generator():
    """Test label generation"""
    print("\n🧪 Testing Label Generator...")
    
    try:
        import numpy as np
        import pandas as pd
        from src.features.feature_engine import FeatureEngine
        from src.labels.label_generator import LabelGenerator
        
        # Generate sample data
        n = 1000
        df = pd.DataFrame({
            'time': pd.date_range('2023-01-01', periods=n, freq='15min'),
            'open': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'high': 101 + np.cumsum(np.random.randn(n) * 0.1),
            'low': 99 + np.cumsum(np.random.randn(n) * 0.1),
            'close': 100 + np.cumsum(np.random.randn(n) * 0.1),
            'volume': np.random.randint(1000, 10000, n)
        })
        
        engine = FeatureEngine(verbose=False)
        features = engine.generate_all_features(df)
        
        labeler = LabelGenerator(verbose=False)
        labels = labeler.generate_all_labels(df, features)
        
        print(f"   ✅ Generated labels shape: {labels.shape}")
        print(f"   ✅ Label columns: {list(labels.columns)}")
        
        return True
    except Exception as e:
        print(f"   ❌ Label Generator error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model():
    """Test model creation"""
    print("\n🧪 Testing Model...")
    
    try:
        import torch
        from src.models.natron_transformer import NatronTransformer
        
        model = NatronTransformer(
            num_features=100,
            d_model=256,
            nhead=8,
            num_encoder_layers=6
        )
        
        # Test forward pass
        batch_size = 4
        seq_len = 96
        x = torch.randn(batch_size, seq_len, 100)
        
        outputs = model(x)
        
        print(f"   ✅ Model created")
        print(f"   ✅ Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   ✅ Forward pass successful")
        print(f"   ✅ Output shapes:")
        print(f"      - buy_prob: {outputs['buy_prob'].shape}")
        print(f"      - sell_prob: {outputs['sell_prob'].shape}")
        print(f"      - direction_logits: {outputs['direction_logits'].shape}")
        print(f"      - regime_logits: {outputs['regime_logits'].shape}")
        
        return True
    except Exception as e:
        print(f"   ❌ Model error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration file"""
    print("\n🧪 Testing Configuration...")
    
    try:
        import yaml
        
        with open('config/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        print("   ✅ Config loaded")
        print(f"   ✅ Sections: {list(config.keys())}")
        
        return True
    except Exception as e:
        print(f"   ❌ Config error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 80)
    print("🧠 NATRON TRANSFORMER - SYSTEM TEST")
    print("=" * 80)
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    results['pytorch'] = test_pytorch()
    results['config'] = test_config()
    results['feature_engine'] = test_feature_engine()
    results['label_generator'] = test_label_generator()
    results['model'] = test_model()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name:20s}: {status}")
    
    print("\n" + "=" * 80)
    print(f"   TOTAL: {passed}/{total} tests passed")
    print("=" * 80)
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready.")
        print("\n📋 Next steps:")
        print("   1. Place your data in: data/data_export.csv")
        print("   2. Train models: python scripts/train_full_pipeline.py")
        print("   3. Start server: ./scripts/start_server.sh")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
