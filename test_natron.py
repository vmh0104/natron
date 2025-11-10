"""
Test Script for Natron System
Verifies all components are working correctly
"""

import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

print("="*60)
print("NATRON SYSTEM TEST")
print("="*60)

# Test 1: Feature Engineering
print("\n[1/6] Testing Feature Engineering...")
try:
    from feature_engine import FeatureEngine
    
    # Create dummy data
    dates = pd.date_range('2024-01-01', periods=200, freq='15min')
    df = pd.DataFrame({
        'time': dates,
        'open': 1.1000 + np.cumsum(np.random.randn(200) * 0.001),
        'high': 1.1050 + np.cumsum(np.random.randn(200) * 0.001),
        'low': 1.0990 + np.cumsum(np.random.randn(200) * 0.001),
        'close': 1.1030 + np.cumsum(np.random.randn(200) * 0.001),
        'volume': np.random.randint(100, 1000, 200)
    })
    
    engine = FeatureEngine()
    features_df = engine.fit_transform(df)
    feature_cols = engine.get_feature_columns(features_df)
    
    print(f"  ✓ Generated {len(feature_cols)} features")
    print(f"  ✓ Feature shape: {features_df.shape}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    exit(1)

# Test 2: Label Generation
print("\n[2/6] Testing Label Generation...")
try:
    from label_generator import LabelGenerator
    
    generator = LabelGenerator()
    labels_df = generator.generate_labels(features_df)
    
    stats = generator.get_label_stats(labels_df)
    print(f"  ✓ Buy rate: {stats['buy_rate']:.3f}")
    print(f"  ✓ Sell rate: {stats['sell_rate']:.3f}")
    print(f"  ✓ Direction up rate: {stats['direction_up_rate']:.3f}")
    print(f"  ✓ Regime distribution: {len(stats['regime_distribution'])} classes")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    exit(1)

# Test 3: Dataset Loader
print("\n[3/6] Testing Dataset Loader...")
try:
    from dataset_loader import SequenceCreator
    
    creator = SequenceCreator(sequence_length=96)
    train_ds, val_ds, test_ds = creator.create_dataset(df)
    
    print(f"  ✓ Train samples: {len(train_ds)}")
    print(f"  ✓ Val samples: {len(val_ds)}")
    print(f"  ✓ Test samples: {len(test_ds)}")
    
    # Test data loading
    X, labels = train_ds[0]
    print(f"  ✓ Sample shape: {X.shape}")
    print(f"  ✓ Labels: {list(labels.keys())}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    exit(1)

# Test 4: Model Architecture
print("\n[4/6] Testing Model Architecture...")
try:
    from model_natron import NatronTransformer
    
    num_features = len(feature_cols)
    model = NatronTransformer(
        num_features=num_features,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        sequence_length=96
    )
    
    # Test forward pass
    x = torch.randn(2, 96, num_features)
    outputs = model(x)
    
    print(f"  ✓ Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  ✓ Buy output shape: {outputs['buy'].shape}")
    print(f"  ✓ Sell output shape: {outputs['sell'].shape}")
    print(f"  ✓ Direction output shape: {outputs['direction'].shape}")
    print(f"  ✓ Regime output shape: {outputs['regime'].shape}")
    
    # Test prediction
    pred = model.predict(x[0])
    print(f"  ✓ Prediction keys: {list(pred.keys())}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Test 5: Loss Functions
print("\n[5/6] Testing Loss Functions...")
try:
    from losses import MultiTaskLoss
    
    criterion = MultiTaskLoss()
    
    # Create dummy predictions and targets
    predictions = {
        'buy': torch.sigmoid(torch.randn(2, 1)),
        'sell': torch.sigmoid(torch.randn(2, 1)),
        'direction': torch.randn(2, 2),
        'regime': torch.randn(2, 6)
    }
    
    targets = {
        'buy': torch.randint(0, 2, (2, 1)).float(),
        'sell': torch.randint(0, 2, (2, 1)).float(),
        'direction': torch.randint(0, 2, (2,)),
        'regime': torch.randint(0, 6, (2,))
    }
    
    loss_dict = criterion(predictions, targets)
    print(f"  ✓ Total loss: {loss_dict['total_loss']:.4f}")
    print(f"  ✓ Buy loss: {loss_dict['buy_loss']:.4f}")
    print(f"  ✓ Sell loss: {loss_dict['sell_loss']:.4f}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    exit(1)

# Test 6: Pretrain Model
print("\n[6/6] Testing Pretrain Model...")
try:
    from model_natron import NatronPretrainModel
    
    pretrain_model = NatronPretrainModel(
        num_features=num_features,
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        sequence_length=96
    )
    
    x = torch.randn(2, 96, num_features)
    outputs = pretrain_model(x)
    
    print(f"  ✓ Reconstruction shape: {outputs['reconstruction'].shape}")
    print(f"  ✓ Contrastive embedding shape: {outputs['contrastive_embedding'].shape}")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    exit(1)

print("\n" + "="*60)
print("ALL TESTS PASSED! ✓")
print("="*60)
print("\nSystem is ready for training and deployment.")
print("\nNext steps:")
print("  1. Prepare your data_export.csv file")
print("  2. Run: python train_pretrain.py --data data_export.csv")
print("  3. Run: python train_natron.py --data data_export.csv --pretrain_model ./models/natron_pretrain_best.pt")
print("  4. Start servers: ./start_natron.sh")
