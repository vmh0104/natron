"""
Example Usage: How to use Natron Transformer components
"""
import pandas as pd
import numpy as np
import torch
from feature_engine import FeatureEngine
from label_generator import LabelGenerator
from sequence_creator import SequenceCreator
from model import NatronModel


def example_feature_generation():
    """Example: Generate features from OHLCV data"""
    print("="*50)
    print("Example 1: Feature Generation")
    print("="*50)
    
    # Load or create sample data
    df = pd.read_csv('data_export.csv')
    
    # Initialize feature engine
    engine = FeatureEngine()
    
    # Generate features
    df_features = engine.generate_features(df)
    feature_cols = engine.get_feature_columns(df_features)
    
    print(f"Original columns: {list(df.columns)}")
    print(f"Total features generated: {len(feature_cols)}")
    print(f"Sample features: {feature_cols[:10]}")
    
    return df_features, feature_cols


def example_label_generation(df_features):
    """Example: Generate trading labels"""
    print("\n" + "="*50)
    print("Example 2: Label Generation")
    print("="*50)
    
    # Initialize label generator
    generator = LabelGenerator()
    
    # Generate labels
    df_labeled = generator.generate_labels(df_features)
    
    print(f"Buy signals: {df_labeled['buy'].sum()} ({df_labeled['buy'].mean()*100:.2f}%)")
    print(f"Sell signals: {df_labeled['sell'].sum()} ({df_labeled['sell'].mean()*100:.2f}%)")
    print(f"\nDirection distribution:")
    print(df_labeled['direction'].value_counts())
    print(f"\nRegime distribution:")
    print(df_labeled['regime'].value_counts())
    
    # Get regime names
    for regime_id in df_labeled['regime'].unique():
        regime_name = generator.get_regime_name(regime_id)
        count = (df_labeled['regime'] == regime_id).sum()
        print(f"  {regime_id}: {regime_name} ({count} samples)")
    
    return df_labeled


def example_sequence_creation(df_labeled, feature_cols):
    """Example: Create sequences for training"""
    print("\n" + "="*50)
    print("Example 3: Sequence Creation")
    print("="*50)
    
    # Initialize sequence creator
    creator = SequenceCreator(sequence_length=96)
    
    # Create sequences with train/val/test splits
    X_splits, y_splits = creator.create_sequences_with_splits(
        df=df_labeled,
        feature_columns=feature_cols,
        label_columns=['buy', 'sell', 'direction', 'regime'],
        train_ratio=0.7,
        val_ratio=0.15
    )
    
    print(f"Train set: {X_splits['train'].shape}")
    print(f"Val set: {X_splits['val'].shape}")
    print(f"Test set: {X_splits['test'].shape}")
    print(f"\nLabels:")
    for split_name in ['train', 'val', 'test']:
        print(f"  {split_name}:")
        for label_name, label_array in y_splits[split_name].items():
            print(f"    {label_name}: shape {label_array.shape}")
    
    return X_splits, y_splits


def example_model_inference(X_sample):
    """Example: Model inference"""
    print("\n" + "="*50)
    print("Example 4: Model Inference")
    print("="*50)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize model
    model = NatronModel(
        input_dim=X_sample.shape[-1],
        d_model=256,
        nhead=8,
        num_layers=6
    ).to(device)
    
    # Prepare input (single sequence)
    input_tensor = torch.FloatTensor(X_sample[:1]).to(device)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
    
    # Display predictions
    print("\nPredictions:")
    print(f"  Buy probability: {outputs['buy'].item():.4f}")
    print(f"  Sell probability: {outputs['sell'].item():.4f}")
    
    direction_probs = outputs['direction'].cpu().numpy()[0]
    print(f"  Direction - Down: {direction_probs[0]:.4f}, Up: {direction_probs[1]:.4f}")
    
    regime_probs = outputs['regime'].cpu().numpy()[0]
    regime_names = ['BULL_STRONG', 'BULL_WEAK', 'RANGE', 'BEAR_WEAK', 'BEAR_STRONG', 'VOLATILE']
    regime_id = np.argmax(regime_probs)
    print(f"  Regime: {regime_names[regime_id]} (prob: {regime_probs[regime_id]:.4f})")
    print(f"    All regime probabilities:")
    for i, (name, prob) in enumerate(zip(regime_names, regime_probs)):
        print(f"      {i}: {name}: {prob:.4f}")
    
    # Calculate confidence
    confidence = max(
        outputs['buy'].item(),
        outputs['sell'].item(),
        direction_probs.max(),
        regime_probs.max()
    )
    print(f"\n  Overall confidence: {confidence:.4f}")
    
    return outputs


def example_api_format():
    """Example: API request/response format"""
    print("\n" + "="*50)
    print("Example 5: API Format")
    print("="*50)
    
    print("Request format (POST /predict):")
    request_example = {
        "candles": [
            {
                "time": "2024-01-01 00:00:00",
                "open": 100.0,
                "high": 100.5,
                "low": 99.8,
                "close": 100.2,
                "volume": 1000
            },
            # ... 95 more candles
        ]
    }
    print("  " + str(request_example)[:200] + "...")
    
    print("\nResponse format:")
    response_example = {
        "buy_prob": 0.71,
        "sell_prob": 0.24,
        "direction_up": 0.69,
        "direction_down": 0.31,
        "regime": "BULL_WEAK",
        "regime_id": 1,
        "regime_probs": [0.1, 0.4, 0.2, 0.1, 0.1, 0.1],
        "confidence": 0.82
    }
    print("  " + str(response_example))


def example_mql5_format():
    """Example: MQL5 socket communication format"""
    print("\n" + "="*50)
    print("Example 6: MQL5 Socket Format")
    print("="*50)
    
    print("MQL5 → Python (PREDICT):")
    mql5_request = {
        "type": "PREDICT",
        "candles": [
            {"time": "2024-01-01 00:00", "open": 100.0, "high": 100.5, 
             "low": 99.8, "close": 100.2, "volume": 1000},
            # ... more candles
        ]
    }
    print("  " + str(mql5_request)[:150] + "...")
    
    print("\nPython → MQL5 (PREDICTION):")
    python_response = {
        "type": "PREDICTION",
        "data": {
            "buy_prob": 0.71,
            "sell_prob": 0.24,
            "direction_up": 0.69,
            "regime": "BULL_WEAK",
            "confidence": 0.82
        }
    }
    print("  " + str(python_response))
    
    print("\nMQL5 → Python (ADD_CANDLE):")
    add_candle = {
        "type": "ADD_CANDLE",
        "candle": {
            "time": "2024-01-01 00:15",
            "open": 100.2,
            "high": 100.7,
            "low": 100.0,
            "close": 100.5,
            "volume": 1200
        }
    }
    print("  " + str(add_candle))


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("NATRON TRANSFORMER - USAGE EXAMPLES")
    print("="*70)
    
    try:
        # Example 1: Feature generation
        df_features, feature_cols = example_feature_generation()
        
        # Example 2: Label generation
        df_labeled = example_label_generation(df_features)
        
        # Example 3: Sequence creation
        X_splits, y_splits = example_sequence_creation(df_labeled, feature_cols)
        
        # Example 4: Model inference
        outputs = example_model_inference(X_splits['train'][:1])
        
        # Example 5: API format
        example_api_format()
        
        # Example 6: MQL5 format
        example_mql5_format()
        
        print("\n" + "="*70)
        print("ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        print("Please run: python generate_sample_data.py")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
