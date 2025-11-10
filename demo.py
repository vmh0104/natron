"""
Natron Demo Script - Generate sample data and test pipeline
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

from feature_engine import FeatureEngine
from label_generator import LabelGenerator
from sequence_creator import SequenceCreator


def generate_sample_data(n_candles: int = 1000) -> pd.DataFrame:
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)
    
    # Generate timestamps
    start_time = datetime.now() - timedelta(days=n_candles // 96)
    timestamps = [start_time + timedelta(minutes=15 * i) for i in range(n_candles)]
    
    # Generate price data (random walk)
    base_price = 100.0
    prices = [base_price]
    
    for i in range(1, n_candles):
        change = np.random.normal(0, 0.5)
        new_price = prices[-1] + change
        prices.append(max(new_price, 1.0))  # Ensure positive prices
    
    # Generate OHLCV
    data = []
    for i, (ts, close) in enumerate(zip(timestamps, prices)):
        high = close + abs(np.random.normal(0, 0.3))
        low = close - abs(np.random.normal(0, 0.3))
        open_price = prices[i-1] if i > 0 else close
        volume = int(np.random.uniform(1000, 10000))
        
        data.append({
            'time': ts,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('time', inplace=True)
    return df


def main():
    print("="*60)
    print("Natron Demo - Testing Feature Engineering Pipeline")
    print("="*60)
    
    # Generate sample data
    print("\n1. Generating sample OHLCV data...")
    df = generate_sample_data(n_candles=1000)
    print(f"   Generated {len(df)} candles")
    print(f"   Columns: {df.columns.tolist()}")
    print(f"   Shape: {df.shape}")
    
    # Save sample data
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/sample_data.csv')
    print(f"   Saved to data/sample_data.csv")
    
    # Extract features
    print("\n2. Extracting features...")
    feature_engine = FeatureEngine()
    features_df = feature_engine.extract_all_features(df)
    print(f"   Features shape: {features_df.shape}")
    print(f"   Feature columns: {len(features_df.columns)}")
    print(f"   Sample features: {list(features_df.columns[:10])}")
    
    # Generate labels
    print("\n3. Generating labels...")
    label_generator = LabelGenerator()
    labels_df = label_generator.generate_labels(df, features_df)
    print(f"   Labels shape: {labels_df.shape}")
    print(f"   Label columns: {labels_df.columns.tolist()}")
    
    # Show label distribution
    print("\n   Label distributions:")
    print(f"   Buy signals: {labels_df['buy'].sum()} ({labels_df['buy'].mean()*100:.1f}%)")
    print(f"   Sell signals: {labels_df['sell'].sum()} ({labels_df['sell'].mean()*100:.1f}%)")
    print(f"   Direction up: {labels_df['direction'].sum()} ({labels_df['direction'].mean()*100:.1f}%)")
    print("\n   Regime distribution:")
    regime_counts = labels_df['regime'].value_counts().sort_index()
    regime_names = ['BULL_STRONG', 'BULL_WEAK', 'RANGE', 'BEAR_WEAK', 'BEAR_STRONG', 'VOLATILE']
    for idx, count in regime_counts.items():
        print(f"   {regime_names[idx]}: {count} ({count/len(labels_df)*100:.1f}%)")
    
    # Create sequences
    print("\n4. Creating sequences...")
    sequence_creator = SequenceCreator(sequence_length=96)
    X, y, metadata = sequence_creator.create_sequences_from_raw(df, features_df, labels_df)
    print(f"   Sequences shape: {X.shape}")
    print(f"   Number of sequences: {len(X)}")
    print(f"   Sequence length: {X.shape[1]}")
    print(f"   Features per timestep: {X.shape[2]}")
    
    print("\n" + "="*60)
    print("Demo completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Train model: python train.py --data data/sample_data.csv --phase both")
    print("2. Test inference: python inference.py --model models/natron_v2.pt --data data/sample_data.csv")
    print("3. Start API: python api_server.py --model models/natron_v2.pt")


if __name__ == '__main__':
    main()
