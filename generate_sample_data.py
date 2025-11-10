"""
Generate sample OHLCV data for testing Natron
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_data(n_candles=1000, timeframe_minutes=15, output_file='data_export.csv'):
    """
    Generate sample OHLCV data for testing.
    
    Args:
        n_candles: Number of candles to generate
        timeframe_minutes: Timeframe in minutes (15 for M15, 60 for H1)
        output_file: Output CSV file path
    """
    print(f"Generating {n_candles} candles ({timeframe_minutes}-minute timeframe)...")
    
    # Start time
    start_time = datetime(2023, 1, 1, 0, 0, 0)
    
    # Generate timestamps
    timestamps = [start_time + timedelta(minutes=timeframe_minutes * i) for i in range(n_candles)]
    
    # Generate price data (random walk with trend)
    np.random.seed(42)
    base_price = 1.1000
    
    # Create trend + noise
    trend = np.linspace(0, 0.01, n_candles)  # Upward trend
    noise = np.cumsum(np.random.randn(n_candles) * 0.0001)
    close_prices = base_price + trend + noise
    
    # Generate OHLC from close prices
    opens = np.roll(close_prices, 1)
    opens[0] = base_price
    
    # High/Low with some spread
    spreads = np.abs(np.random.randn(n_candles) * 0.0005)
    highs = close_prices + spreads
    lows = close_prices - spreads
    
    # Ensure high >= close >= low and high >= open >= low
    highs = np.maximum(highs, np.maximum(close_prices, opens))
    lows = np.minimum(lows, np.minimum(close_prices, opens))
    
    # Generate volume
    volumes = np.random.randint(1000, 10000, n_candles)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': [int(ts.timestamp()) for ts in timestamps],
        'open': opens,
        'high': highs,
        'low': lows,
        'close': close_prices,
        'volume': volumes
    })
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✓ Saved sample data to {output_file}")
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"  Price range: {df['low'].min():.5f} to {df['high'].max():.5f}")
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample OHLCV data')
    parser.add_argument('--n', type=int, default=1000, help='Number of candles')
    parser.add_argument('--timeframe', type=int, default=15, help='Timeframe in minutes')
    parser.add_argument('--output', type=str, default='data_export.csv', help='Output file')
    args = parser.parse_args()
    
    generate_sample_data(n_candles=args.n, timeframe_minutes=args.timeframe, output_file=args.output)
