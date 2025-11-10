"""
Sample Data Generator for Natron
Generates synthetic OHLCV data for testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse


def generate_sample_data(
    num_candles: int = 10000,
    start_date: str = "2020-01-01",
    timeframe_minutes: int = 15,
    output_file: str = "data/raw_data.csv"
):
    """
    Generate synthetic OHLCV data.
    
    Args:
        num_candles: Number of candles to generate
        start_date: Start date string
        timeframe_minutes: Candle timeframe in minutes
        output_file: Output CSV file path
    """
    print(f"Generating {num_candles} candles...")
    
    # Initialize
    start = datetime.strptime(start_date, "%Y-%m-%d")
    dates = [start + timedelta(minutes=timeframe_minutes * i) for i in range(num_candles)]
    
    # Generate price data with random walk + trend
    np.random.seed(42)
    base_price = 1.1000  # EUR/USD example
    
    # Random walk with drift
    returns = np.random.normal(0.0001, 0.001, num_candles)  # Small drift, volatility
    prices = base_price + np.cumsum(returns)
    
    # Generate OHLC
    opens = prices.copy()
    closes = prices.copy()
    
    # Add some intraday movement
    highs = closes + np.abs(np.random.normal(0, 0.0005, num_candles))
    lows = closes - np.abs(np.random.normal(0, 0.0005, num_candles))
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    for i in range(num_candles):
        if i > 0:
            opens[i] = closes[i-1] + np.random.normal(0, 0.0002)
        highs[i] = max(opens[i], closes[i]) + abs(np.random.normal(0, 0.0003))
        lows[i] = min(opens[i], closes[i]) - abs(np.random.normal(0, 0.0003))
    
    # Generate volume
    volumes = np.random.lognormal(10, 0.5, num_candles).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    # Round to 5 decimal places
    for col in ['open', 'high', 'low', 'close']:
        df[col] = df[col].round(5)
    
    # Save
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    
    print(f"Generated {len(df)} candles")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
    print(f"Price range: {df['low'].min():.5f} to {df['high'].max():.5f}")
    print(f"Saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Generate sample OHLCV data')
    parser.add_argument('--num-candles', type=int, default=10000, help='Number of candles')
    parser.add_argument('--start-date', type=str, default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', type=int, default=15, help='Timeframe in minutes')
    parser.add_argument('--output', type=str, default='data/raw_data.csv', help='Output file')
    
    args = parser.parse_args()
    
    generate_sample_data(
        num_candles=args.num_candles,
        start_date=args.start_date,
        timeframe_minutes=args.timeframe,
        output_file=args.output
    )


if __name__ == '__main__':
    main()
