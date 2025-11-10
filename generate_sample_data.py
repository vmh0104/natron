"""
Generate sample OHLCV data for testing
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_data(
    num_candles: int = 1000,
    start_price: float = 100.0,
    timeframe: str = '15min',
    output_path: str = 'data_export.csv'
):
    """
    Generate sample OHLCV data
    
    Args:
        num_candles: Number of candles to generate
        start_price: Starting price
        timeframe: Timeframe ('15min', '1H', etc.)
        output_path: Output CSV path
    """
    np.random.seed(42)
    
    # Generate timestamps
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    if timeframe == '15min':
        delta = timedelta(minutes=15)
    elif timeframe == '1H':
        delta = timedelta(hours=1)
    else:
        delta = timedelta(minutes=15)
    
    timestamps = [start_time + i * delta for i in range(num_candles)]
    
    # Generate price data (random walk with trend)
    returns = np.random.randn(num_candles) * 0.5 + 0.001  # Slight upward trend
    prices = start_price + np.cumsum(returns)
    
    # Generate OHLCV
    data = []
    for i in range(num_candles):
        base_price = prices[i]
        volatility = np.random.uniform(0.3, 1.5)
        
        # Generate OHLC
        open_price = base_price + np.random.randn() * volatility * 0.1
        close_price = base_price + np.random.randn() * volatility * 0.1
        
        high_price = max(open_price, close_price) + abs(np.random.randn() * volatility * 0.2)
        low_price = min(open_price, close_price) - abs(np.random.randn() * volatility * 0.2)
        
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'time': timestamps[i],
            'open': round(open_price, 5),
            'high': round(high_price, 5),
            'low': round(low_price, 5),
            'close': round(close_price, 5),
            'volume': volume
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Generated {num_candles} candles")
    print(f"Saved to {output_path}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nLast few rows:")
    print(df.tail())
    
    return df


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate sample OHLCV data')
    parser.add_argument('--num_candles', type=int, default=1000,
                       help='Number of candles to generate')
    parser.add_argument('--start_price', type=float, default=100.0,
                       help='Starting price')
    parser.add_argument('--timeframe', type=str, default='15min',
                       help='Timeframe (15min, 1H, etc.)')
    parser.add_argument('--output', type=str, default='data_export.csv',
                       help='Output CSV path')
    
    args = parser.parse_args()
    
    generate_sample_data(
        num_candles=args.num_candles,
        start_price=args.start_price,
        timeframe=args.timeframe,
        output_path=args.output
    )
