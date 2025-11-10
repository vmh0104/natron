#!/usr/bin/env python3
"""
Generate Sample OHLCV Data for Testing Natron
Creates realistic synthetic financial data for training and testing

Usage:
    python generate_sample_data.py --rows 5000 --output data/data_export.csv
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse


def generate_realistic_ohlcv(
    n_candles: int = 5000,
    initial_price: float = 1.0850,
    volatility: float = 0.001,
    trend: float = 0.0001,
    timeframe: str = '15min'
) -> pd.DataFrame:
    """
    Generate realistic OHLCV data with trends and volatility
    
    Args:
        n_candles: Number of candles to generate
        initial_price: Starting price
        volatility: Price volatility (standard deviation)
        trend: Trend strength (positive = uptrend)
        timeframe: Candle timeframe
        
    Returns:
        DataFrame with OHLCV data
    """
    print(f"🔧 Generating {n_candles} candles of {timeframe} data...")
    
    # Generate timestamps
    start_date = datetime(2023, 1, 1)
    if timeframe == '15min':
        delta = timedelta(minutes=15)
    elif timeframe == '1h':
        delta = timedelta(hours=1)
    elif timeframe == '1d':
        delta = timedelta(days=1)
    else:
        delta = timedelta(minutes=15)
    
    timestamps = [start_date + i * delta for i in range(n_candles)]
    
    # Generate price movements with trend and mean reversion
    returns = np.random.randn(n_candles) * volatility + trend
    
    # Add some autocorrelation (realistic price behavior)
    for i in range(1, len(returns)):
        returns[i] += 0.3 * returns[i-1]  # Momentum effect
    
    # Add some regime changes (market conditions)
    regime_length = n_candles // 10
    for i in range(0, n_candles, regime_length):
        regime_trend = np.random.choice([0.0002, -0.0002, 0.0])
        returns[i:i+regime_length] += regime_trend
    
    # Calculate close prices
    close_prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close prices
    data = []
    for i, close in enumerate(close_prices):
        # Generate realistic OHLC
        candle_volatility = abs(np.random.randn()) * volatility * close * 2
        
        # Open is previous close (with small gap)
        if i == 0:
            open_price = initial_price
        else:
            open_price = close_prices[i-1] * (1 + np.random.randn() * volatility * 0.5)
        
        # High and low based on volatility
        high = max(open_price, close) + abs(np.random.randn()) * candle_volatility
        low = min(open_price, close) - abs(np.random.randn()) * candle_volatility
        
        # Ensure high >= low
        if high < low:
            high, low = low, high
        
        # Generate volume (higher volume on larger price moves)
        price_change = abs(close - open_price) / open_price
        base_volume = 1000 + np.random.randint(0, 500)
        volume = int(base_volume * (1 + price_change * 100) * (1 + abs(np.random.randn()) * 0.5))
        
        data.append({
            'time': timestamps[i].strftime('%Y-%m-%d %H:%M'),
            'open': round(open_price, 5),
            'high': round(high, 5),
            'low': round(low, 5),
            'close': round(close, 5),
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    
    print(f"✅ Generated data:")
    print(f"   Rows: {len(df)}")
    print(f"   Date range: {df['time'].iloc[0]} to {df['time'].iloc[-1]}")
    print(f"   Price range: {df['close'].min():.5f} to {df['close'].max():.5f}")
    print(f"   Volume range: {df['volume'].min()} to {df['volume'].max()}")
    
    return df


def add_market_events(df: pd.DataFrame) -> pd.DataFrame:
    """Add realistic market events (spikes, gaps, etc.)"""
    df = df.copy()
    
    n = len(df)
    
    # Add some volatility spikes
    n_spikes = max(5, n // 500)
    spike_indices = np.random.choice(n, n_spikes, replace=False)
    
    for idx in spike_indices:
        if idx < n - 10:
            # Volatility spike
            spike_factor = 1 + abs(np.random.randn()) * 0.02
            df.loc[idx:idx+10, 'high'] *= spike_factor
            df.loc[idx:idx+10, 'low'] /= spike_factor
            df.loc[idx:idx+10, 'volume'] *= 2
    
    # Add some gaps
    n_gaps = max(3, n // 1000)
    gap_indices = np.random.choice(n, n_gaps, replace=False)
    
    for idx in gap_indices:
        if idx > 0:
            gap_size = np.random.choice([0.001, -0.001, 0.002, -0.002])
            df.loc[idx, 'open'] = df.loc[idx-1, 'close'] * (1 + gap_size)
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Generate sample OHLCV data')
    parser.add_argument('--rows', type=int, default=5000, help='Number of candles')
    parser.add_argument('--output', type=str, default='data/data_export.csv', help='Output file path')
    parser.add_argument('--initial-price', type=float, default=1.0850, help='Initial price')
    parser.add_argument('--volatility', type=float, default=0.001, help='Volatility')
    parser.add_argument('--trend', type=float, default=0.00005, help='Trend strength')
    parser.add_argument('--timeframe', type=str, default='15min', choices=['15min', '1h', '1d'], help='Timeframe')
    parser.add_argument('--with-events', action='store_true', help='Add market events')
    
    args = parser.parse_args()
    
    print("="*60)
    print("📊 Natron Sample Data Generator")
    print("="*60)
    
    # Generate data
    df = generate_realistic_ohlcv(
        n_candles=args.rows,
        initial_price=args.initial_price,
        volatility=args.volatility,
        trend=args.trend,
        timeframe=args.timeframe
    )
    
    # Add events if requested
    if args.with_events:
        print("\n🎭 Adding market events...")
        df = add_market_events(df)
        print("✅ Market events added")
    
    # Save to file
    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    df.to_csv(args.output, index=False)
    print(f"\n💾 Data saved to: {args.output}")
    
    # Display sample
    print("\n📋 Sample data (first 5 rows):")
    print(df.head().to_string())
    
    # Statistics
    print("\n📊 Statistics:")
    print(f"   Total candles: {len(df)}")
    print(f"   Start price: {df['close'].iloc[0]:.5f}")
    print(f"   End price: {df['close'].iloc[-1]:.5f}")
    print(f"   Price change: {((df['close'].iloc[-1] / df['close'].iloc[0]) - 1) * 100:.2f}%")
    print(f"   Avg volume: {df['volume'].mean():.0f}")
    
    print("\n" + "="*60)
    print("✅ Data generation complete!")
    print("\n💡 Next steps:")
    print(f"   1. Review the data: head {args.output}")
    print(f"   2. Train Natron: python train_natron.py")
    print("="*60)


if __name__ == "__main__":
    main()
