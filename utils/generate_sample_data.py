"""
Sample Data Generator for Natron V2
Generates synthetic OHLCV data for testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse


def generate_synthetic_ohlcv(num_candles: int = 10000, 
                             start_price: float = 1.0850,
                             timeframe_minutes: int = 15,
                             volatility: float = 0.002,
                             trend: float = 0.0001,
                             output_path: str = 'data/data_export.csv'):
    """
    Generate synthetic OHLCV data with realistic market behavior.
    
    Args:
        num_candles: Number of candles to generate
        start_price: Starting price
        timeframe_minutes: Timeframe in minutes (15 for M15, 60 for H1)
        volatility: Price volatility (standard deviation)
        trend: Trend component (positive = uptrend, negative = downtrend)
        output_path: Output file path
    """
    
    print(f"Generating {num_candles} synthetic OHLCV candles...")
    
    # Initialize arrays
    timestamps = []
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    # Starting values
    current_time = datetime(2024, 1, 1, 0, 0)
    current_price = start_price
    
    # Generate regime changes (makes data more realistic)
    regime_changes = np.sort(np.random.choice(num_candles, size=10, replace=False))
    current_regime = 0
    
    for i in range(num_candles):
        # Change regime periodically
        if i in regime_changes:
            current_regime = (current_regime + 1) % 3
            if current_regime == 0:  # Trending up
                trend = abs(trend)
                volatility = 0.002
            elif current_regime == 1:  # Ranging
                trend = 0.0
                volatility = 0.001
            else:  # Trending down
                trend = -abs(trend)
                volatility = 0.003
        
        # Generate price movement
        price_change = np.random.normal(trend, volatility)
        new_price = current_price * (1 + price_change)
        
        # Generate OHLC
        open_price = current_price
        close_price = new_price
        
        # High and low with some randomness
        if close_price > open_price:
            high = close_price * (1 + abs(np.random.normal(0, volatility/2)))
            low = open_price * (1 - abs(np.random.normal(0, volatility/2)))
        else:
            high = open_price * (1 + abs(np.random.normal(0, volatility/2)))
            low = close_price * (1 - abs(np.random.normal(0, volatility/2)))
        
        # Ensure high >= open, close and low <= open, close
        high = max(high, open_price, close_price)
        low = min(low, open_price, close_price)
        
        # Generate volume (random with some correlation to price movement)
        base_volume = 10000
        volume_multiplier = 1 + abs(price_change) * 100  # More volume on big moves
        volume = int(base_volume * volume_multiplier * np.random.uniform(0.8, 1.2))
        
        # Store data
        timestamps.append(int(current_time.timestamp()))
        opens.append(round(open_price, 5))
        highs.append(round(high, 5))
        lows.append(round(low, 5))
        closes.append(round(close_price, 5))
        volumes.append(volume)
        
        # Update for next iteration
        current_price = new_price
        current_time += timedelta(minutes=timeframe_minutes)
    
    # Create DataFrame
    df = pd.DataFrame({
        'time': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    print(f"✓ Generated {num_candles} candles")
    print(f"✓ Saved to {output_path}")
    print(f"\nData Summary:")
    print(f"  Start Price: {opens[0]:.5f}")
    print(f"  End Price: {closes[-1]:.5f}")
    print(f"  Total Change: {((closes[-1] / opens[0]) - 1) * 100:.2f}%")
    print(f"  Price Range: {min(lows):.5f} - {max(highs):.5f}")
    print(f"  Avg Volume: {int(np.mean(volumes))}")
    print(f"\nYou can now train Natron with:")
    print(f"  python train_natron.py --data {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic OHLCV data')
    parser.add_argument('--candles', type=int, default=10000,
                       help='Number of candles to generate')
    parser.add_argument('--start-price', type=float, default=1.0850,
                       help='Starting price')
    parser.add_argument('--timeframe', type=int, default=15,
                       help='Timeframe in minutes (15 for M15, 60 for H1)')
    parser.add_argument('--volatility', type=float, default=0.002,
                       help='Price volatility')
    parser.add_argument('--trend', type=float, default=0.0001,
                       help='Trend component')
    parser.add_argument('--output', type=str, default='data/data_export.csv',
                       help='Output file path')
    
    args = parser.parse_args()
    
    generate_synthetic_ohlcv(
        num_candles=args.candles,
        start_price=args.start_price,
        timeframe_minutes=args.timeframe,
        volatility=args.volatility,
        trend=args.trend,
        output_path=args.output
    )


if __name__ == '__main__':
    main()
