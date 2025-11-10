"""
Create sample OHLCV data for testing
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_data(num_rows: int = 500, output_path: str = "data_export.csv"):
    """
    Generate sample OHLCV data for testing
    
    Args:
        num_rows: Number of candles to generate
        output_path: Output CSV file path
    """
    print(f"Generating {num_rows} sample candles...")
    
    # Generate timestamps (M15 timeframe)
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(minutes=15*i) for i in range(num_rows)]
    
    # Generate price data (random walk)
    np.random.seed(42)
    base_price = 100.0
    prices = [base_price]
    
    for i in range(1, num_rows):
        # Random walk with slight trend
        change = np.random.normal(0.01, 0.5)
        new_price = prices[-1] + change
        prices.append(max(new_price, 1.0))  # Ensure positive prices
    
    # Generate OHLCV
    data = []
    for i in range(num_rows):
        close = prices[i]
        volatility = np.random.uniform(0.1, 0.5)
        
        # Generate OHLC around close
        high = close + np.random.uniform(0, volatility)
        low = close - np.random.uniform(0, volatility)
        
        if i == 0:
            open_price = close
        else:
            open_price = prices[i-1] + np.random.uniform(-volatility*0.5, volatility*0.5)
        
        # Ensure high >= max(open, close) and low <= min(open, close)
        high = max(high, open_price, close)
        low = min(low, open_price, close)
        
        # Generate volume
        volume = int(np.random.uniform(1000, 10000))
        
        data.append({
            'time': timestamps[i],
            'open': round(open_price, 5),
            'high': round(high, 5),
            'low': round(low, 5),
            'close': round(close, 5),
            'volume': volume
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Sample data saved to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nLast few rows:")
    print(df.tail())
    
    return df

if __name__ == "__main__":
    create_sample_data(num_rows=500)
