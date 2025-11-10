"""
Data Validation Script
Validates data_export.csv format and provides statistics
"""

import pandas as pd
import sys
import os


def validate_data(csv_path: str = "data_export.csv"):
    """
    Validate OHLCV data file.
    
    Required columns: time, open, high, low, close, volume
    """
    print("=" * 60)
    print("📊 Natron Data Validation")
    print("=" * 60)
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"❌ Error: File not found: {csv_path}")
        return False
    
    print(f"✅ File found: {csv_path}")
    
    try:
        # Load data
        df = pd.read_csv(csv_path)
        print(f"✅ File loaded successfully")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        
        # Check required columns
        required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ Missing required columns: {missing_columns}")
            print(f"   Available columns: {list(df.columns)}")
            return False
        
        print(f"✅ All required columns present")
        
        # Check data types
        print("\n📋 Column Information:")
        for col in required_columns:
            dtype = df[col].dtype
            null_count = df[col].isnull().sum()
            print(f"   {col:10s} | Type: {str(dtype):15s} | Nulls: {null_count:6d}")
        
        # Validate OHLCV relationships
        print("\n🔍 Validating OHLCV relationships...")
        errors = []
        
        # High >= Low
        invalid_hl = (df['high'] < df['low']).sum()
        if invalid_hl > 0:
            errors.append(f"High < Low: {invalid_hl} rows")
        
        # High >= Open, Close
        invalid_ho = (df['high'] < df['open']).sum()
        invalid_hc = (df['high'] < df['close']).sum()
        if invalid_ho > 0:
            errors.append(f"High < Open: {invalid_ho} rows")
        if invalid_hc > 0:
            errors.append(f"High < Close: {invalid_hc} rows")
        
        # Low <= Open, Close
        invalid_lo = (df['low'] > df['open']).sum()
        invalid_lc = (df['low'] > df['close']).sum()
        if invalid_lo > 0:
            errors.append(f"Low > Open: {invalid_lo} rows")
        if invalid_lc > 0:
            errors.append(f"Low > Close: {invalid_lc} rows")
        
        # Volume >= 0
        invalid_vol = (df['volume'] < 0).sum()
        if invalid_vol > 0:
            errors.append(f"Volume < 0: {invalid_vol} rows")
        
        if errors:
            print("❌ Validation errors found:")
            for error in errors:
                print(f"   - {error}")
            return False
        else:
            print("✅ All OHLCV relationships valid")
        
        # Check for duplicates
        if 'time' in df.columns:
            duplicates = df['time'].duplicated().sum()
            if duplicates > 0:
                print(f"⚠️  Warning: {duplicates} duplicate timestamps found")
            else:
                print("✅ No duplicate timestamps")
        
        # Statistics
        print("\n📈 Data Statistics:")
        print(f"   Date Range: {df['time'].min()} to {df['time'].max()}")
        print(f"   Price Range: ${df['low'].min():.5f} - ${df['high'].max():.5f}")
        print(f"   Average Volume: {df['volume'].mean():,.0f}")
        print(f"   Total Volume: {df['volume'].sum():,.0f}")
        
        # Check minimum sequence length
        sequence_length = 96  # From config
        if len(df) < sequence_length:
            print(f"\n⚠️  Warning: Only {len(df)} rows, need at least {sequence_length} for training")
            print("   Model requires sequences of 96 consecutive candles")
        else:
            num_sequences = len(df) - sequence_length + 1
            print(f"\n✅ Sufficient data for {num_sequences:,} training sequences")
        
        print("\n" + "=" * 60)
        print("✅ Data validation complete!")
        print("=" * 60)
        return True
    
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return False


if __name__ == '__main__':
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data_export.csv"
    success = validate_data(csv_path)
    sys.exit(0 if success else 1)
