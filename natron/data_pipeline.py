"""
Data Pipeline Module for Natron AI Trading System
Main preprocessing pipeline that orchestrates feature engineering and regime labeling.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import FeatureEngineer
from regime_labeler import RegimeLabeler


class DataPipeline:
    """
    Main data preprocessing pipeline for Natron.
    Processes raw OHLCV data into ML-ready format with features and regime labels.
    """
    
    def __init__(self, 
                 output_dir: str = "data/processed",
                 trend_period: int = 50,
                 atr_percentile_window: int = 100):
        """
        Initialize Data Pipeline.
        
        Args:
            output_dir: Directory to save processed data
            trend_period: Period for trend calculation
            atr_percentile_window: Window for ATR percentile
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.feature_engineer = FeatureEngineer()
        self.regime_labeler = RegimeLabeler(
            trend_period=trend_period,
            atr_percentile_window=atr_percentile_window
        )
    
    def load_raw_data(self, file_path: str) -> pd.DataFrame:
        """
        Load raw OHLCV data from CSV.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with OHLCV columns
        """
        df = pd.read_csv(file_path)
        
        # Standardize column names (case-insensitive)
        column_mapping = {
            'time': 'time',
            'timestamp': 'time',
            'date': 'time',
            'datetime': 'time',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        
        df.columns = df.columns.str.lower()
        df = df.rename(columns={col: column_mapping.get(col, col) 
                               for col in df.columns})
        
        # Ensure time column is datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
        
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort by index
        df = df.sort_index()
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        print(f"Loaded {len(df)} candles from {file_path}")
        return df
    
    def standardize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize and clean data.
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        # Remove rows with invalid data
        df = df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
        
        # Validate OHLC relationships
        invalid_mask = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        if invalid_mask.sum() > 0:
            print(f"Warning: Removing {invalid_mask.sum()} rows with invalid OHLC data")
            df = df[~invalid_mask]
        
        # Ensure volume is non-negative
        df['volume'] = df['volume'].clip(lower=0)
        
        return df
    
    def process(self, 
                input_file: str,
                output_file: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process raw data through entire pipeline.
        
        Args:
            input_file: Path to input CSV file
            output_file: Optional output file path (default: processed_data.csv)
            
        Returns:
            Tuple of (features_df, labels_series)
        """
        print("\n" + "="*60)
        print("NATRON DATA PIPELINE - Processing Started")
        print("="*60)
        
        # Load raw data
        print("\n[1/4] Loading raw data...")
        df = self.load_raw_data(input_file)
        
        # Standardize data
        print("[2/4] Standardizing data...")
        df = self.standardize_data(df)
        print(f"   Cleaned dataset: {len(df)} candles")
        
        # Compute features
        print("[3/4] Computing features (50-70 features)...")
        feature_df = self.feature_engineer.compute_all_features(df)
        print(f"   Computed {len(feature_df.columns)} features")
        print(f"   Feature names: {', '.join(feature_df.columns[:10])}...")
        
        # Label regimes
        print("[4/4] Labeling regimes...")
        labels = self.regime_labeler.label_data(df, feature_df)
        
        # Print regime distribution
        self.regime_labeler.print_regime_distribution(labels)
        
        # Combine features and labels
        processed_df = feature_df.copy()
        processed_df['regime'] = labels
        
        # Add original OHLCV for reference
        processed_df['open'] = df['open']
        processed_df['high'] = df['high']
        processed_df['low'] = df['low']
        processed_df['close'] = df['close']
        processed_df['volume'] = df['volume']
        
        # Save processed data
        if output_file is None:
            output_file = self.output_dir / "processed_data.csv"
        else:
            output_file = Path(output_file)
        
        print(f"\n[SAVE] Saving processed data to {output_file}...")
        processed_df.to_csv(output_file)
        print(f"   Saved {len(processed_df)} rows × {len(processed_df.columns)} columns")
        
        print("\n" + "="*60)
        print("DATA PIPELINE - Processing Complete!")
        print("="*60 + "\n")
        
        return feature_df, labels
    
    def create_sequences(self, 
                        feature_df: pd.DataFrame,
                        labels: pd.Series,
                        sequence_length: int = 96,
                        forecast_horizon: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for transformer model training.
        
        Args:
            feature_df: Feature DataFrame
            labels: Regime labels series
            sequence_length: Number of candles per sequence (default 96)
            forecast_horizon: Number of candles ahead to forecast (default 5)
            
        Returns:
            Tuple of (X, y_regime, y_context, y_forecast)
            - X: (n_samples, sequence_length, n_features)
            - y_regime: (n_samples,) regime labels
            - y_context: (n_samples,) context strength scores
            - y_forecast: (n_samples,) forecast direction (0=down, 1=up)
        """
        n_samples = len(feature_df) - sequence_length - forecast_horizon + 1
        n_features = len(feature_df.columns)
        
        X = np.zeros((n_samples, sequence_length, n_features))
        y_regime = np.zeros(n_samples, dtype=int)
        y_context = np.zeros(n_samples)
        y_forecast = np.zeros(n_samples, dtype=int)
        
        # Extract feature columns (exclude OHLCV and regime)
        feature_cols = [col for col in feature_df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', 'regime']]
        
        for i in range(n_samples):
            # Input sequence
            X[i] = feature_df[feature_cols].iloc[i:i+sequence_length].values
            
            # Regime label (use label at end of sequence)
            y_regime[i] = labels.iloc[i + sequence_length - 1]
            
            # Context strength: |bull_prob - bear_prob|
            # Approximate from regime: BULL_STRONG=1.0, BULL_WEAK=0.5, BEAR_STRONG=-1.0, BEAR_WEAK=-0.5, RANGE=0, VOLATILE=0
            regime_id = y_regime[i]
            if regime_id == 0:  # BULL_STRONG
                bull_prob, bear_prob = 1.0, 0.0
            elif regime_id == 1:  # BULL_WEAK
                bull_prob, bear_prob = 0.6, 0.2
            elif regime_id == 2:  # BEAR_STRONG
                bull_prob, bear_prob = 0.0, 1.0
            elif regime_id == 3:  # BEAR_WEAK
                bull_prob, bear_prob = 0.2, 0.6
            else:  # RANGE or VOLATILE
                bull_prob, bear_prob = 0.4, 0.4
            
            y_context[i] = abs(bull_prob - bear_prob)
            
            # Forecast direction: 1 if price goes up in next N candles, else 0
            current_price = feature_df['close'].iloc[i + sequence_length - 1]
            future_price = feature_df['close'].iloc[i + sequence_length - 1 + forecast_horizon]
            y_forecast[i] = 1 if future_price > current_price else 0
        
        print(f"\nCreated {n_samples} sequences:")
        print(f"  X shape: {X.shape}")
        print(f"  y_regime shape: {y_regime.shape}")
        print(f"  y_context shape: {y_context.shape}")
        print(f"  y_forecast shape: {y_forecast.shape}")
        
        return X, y_regime, y_context, y_forecast


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_pipeline.py <input_csv_file> [output_csv_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    pipeline = DataPipeline()
    feature_df, labels = pipeline.process(input_file, output_file)
