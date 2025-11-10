"""
Main Data Pipeline Module for Natron Trading System

This module orchestrates the complete data preprocessing pipeline:
1. Load raw OHLCV data
2. Engineer features
3. Label regimes
4. Standardize and prepare for ML
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

from .feature_engineering import FeatureEngineer
from .regime_labeler import RegimeLabeler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Main data pipeline class that orchestrates preprocessing.
    """
    
    def __init__(self, 
                 config: Optional[dict] = None):
        """
        Initialize data pipeline.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.feature_engineer = FeatureEngineer()
        self.regime_labeler = RegimeLabeler(
            trend_window=self.config.get('trend_window', 20),
            atr_percentile_window=self.config.get('atr_percentile_window', 100),
            volatility_threshold=self.config.get('volatility_threshold', 0.75),
            trend_strength_threshold=self.config.get('trend_strength_threshold', 0.02)
        )
        
        self.feature_scalers = {}
        self.feature_stats = {}
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load raw OHLCV data from CSV.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame with OHLCV data
        """
        logger.info(f"Loading data from {file_path}")
        
        df = pd.read_csv(file_path)
        
        # Standardize column names (case-insensitive)
        column_mapping = {}
        for col in df.columns:
            col_lower = col.lower()
            if 'time' in col_lower or 'date' in col_lower or 'timestamp' in col_lower:
                column_mapping[col] = 'time'
            elif 'open' in col_lower:
                column_mapping[col] = 'open'
            elif 'high' in col_lower:
                column_mapping[col] = 'high'
            elif 'low' in col_lower:
                column_mapping[col] = 'low'
            elif 'close' in col_lower:
                column_mapping[col] = 'close'
            elif 'volume' in col_lower:
                column_mapping[col] = 'volume'
        
        df = df.rename(columns=column_mapping)
        
        # Convert time column to datetime if present
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add volume if missing (fill with 0)
        if 'volume' not in df.columns:
            logger.warning("Volume column not found, filling with zeros")
            df['volume'] = 0
        
        # Sort by index
        df = df.sort_index()
        
        logger.info(f"Loaded {len(df)} rows of data")
        return df
    
    def standardize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Standardize features using z-score normalization.
        
        Args:
            df: DataFrame with features
            fit: Whether to fit scalers (True for training, False for inference)
            
        Returns:
            DataFrame with standardized features
        """
        df = df.copy()
        
        # Exclude non-feature columns
        exclude_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 
                       'regime', 'regime_name', 'target']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if fit:
            # Compute statistics for standardization
            for col in feature_cols:
                mean = df[col].mean()
                std = df[col].std()
                self.feature_stats[col] = {'mean': mean, 'std': std}
                
                # Standardize
                df[col] = (df[col] - mean) / (std + 1e-8)
        else:
            # Use pre-computed statistics
            for col in feature_cols:
                if col in self.feature_stats:
                    stats = self.feature_stats[col]
                    df[col] = (df[col] - stats['mean']) / (stats['std'] + 1e-8)
        
        return df
    
    def create_forecast_target(self, df: pd.DataFrame, horizon: int = 5) -> pd.Series:
        """
        Create forecast target (direction of next N candles).
        
        Args:
            df: DataFrame with price data
            horizon: Number of candles ahead to forecast
            
        Returns:
            Series with target labels (0=down, 1=up)
        """
        # Compute future return
        future_price = df['close'].shift(-horizon)
        current_price = df['close']
        
        # Target: 1 if price goes up, 0 if down
        target = (future_price > current_price).astype(int)
        
        return target
    
    def process(self, 
                input_file: str,
                output_file: Optional[str] = None,
                create_target: bool = True,
                forecast_horizon: int = 5) -> pd.DataFrame:
        """
        Main processing pipeline.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to save processed data (optional)
            create_target: Whether to create forecast target
            forecast_horizon: Number of candles for forecast target
            
        Returns:
            Processed DataFrame ready for ML
        """
        logger.info("Starting data pipeline processing")
        
        # Step 1: Load raw data
        df = self.load_data(input_file)
        
        # Step 2: Engineer features
        logger.info("Engineering features...")
        feature_df = self.feature_engineer.engineer_features(df)
        
        # Combine with original OHLCV
        processed_df = pd.concat([df, feature_df], axis=1)
        
        # Step 3: Label regimes
        logger.info("Labeling regimes...")
        processed_df = self.regime_labeler.add_regime_labels(processed_df)
        
        # Step 4: Create forecast target
        if create_target:
            logger.info(f"Creating forecast target (horizon={forecast_horizon})...")
            processed_df['target'] = self.create_forecast_target(processed_df, forecast_horizon)
        
        # Step 5: Standardize features
        logger.info("Standardizing features...")
        processed_df = self.standardize_features(processed_df, fit=True)
        
        # Step 6: Remove rows with NaN (from rolling windows)
        initial_len = len(processed_df)
        processed_df = processed_df.dropna()
        final_len = len(processed_df)
        logger.info(f"Removed {initial_len - final_len} rows with NaN values")
        
        # Step 7: Save if output file specified
        if output_file:
            logger.info(f"Saving processed data to {output_file}")
            processed_df.to_csv(output_file, index=True)
            logger.info(f"Saved {len(processed_df)} rows to {output_file}")
        
        logger.info("Data pipeline processing complete")
        return processed_df
    
    def get_feature_names(self) -> list:
        """
        Get list of feature names.
        
        Returns:
            List of feature column names
        """
        return self.feature_engineer.feature_names


if __name__ == "__main__":
    # Example usage
    pipeline = DataPipeline()
    
    # Process data
    processed_df = pipeline.process(
        input_file="data/raw/market_data.csv",
        output_file="data/processed/processed_data.csv",
        create_target=True,
        forecast_horizon=5
    )
    
    print(f"Processed {len(processed_df)} rows")
    print(f"Features: {len(pipeline.get_feature_names())}")
    print(f"Regime distribution:\n{processed_df['regime_name'].value_counts()}")
