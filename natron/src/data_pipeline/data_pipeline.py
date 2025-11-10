"""
Main Data Pipeline Module for Natron
Orchestrates data preprocessing, feature engineering, and regime labeling.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Optional, Tuple
import logging

from .feature_engineering import FeatureEngineer
from .regime_labeler import RegimeLabeler


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Main data pipeline for preprocessing OHLCV data.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize data pipeline.
        
        Args:
            config_path: Path to YAML config file (optional)
        """
        self.config = self._load_config(config_path) if config_path else {}
        self.feature_engineer = FeatureEngineer()
        self.regime_labeler = RegimeLabeler(
            trend_window=self.config.get('regime', {}).get('trend_window', 20),
            atr_percentile_window=self.config.get('regime', {}).get('atr_percentile_window', 50),
            breakout_threshold=self.config.get('regime', {}).get('breakout_threshold', 0.02),
            volatility_threshold=self.config.get('regime', {}).get('volatility_threshold', 0.75)
        )
    
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_data(self, input_file: str) -> pd.DataFrame:
        """
        Load raw OHLCV data from CSV.
        
        Args:
            input_file: Path to input CSV file
            
        Returns:
            DataFrame with columns: time, open, high, low, close, volume
        """
        logger.info(f"Loading data from {input_file}")
        
        df = pd.read_csv(input_file)
        
        # Ensure required columns exist
        required_cols = ['time', 'open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Convert time to datetime if it's not already
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df = df.set_index('time')
        
        # Ensure numeric columns
        numeric_cols = ['open', 'high', 'low', 'close']
        if 'volume' in df.columns:
            numeric_cols.append('volume')
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN in critical columns
        df = df.dropna(subset=['open', 'high', 'low', 'close'])
        
        logger.info(f"Loaded {len(df)} candles")
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data: feature engineering and regime labeling.
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            Processed DataFrame with features and regime labels
        """
        logger.info("Starting feature engineering...")
        
        # Engineer features
        df_processed = self.feature_engineer.engineer_features(df)
        
        logger.info(f"Engineered {len(self.feature_engineer.get_feature_names())} features")
        
        # Label regimes
        logger.info("Labeling regimes...")
        df_processed['regime'] = self.regime_labeler.label_regime(df_processed, df_processed['ATR'])
        df_processed['regime_name'] = df_processed['regime'].map(self.regime_labeler.get_regime_name)
        
        # Get regime distribution
        regime_dist = self.regime_labeler.get_regime_distribution(df_processed['regime'])
        logger.info("Regime distribution:")
        for regime_name, stats in regime_dist.items():
            logger.info(f"  {regime_name}: {stats['count']} ({stats['percentage']:.2f}%)")
        
        # Standardize features (optional, can be done during training)
        # For now, we'll keep raw values and standardize during model training
        
        return df_processed
    
    def save_processed_data(self, df: pd.DataFrame, output_file: str):
        """
        Save processed data to CSV.
        
        Args:
            df: Processed DataFrame
            output_file: Path to output CSV file
        """
        logger.info(f"Saving processed data to {output_file}")
        
        # Ensure directory exists
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Reset index if time is the index
        df_to_save = df.reset_index() if df.index.name == 'time' else df.copy()
        
        df_to_save.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df_to_save)} rows to {output_file}")
    
    def run(self, input_file: str, output_file: str) -> pd.DataFrame:
        """
        Run complete data pipeline.
        
        Args:
            input_file: Path to input CSV file
            output_file: Path to output CSV file
            
        Returns:
            Processed DataFrame
        """
        logger.info("=" * 60)
        logger.info("NATRON DATA PIPELINE - Starting")
        logger.info("=" * 60)
        
        # Load data
        df_raw = self.load_data(input_file)
        
        # Preprocess
        df_processed = self.preprocess(df_raw)
        
        # Save
        self.save_processed_data(df_processed, output_file)
        
        logger.info("=" * 60)
        logger.info("NATRON DATA PIPELINE - Complete")
        logger.info("=" * 60)
        
        return df_processed


def main():
    """Main entry point for data pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Natron Data Pipeline')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    parser.add_argument('--config', type=str, default=None, help='Config YAML file')
    
    args = parser.parse_args()
    
    pipeline = DataPipeline(config_path=args.config)
    pipeline.run(args.input, args.output)


if __name__ == '__main__':
    main()
