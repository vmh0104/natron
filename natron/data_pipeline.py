"""
Natron Data Pipeline
Preprocesses OHLCV market data and prepares it for ML training.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging

from feature_engineering import FeatureEngineer
from regime_labeler import RegimeLabeler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Main data pipeline orchestrator for Natron.
    Handles data loading, feature engineering, and regime labeling.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the data pipeline.
        
        Args:
            config_path: Optional path to YAML config file
        """
        self.feature_engineer = FeatureEngineer()
        self.regime_labeler = RegimeLabeler()
        logger.info("DataPipeline initialized")
    
    def load_raw_data(self, csv_path: str) -> pd.DataFrame:
        """
        Load raw OHLCV data from CSV.
        
        Expected columns: time, open, high, low, close, volume
        
        Args:
            csv_path: Path to input CSV file
            
        Returns:
            DataFrame with raw market data
        """
        logger.info(f"Loading raw data from {csv_path}")
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        # Convert time to datetime if needed
        if df['time'].dtype == 'object':
            df['time'] = pd.to_datetime(df['time'])
        
        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} candles")
        return df
    
    def process(self, csv_path: str, output_path: str = "processed_data.csv") -> pd.DataFrame:
        """
        Complete pipeline: load, engineer features, label regimes.
        
        Args:
            csv_path: Path to input CSV
            output_path: Path to save processed data
            
        Returns:
            Processed DataFrame ready for ML
        """
        # Load raw data
        df = self.load_raw_data(csv_path)
        
        # Engineer features (50-70 features)
        logger.info("Engineering features...")
        df = self.feature_engineer.engineer_all_features(df)
        
        # Label regimes
        logger.info("Labeling regimes...")
        df = self.regime_labeler.label_regimes(df)
        
        # Remove rows with NaN (from indicator calculations)
        initial_len = len(df)
        df = df.dropna().reset_index(drop=True)
        logger.info(f"Removed {initial_len - len(df)} rows with NaN")
        
        # Save processed data
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")
        
        return df


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_pipeline.py <input_csv> [output_csv]")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    output_csv = sys.argv[2] if len(sys.argv) > 2 else "processed_data.csv"
    
    pipeline = DataPipeline()
    processed_df = pipeline.process(input_csv, output_csv)
    print(f"\nProcessed {len(processed_df)} rows with {len(processed_df.columns)} features")
    print(f"Regime distribution:\n{processed_df['regime'].value_counts()}")
