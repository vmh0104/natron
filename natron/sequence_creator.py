"""
Natron Sequence Creator - Builds sequences of 96 consecutive candles
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler


class SequenceCreator:
    """Creates sequences of 96 consecutive candles for model training."""
    
    def __init__(self, sequence_length: int = 96):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def create_sequences(self, 
                       df: pd.DataFrame,
                       feature_columns: Optional[list] = None,
                       scale_features: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Create sequences from DataFrame.
        
        Args:
            df: DataFrame with features and labels
            feature_columns: List of feature column names (if None, auto-detect)
            scale_features: Whether to standardize features
            
        Returns:
            X: Sequences array of shape (N, 96, num_features)
            y: Dictionary with keys ['buy', 'sell', 'direction', 'regime']
        """
        # Auto-detect feature columns if not provided
        if feature_columns is None:
            base_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 
                        'buy', 'sell', 'direction', 'regime']
            feature_columns = [col for col in df.columns if col not in base_cols]
        
        self.feature_columns = feature_columns
        
        # Extract features and labels
        X_features = df[feature_columns].values.astype(np.float32)
        y_buy = df['buy'].values.astype(np.int64)
        y_sell = df['sell'].values.astype(np.int64)
        y_direction = df['direction'].values.astype(np.int64)
        y_regime = df['regime'].values.astype(np.int64)
        
        # Scale features
        if scale_features:
            X_features = self.scaler.fit_transform(X_features)
        
        # Create sequences
        num_samples = len(df) - self.sequence_length + 1
        num_features = len(feature_columns)
        
        X = np.zeros((num_samples, self.sequence_length, num_features), dtype=np.float32)
        y = {
            'buy': np.zeros(num_samples, dtype=np.int64),
            'sell': np.zeros(num_samples, dtype=np.int64),
            'direction': np.zeros(num_samples, dtype=np.int64),
            'regime': np.zeros(num_samples, dtype=np.int64)
        }
        
        for i in range(num_samples):
            # Input sequence: features from i to i+96
            X[i] = X_features[i:i+self.sequence_length]
            
            # Labels: use the label at the end of the sequence (i+95)
            label_idx = i + self.sequence_length - 1
            y['buy'][i] = y_buy[label_idx]
            y['sell'][i] = y_sell[label_idx]
            y['direction'][i] = y_direction[label_idx]
            y['regime'][i] = y_regime[label_idx]
        
        return X, y
    
    def create_sequences_from_raw(self, 
                                  df: pd.DataFrame,
                                  feature_columns: Optional[list] = None) -> Tuple[np.ndarray, dict]:
        """
        Create sequences from raw OHLCV data (assumes features already computed).
        """
        return self.create_sequences(df, feature_columns, scale_features=True)
    
    def transform_new_data(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted scaler.
        
        Args:
            df: DataFrame with feature columns
            
        Returns:
            Scaled feature array
        """
        if self.feature_columns is None:
            raise ValueError("Scaler not fitted. Call create_sequences first.")
        
        X_features = df[self.feature_columns].values.astype(np.float32)
        X_scaled = self.scaler.transform(X_features)
        return X_scaled
