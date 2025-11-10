"""
Natron Sequence Creator - Builds sequences of 96 consecutive candles
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional


class SequenceCreator:
    """Creates sequences of 96 consecutive candles for training"""
    
    def __init__(self, sequence_length: int = 96):
        self.sequence_length = sequence_length
    
    def create_sequences(
        self, 
        features_df: pd.DataFrame, 
        labels_df: pd.DataFrame
    ) -> Tuple[np.ndarray, dict]:
        """
        Create sequences from features and labels
        
        Returns:
            X: (N, 96, num_features) array
            y: dict with keys 'buy', 'sell', 'direction', 'regime'
        """
        # Align indices
        common_idx = features_df.index.intersection(labels_df.index)
        features_aligned = features_df.loc[common_idx]
        labels_aligned = labels_df.loc[common_idx]
        
        # Convert to numpy
        features_array = features_aligned.values.astype(np.float32)
        labels_array = labels_aligned.values
        
        # Create sequences
        num_samples = len(features_array) - self.sequence_length + 1
        
        if num_samples <= 0:
            raise ValueError(f"Not enough data. Need at least {self.sequence_length} samples.")
        
        X = np.zeros((num_samples, self.sequence_length, features_array.shape[1]), dtype=np.float32)
        
        y_buy = np.zeros(num_samples, dtype=np.int64)
        y_sell = np.zeros(num_samples, dtype=np.int64)
        y_direction = np.zeros(num_samples, dtype=np.int64)
        y_regime = np.zeros(num_samples, dtype=np.int64)
        
        for i in range(num_samples):
            X[i] = features_array[i:i + self.sequence_length]
            # Label is at the end of the sequence
            y_buy[i] = labels_array[i + self.sequence_length - 1, 0]
            y_sell[i] = labels_array[i + self.sequence_length - 1, 1]
            y_direction[i] = labels_array[i + self.sequence_length - 1, 2]
            y_regime[i] = labels_array[i + self.sequence_length - 1, 3]
        
        y = {
            'buy': y_buy,
            'sell': y_sell,
            'direction': y_direction,
            'regime': y_regime
        }
        
        return X, y
    
    def create_sequences_from_raw(
        self,
        df: pd.DataFrame,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame
    ) -> Tuple[np.ndarray, dict, pd.DataFrame]:
        """
        Create sequences and return metadata (timestamps, etc.)
        """
        X, y = self.create_sequences(features_df, labels_df)
        
        # Create metadata DataFrame
        common_idx = features_df.index.intersection(labels_df.index)
        metadata = pd.DataFrame({
            'timestamp': common_idx[self.sequence_length - 1:],
            'close': df.loc[common_idx, 'close'].values[self.sequence_length - 1:]
        })
        
        return X, y, metadata
