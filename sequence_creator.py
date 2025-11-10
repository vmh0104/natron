"""
Sequence Creator - Build sequences of 96 consecutive candles
"""
import numpy as np
import pandas as pd
from typing import Tuple


class SequenceCreator:
    """Create sequences for Transformer training"""
    
    def __init__(self, sequence_length: int = 96):
        self.sequence_length = sequence_length
    
    def create_sequences(self, 
                       features_df: pd.DataFrame,
                       labels_df: pd.DataFrame) -> Tuple[np.ndarray, dict]:
        """
        Create sequences of features and labels
        
        Args:
            features_df: DataFrame with feature columns
            labels_df: DataFrame with buy, sell, direction, regime columns
            
        Returns:
            X: (N, sequence_length, num_features) array
            y: dict with keys 'buy', 'sell', 'direction', 'regime', each (N,)
        """
        features_array = features_df.values.astype(np.float32)
        
        # Get labels
        buy_labels = labels_df['buy'].values.astype(np.int64)
        sell_labels = labels_df['sell'].values.astype(np.int64)
        direction_labels = labels_df['direction'].values.astype(np.int64)
        regime_labels = labels_df['regime'].values.astype(np.int64)
        
        # Create sequences
        num_samples = len(features_df) - self.sequence_length + 1
        
        if num_samples <= 0:
            raise ValueError(f"Not enough data. Need at least {self.sequence_length} rows.")
        
        X = np.zeros((num_samples, self.sequence_length, features_array.shape[1]), dtype=np.float32)
        y_buy = np.zeros(num_samples, dtype=np.int64)
        y_sell = np.zeros(num_samples, dtype=np.int64)
        y_direction = np.zeros(num_samples, dtype=np.int64)
        y_regime = np.zeros(num_samples, dtype=np.int64)
        
        for i in range(num_samples):
            # Input: sequence of features
            X[i] = features_array[i:i+self.sequence_length]
            
            # Labels: use the label at the end of the sequence
            y_buy[i] = buy_labels[i + self.sequence_length - 1]
            y_sell[i] = sell_labels[i + self.sequence_length - 1]
            y_direction[i] = direction_labels[i + self.sequence_length - 1]
            y_regime[i] = regime_labels[i + self.sequence_length - 1]
        
        y = {
            'buy': y_buy,
            'sell': y_sell,
            'direction': y_direction,
            'regime': y_regime
        }
        
        return X, y
