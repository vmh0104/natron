"""
SequenceCreator: Build sequences of 96 consecutive candles for model input
"""
import numpy as np
import pandas as pd
from typing import Tuple, Optional


class SequenceCreator:
    """Create sequences from time series data"""
    
    def __init__(self, sequence_length: int = 96):
        self.sequence_length = sequence_length
    
    def create_sequences(
        self, 
        df: pd.DataFrame, 
        feature_columns: list,
        label_columns: Optional[list] = None
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Create sequences from DataFrame
        
        Args:
            df: DataFrame with features and labels
            feature_columns: List of feature column names to use
            label_columns: Optional list of label column names
            
        Returns:
            X: Array of shape (N, sequence_length, num_features)
            y: Dict with label arrays, or None if label_columns not provided
        """
        # Extract feature matrix
        feature_data = df[feature_columns].values.astype(np.float32)
        
        # Normalize features (z-score normalization)
        feature_mean = np.nanmean(feature_data, axis=0, keepdims=True)
        feature_std = np.nanstd(feature_data, axis=0, keepdims=True) + 1e-8
        feature_data = (feature_data - feature_mean) / feature_std
        
        # Replace any remaining NaN/inf with 0
        feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Create sequences
        num_samples = len(df) - self.sequence_length + 1
        if num_samples <= 0:
            raise ValueError(f"DataFrame too short for sequence length {self.sequence_length}")
        
        X = np.zeros((num_samples, self.sequence_length, len(feature_columns)), dtype=np.float32)
        
        for i in range(num_samples):
            X[i] = feature_data[i:i + self.sequence_length]
        
        # Create labels if provided
        y = None
        if label_columns:
            y = {}
            for label_col in label_columns:
                if label_col in df.columns:
                    label_data = df[label_col].values
                    # Labels correspond to the last candle in each sequence
                    y[label_col] = label_data[self.sequence_length - 1:]
                else:
                    print(f"Warning: Label column '{label_col}' not found in DataFrame")
        
        return X, y
    
    def create_sequences_with_splits(
        self,
        df: pd.DataFrame,
        feature_columns: list,
        label_columns: Optional[list] = None,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ) -> Tuple[dict, dict]:
        """
        Create sequences and split into train/val/test sets
        
        Args:
            df: DataFrame with features and labels
            feature_columns: List of feature column names
            label_columns: Optional list of label column names
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            
        Returns:
            X_splits: Dict with 'train', 'val', 'test' keys containing feature arrays
            y_splits: Dict with 'train', 'val', 'test' keys containing label dicts
        """
        # Create all sequences
        X, y = self.create_sequences(df, feature_columns, label_columns)
        
        # Calculate split indices
        total_samples = len(X)
        train_end = int(total_samples * train_ratio)
        val_end = train_end + int(total_samples * val_ratio)
        
        # Split features
        X_splits = {
            'train': X[:train_end],
            'val': X[train_end:val_end],
            'test': X[val_end:]
        }
        
        # Split labels
        y_splits = {'train': {}, 'val': {}, 'test': {}}
        if y:
            for label_name, label_array in y.items():
                y_splits['train'][label_name] = label_array[:train_end]
                y_splits['val'][label_name] = label_array[train_end:val_end]
                y_splits['test'][label_name] = label_array[val_end:]
        
        return X_splits, y_splits
