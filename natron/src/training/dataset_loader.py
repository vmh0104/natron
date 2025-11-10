"""
Dataset Loader for Natron Training
Creates sequences from processed data for transformer training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


class NatronDataset(Dataset):
    """
    Dataset for Natron transformer model.
    Creates sequences of candles with features and labels.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 96,
        feature_cols: Optional[list] = None,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False
    ):
        """
        Initialize dataset.
        
        Args:
            data: Processed DataFrame with features and labels
            sequence_length: Length of input sequences
            feature_cols: List of feature column names (if None, auto-detect)
            scaler: StandardScaler instance (optional)
            fit_scaler: Whether to fit scaler on this data
        """
        self.data = data.copy()
        self.sequence_length = sequence_length
        
        # Identify feature columns (exclude labels and metadata)
        exclude_cols = ['regime', 'regime_name', 'time', 'open', 'high', 'low', 'close']
        if 'volume' in self.data.columns:
            exclude_cols.append('volume')
        
        if feature_cols is None:
            self.feature_cols = [col for col in self.data.columns if col not in exclude_cols]
        else:
            self.feature_cols = feature_cols
        
        # Extract features and labels
        self.features = self.data[self.feature_cols].values.astype(np.float32)
        self.regime_labels = self.data['regime'].values.astype(np.int64)
        
        # Create forecast labels (next candle direction)
        self.forecast_labels = (self.data['close'].shift(-1) > self.data['close']).astype(int)
        self.forecast_labels = self.forecast_labels.fillna(0).values.astype(np.int64)
        
        # Compute context strength (absolute difference between bull and bear probabilities)
        # For now, we'll use a simple proxy: trend strength
        trend_strength = np.abs(self.data['EMA_20_50_diff'] / (self.data['close'] + 1e-8))
        self.context_labels = np.clip(trend_strength.values.astype(np.float32), 0, 1)
        
        # Scale features
        if scaler is None:
            self.scaler = StandardScaler()
            if fit_scaler:
                self.features = self.scaler.fit_transform(self.features)
        else:
            self.scaler = scaler
            if fit_scaler:
                self.features = self.scaler.fit_transform(self.features)
            else:
                self.features = self.scaler.transform(self.features)
        
        # Create valid indices (sequences that don't go out of bounds)
        self.valid_indices = list(range(len(self.data) - sequence_length))
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with features and labels
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length
        
        # Extract sequence
        sequence = self.features[start_idx:end_idx]  # [sequence_length, num_features]
        
        # Labels are for the last timestep
        regime_label = self.regime_labels[end_idx - 1]
        forecast_label = self.forecast_labels[end_idx - 1]
        context_label = self.context_labels[end_idx - 1]
        
        return {
            'features': torch.FloatTensor(sequence),
            'regime': torch.LongTensor([regime_label])[0],
            'forecast': torch.LongTensor([forecast_label])[0],
            'context': torch.FloatTensor([context_label])[0]
        }
    
    def get_feature_names(self) -> list:
        """Return list of feature names."""
        return self.feature_cols
    
    def save_scaler(self, path: str):
        """Save scaler to file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    @staticmethod
    def load_scaler(path: str) -> StandardScaler:
        """Load scaler from file."""
        with open(path, 'rb') as f:
            return pickle.load(f)


def create_data_loaders(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: Optional[pd.DataFrame] = None,
    sequence_length: int = 96,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], StandardScaler]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        train_data: Training DataFrame
        val_data: Validation DataFrame
        test_data: Test DataFrame (optional)
        sequence_length: Length of input sequences
        batch_size: Batch size
        num_workers: Number of data loader workers
        pin_memory: Whether to pin memory for GPU transfer
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, scaler)
    """
    # Create training dataset (fit scaler on training data)
    train_dataset = NatronDataset(
        train_data,
        sequence_length=sequence_length,
        fit_scaler=True
    )
    
    # Get scaler from training dataset
    scaler = train_dataset.scaler
    
    # Create validation dataset (use training scaler)
    val_dataset = NatronDataset(
        val_data,
        sequence_length=sequence_length,
        scaler=scaler,
        fit_scaler=False,
        feature_cols=train_dataset.get_feature_names()
    )
    
    # Create test dataset if provided
    test_dataset = None
    if test_data is not None:
        test_dataset = NatronDataset(
            test_data,
            sequence_length=sequence_length,
            scaler=scaler,
            fit_scaler=False,
            feature_cols=train_dataset.get_feature_names()
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return train_loader, val_loader, test_loader, scaler
