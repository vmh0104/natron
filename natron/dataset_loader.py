"""
Dataset Loader for Natron Model Training
PyTorch Dataset and DataLoader implementations.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path


class NatronDataset(Dataset):
    """
    PyTorch Dataset for Natron model training.
    """
    
    def __init__(self,
                 X: np.ndarray,
                 y_regime: np.ndarray,
                 y_context: np.ndarray,
                 y_forecast: np.ndarray,
                 normalize: bool = True,
                 feature_mean: Optional[np.ndarray] = None,
                 feature_std: Optional[np.ndarray] = None):
        """
        Initialize dataset.
        
        Args:
            X: Input sequences (n_samples, seq_len, n_features)
            y_regime: Regime labels (n_samples,)
            y_context: Context strength scores (n_samples,)
            y_forecast: Forecast direction labels (n_samples,)
            normalize: Whether to normalize features
            feature_mean: Precomputed feature means (for normalization)
            feature_std: Precomputed feature stds (for normalization)
        """
        self.X = X.astype(np.float32)
        self.y_regime = y_regime.astype(np.int64)
        self.y_context = y_context.astype(np.float32)
        self.y_forecast = y_forecast.astype(np.int64)
        
        self.normalize = normalize
        
        if normalize:
            if feature_mean is None or feature_std is None:
                # Compute normalization statistics
                # Reshape to (n_samples * seq_len, n_features) for statistics
                X_flat = self.X.reshape(-1, self.X.shape[-1])
                self.feature_mean = np.nanmean(X_flat, axis=0)
                self.feature_std = np.nanstd(X_flat, axis=0)
                # Avoid division by zero
                self.feature_std = np.where(self.feature_std < 1e-8, 1.0, self.feature_std)
            else:
                self.feature_mean = feature_mean
                self.feature_std = feature_std
            
            # Normalize
            self.X = (self.X - self.feature_mean) / self.feature_std
            # Replace NaN and Inf with 0
            self.X = np.nan_to_num(self.X, nan=0.0, posinf=0.0, neginf=0.0)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (X, y_regime, y_context, y_forecast)
        """
        return (
            torch.tensor(self.X[idx]),
            torch.tensor(self.y_regime[idx]),
            torch.tensor(self.y_context[idx]),
            torch.tensor(self.y_forecast[idx])
        )
    
    def get_normalization_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get normalization statistics.
        
        Returns:
            Tuple of (feature_mean, feature_std)
        """
        if not self.normalize:
            raise ValueError("Dataset is not normalized")
        return self.feature_mean, self.feature_std


def create_dataloaders(X: np.ndarray,
                      y_regime: np.ndarray,
                      y_context: np.ndarray,
                      y_forecast: np.ndarray,
                      train_ratio: float = 0.7,
                      val_ratio: float = 0.15,
                      batch_size: int = 32,
                      shuffle: bool = True,
                      num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple[np.ndarray, np.ndarray]]:
    """
    Create train/val/test dataloaders.
    
    Args:
        X: Input sequences
        y_regime: Regime labels
        y_context: Context strength scores
        y_forecast: Forecast direction labels
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        batch_size: Batch size
        shuffle: Whether to shuffle training data
        num_workers: Number of data loading workers
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader, normalization_stats)
    """
    n_samples = len(X)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    
    # Split indices
    indices = np.arange(n_samples)
    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    # Create datasets (train set computes normalization stats)
    train_dataset = NatronDataset(
        X[train_indices],
        y_regime[train_indices],
        y_context[train_indices],
        y_forecast[train_indices],
        normalize=True
    )
    
    # Get normalization stats from training set
    feature_mean, feature_std = train_dataset.get_normalization_stats()
    
    # Apply same normalization to val and test sets
    val_dataset = NatronDataset(
        X[val_indices],
        y_regime[val_indices],
        y_context[val_indices],
        y_forecast[val_indices],
        normalize=True,
        feature_mean=feature_mean,
        feature_std=feature_std
    )
    
    test_dataset = NatronDataset(
        X[test_indices],
        y_regime[test_indices],
        y_context[test_indices],
        y_forecast[test_indices],
        normalize=True,
        feature_mean=feature_mean,
        feature_std=feature_std
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples")
    print(f"  Test:  {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader, (feature_mean, feature_std)
