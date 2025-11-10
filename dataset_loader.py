"""
Natron Dataset Loader
Creates sequences of 96 consecutive candles with features and labels.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import pickle
import os


class NatronDataset(Dataset):
    """
    PyTorch Dataset for Natron Transformer.
    Each sample: (96, 100) feature sequence → (buy, sell, direction, regime) labels
    """
    
    def __init__(self, X: np.ndarray, y_buy: np.ndarray, y_sell: np.ndarray, 
                 y_direction: np.ndarray, y_regime: np.ndarray):
        """
        Args:
            X: Feature sequences (N, 96, 100)
            y_buy: Buy signals (N,)
            y_sell: Sell signals (N,)
            y_direction: Direction labels (N,)
            y_regime: Regime labels (N,)
        """
        self.X = torch.FloatTensor(X)
        self.y_buy = torch.FloatTensor(y_buy)
        self.y_sell = torch.FloatTensor(y_sell)
        self.y_direction = torch.LongTensor(y_direction)
        self.y_regime = torch.LongTensor(y_regime)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'features': self.X[idx],
            'buy': self.y_buy[idx],
            'sell': self.y_sell[idx],
            'direction': self.y_direction[idx],
            'regime': self.y_regime[idx]
        }


class SequenceCreator:
    """
    Creates sequences from feature DataFrame and labels.
    """
    
    def __init__(self, sequence_length: int = 96, feature_dim: int = 100):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.scaler = StandardScaler()
        self.scaler_fitted = False
    
    def create_sequences(self, features_df: pd.DataFrame, labels_df: pd.DataFrame,
                        fit_scaler: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                          np.ndarray, np.ndarray]:
        """
        Create sequences from features and labels.
        
        Args:
            features_df: DataFrame with feature columns
            labels_df: DataFrame with label columns (buy, sell, direction, regime)
            fit_scaler: Whether to fit scaler on this data
        
        Returns:
            X: (N, 96, 100) feature sequences
            y_buy: (N,) buy labels
            y_sell: (N,) sell labels
            y_direction: (N,) direction labels
            y_regime: (N,) regime labels
        """
        # Drop NaN rows
        features_df = features_df.dropna()
        labels_df = labels_df.loc[features_df.index]
        
        # Ensure we have enough data
        if len(features_df) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} rows, got {len(features_df)}")
        
        # Extract feature values
        feature_values = features_df.values.astype(np.float32)
        
        # Scale features
        if fit_scaler or not self.scaler_fitted:
            # Fit on all data (flatten sequences)
            feature_flat = feature_values.reshape(-1, feature_values.shape[-1])
            self.scaler.fit(feature_flat)
            self.scaler_fitted = True
        
        # Scale
        feature_flat = feature_values.reshape(-1, feature_values.shape[-1])
        feature_scaled_flat = self.scaler.transform(feature_flat)
        feature_scaled = feature_scaled_flat.reshape(feature_values.shape)
        
        # Create sequences
        sequences = []
        y_buy_list = []
        y_sell_list = []
        y_direction_list = []
        y_regime_list = []
        
        for i in range(len(feature_scaled) - self.sequence_length + 1):
            seq = feature_scaled[i:i + self.sequence_length]
            sequences.append(seq)
            
            # Labels are at the end of the sequence
            label_idx = i + self.sequence_length - 1
            y_buy_list.append(labels_df.iloc[label_idx]['buy'])
            y_sell_list.append(labels_df.iloc[label_idx]['sell'])
            y_direction_list.append(labels_df.iloc[label_idx]['direction'])
            y_regime_list.append(labels_df.iloc[label_idx]['regime'])
        
        X = np.array(sequences)
        y_buy = np.array(y_buy_list)
        y_sell = np.array(y_sell_list)
        y_direction = np.array(y_direction_list)
        y_regime = np.array(y_regime_list)
        
        return X, y_buy, y_sell, y_direction, y_regime
    
    def save_scaler(self, path: str):
        """Save fitted scaler"""
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
    
    def load_scaler(self, path: str):
        """Load fitted scaler"""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        self.scaler_fitted = True
    
    def transform_features(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted scaler"""
        if not self.scaler_fitted:
            raise ValueError("Scaler not fitted. Call create_sequences with fit_scaler=True first.")
        
        original_shape = features.shape
        features_flat = features.reshape(-1, features.shape[-1])
        features_scaled = self.scaler.transform(features_flat)
        return features_scaled.reshape(original_shape)


def create_data_loaders(X: np.ndarray, y_buy: np.ndarray, y_sell: np.ndarray,
                       y_direction: np.ndarray, y_regime: np.ndarray,
                       train_ratio: float = 0.8, val_ratio: float = 0.1,
                       batch_size: int = 32, shuffle: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test data loaders.
    
    Args:
        X: Feature sequences
        y_buy, y_sell, y_direction, y_regime: Labels
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        batch_size: Batch size
        shuffle: Whether to shuffle training data
    
    Returns:
        train_loader, val_loader, test_loader
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
    
    # Create datasets
    train_dataset = NatronDataset(
        X[train_indices], y_buy[train_indices], y_sell[train_indices],
        y_direction[train_indices], y_regime[train_indices]
    )
    val_dataset = NatronDataset(
        X[val_indices], y_buy[val_indices], y_sell[val_indices],
        y_direction[val_indices], y_regime[val_indices]
    )
    test_dataset = NatronDataset(
        X[test_indices], y_buy[test_indices], y_sell[test_indices],
        y_direction[test_indices], y_regime[test_indices]
    )
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader
