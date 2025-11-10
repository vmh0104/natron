"""
DatasetLoader - Creates sequences of 96 consecutive candles for training
Part of Natron Transformer Multi-Task Trading System
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import pickle
import os


class TradingSequenceDataset(Dataset):
    """
    PyTorch Dataset for trading sequences.
    Each sample: 96 consecutive candles with 100 features.
    Labels: buy, sell, direction, regime
    """
    
    def __init__(self, 
                 features: np.ndarray,
                 labels: np.ndarray,
                 sequence_length: int = 96):
        """
        Args:
            features: (N, feature_dim) array of features
            labels: (N, 4) array with [buy, sell, direction, regime]
            sequence_length: Number of consecutive candles (default: 96)
        """
        self.sequence_length = sequence_length
        self.features = features
        self.labels = labels
        
        # Create valid indices (need sequence_length candles before each label)
        self.valid_indices = []
        for i in range(sequence_length - 1, len(features)):
            self.valid_indices.append(i)
        
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """
        Returns:
            X: (sequence_length, feature_dim) tensor
            y: dict with 'buy', 'sell', 'direction', 'regime'
        """
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.sequence_length + 1
        
        # Extract sequence
        X = self.features[start_idx:end_idx + 1]  # (sequence_length, feature_dim)
        
        # Extract label at the end of sequence
        y = self.labels[end_idx]  # (4,)
        
        # Convert to tensors
        X = torch.FloatTensor(X)
        
        return {
            'sequence': X,
            'buy': torch.FloatTensor([y[0]]),
            'sell': torch.FloatTensor([y[1]]),
            'direction': torch.LongTensor([y[2]]),
            'regime': torch.LongTensor([y[3]])
        }


class SequenceCreator:
    """
    Creates sequences from features and labels, handles train/val/test splits.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.sequence_length = config['model']['sequence_length']
        self.normalize = config['data'].get('normalize', True)
        self.scaler = StandardScaler() if self.normalize else None
        
    def create_datasets(self, 
                       features_df: pd.DataFrame,
                       labels_df: pd.DataFrame) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create train/val/test datasets.
        
        Args:
            features_df: DataFrame with features
            labels_df: DataFrame with labels ['buy', 'sell', 'direction', 'regime']
            
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        # Align indices
        common_idx = features_df.index.intersection(labels_df.index)
        features_df = features_df.loc[common_idx]
        labels_df = labels_df.loc[common_idx]
        
        # Convert to numpy
        features = features_df.values.astype(np.float32)
        labels = labels_df[['buy', 'sell', 'direction', 'regime']].values.astype(np.float32)
        
        # Split data
        train_split = self.config['data']['train_split']
        val_split = self.config['data']['val_split']
        
        n_total = len(features)
        n_train = int(n_total * train_split)
        n_val = int(n_total * val_split)
        
        # Split indices
        train_features = features[:n_train]
        train_labels = labels[:n_train]
        
        val_features = features[n_train:n_train + n_val]
        val_labels = labels[n_train:n_train + n_val]
        
        test_features = features[n_train + n_val:]
        test_labels = labels[n_train + n_val:]
        
        # Normalize features
        if self.normalize:
            train_features = self.scaler.fit_transform(train_features)
            val_features = self.scaler.transform(val_features)
            test_features = self.scaler.transform(test_features)
        
        # Create datasets
        train_dataset = TradingSequenceDataset(
            train_features, train_labels, self.sequence_length
        )
        val_dataset = TradingSequenceDataset(
            val_features, val_labels, self.sequence_length
        )
        test_dataset = TradingSequenceDataset(
            test_features, test_labels, self.sequence_length
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self,
                          train_dataset: Dataset,
                          val_dataset: Dataset,
                          test_dataset: Dataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders.
        """
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['training'].get('num_workers', 4)
        pin_memory = self.config['training'].get('pin_memory', True)
        
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
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return train_loader, val_loader, test_loader
    
    def save_scaler(self, path: str):
        """Save the scaler for inference"""
        if self.scaler is not None:
            with open(path, 'wb') as f:
                pickle.dump(self.scaler, f)
    
    def load_scaler(self, path: str):
        """Load the scaler for inference"""
        if os.path.exists(path):
            with open(path, 'rb') as f:
                self.scaler = pickle.load(f)
