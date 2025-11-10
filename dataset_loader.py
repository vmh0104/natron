"""
Dataset Loader - Sequence Construction for Natron Transformer
Creates sequences of 96 consecutive candles with multi-task labels.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional
try:
    from feature_engine import FeatureEngine
    from label_generator import LabelGenerator
except ImportError:
    # Handle relative imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from feature_engine import FeatureEngine
    from label_generator import LabelGenerator


class NatronDataset(Dataset):
    """
    PyTorch Dataset for Natron Transformer.
    Each sample: (96, 100) feature sequence → (buy, sell, direction, regime) labels
    """
    
    def __init__(
        self,
        features_df: pd.DataFrame,
        sequence_length: int = 96,
        feature_columns: Optional[list] = None
    ):
        """
        Args:
            features_df: DataFrame with features and labels
            sequence_length: Number of consecutive candles (default: 96)
            feature_columns: List of feature column names (auto-detected if None)
        """
        self.sequence_length = sequence_length
        self.features_df = features_df.copy()
        
        # Identify feature columns (exclude time, OHLCV, labels)
        exclude_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 
                        'buy', 'sell', 'direction', 'regime']
        
        if feature_columns is None:
            self.feature_columns = [col for col in self.features_df.columns 
                                   if col not in exclude_cols]
        else:
            self.feature_columns = feature_columns
        
        # Ensure we have required label columns
        required_labels = ['buy', 'sell', 'direction', 'regime']
        for label in required_labels:
            if label not in self.features_df.columns:
                raise ValueError(f"Missing required label column: {label}")
        
        # Extract feature matrix and labels
        self.feature_matrix = self.features_df[self.feature_columns].values.astype(np.float32)
        self.buy_labels = self.features_df['buy'].values.astype(np.float32)
        self.sell_labels = self.features_df['sell'].values.astype(np.float32)
        self.direction_labels = self.features_df['direction'].values.astype(np.int64)
        self.regime_labels = self.features_df['regime'].values.astype(np.int64)
        
        # Create valid indices (need sequence_length candles before label)
        self.valid_indices = list(range(len(self.feature_matrix) - sequence_length))
        
        print(f"Dataset initialized: {len(self.valid_indices)} samples, "
              f"{len(self.feature_columns)} features, sequence length: {sequence_length}")
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        """
        Returns:
            X: (sequence_length, num_features) tensor
            labels: dict with 'buy', 'sell', 'direction', 'regime'
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length
        
        # Extract sequence
        X = torch.FloatTensor(self.feature_matrix[start_idx:end_idx])
        
        # Extract labels (at the end of sequence)
        label_idx = end_idx - 1
        
        labels = {
            'buy': torch.FloatTensor([self.buy_labels[label_idx]]),
            'sell': torch.FloatTensor([self.sell_labels[label_idx]]),
            'direction': torch.LongTensor([self.direction_labels[label_idx]]),
            'regime': torch.LongTensor([self.regime_labels[label_idx]])
        }
        
        return X, labels


class SequenceCreator:
    """
    High-level interface for creating sequences from raw OHLCV data.
    Handles feature engineering, labeling, and dataset creation.
    """
    
    def __init__(self, sequence_length: int = 96):
        """
        Args:
            sequence_length: Number of consecutive candles per sequence
        """
        self.sequence_length = sequence_length
        self.feature_engine = FeatureEngine()
        self.label_generator = LabelGenerator()
        self.feature_columns = None
    
    def create_dataset(
        self,
        df: pd.DataFrame,
        train_split: float = 0.8,
        val_split: float = 0.1
    ) -> Tuple[NatronDataset, NatronDataset, NatronDataset]:
        """
        Create train/val/test datasets from raw OHLCV DataFrame.
        
        Args:
            df: DataFrame with columns ['time', 'open', 'high', 'low', 'close', 'volume']
            train_split: Fraction for training (default: 0.8)
            val_split: Fraction for validation (default: 0.1)
            
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        print("Step 1: Feature Engineering...")
        features_df = self.feature_engine.fit_transform(df)
        
        print("Step 2: Label Generation...")
        labels_df = self.label_generator.generate_labels(features_df)
        
        # Store feature columns for later use
        self.feature_columns = self.feature_engine.get_feature_columns(labels_df)
        
        # Print label statistics
        stats = self.label_generator.get_label_stats(labels_df)
        print(f"\nLabel Statistics:")
        print(f"  Buy rate: {stats['buy_rate']:.3f}")
        print(f"  Sell rate: {stats['sell_rate']:.3f}")
        print(f"  Direction up rate: {stats['direction_up_rate']:.3f}")
        print(f"  Regime distribution: {stats['regime_distribution']}")
        
        # Split data (temporal split, no shuffling)
        n_total = len(labels_df)
        n_train = int(n_total * train_split)
        n_val = int(n_total * (train_split + val_split))
        
        train_df = labels_df.iloc[:n_train].copy()
        val_df = labels_df.iloc[n_train:n_val].copy()
        test_df = labels_df.iloc[n_val:].copy()
        
        print(f"\nData Split:")
        print(f"  Train: {len(train_df)} samples")
        print(f"  Val: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        
        # Create datasets
        train_dataset = NatronDataset(
            train_df,
            sequence_length=self.sequence_length,
            feature_columns=self.feature_columns
        )
        val_dataset = NatronDataset(
            val_df,
            sequence_length=self.sequence_length,
            feature_columns=self.feature_columns
        )
        test_dataset = NatronDataset(
            test_df,
            sequence_length=self.sequence_length,
            feature_columns=self.feature_columns
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(
        self,
        train_dataset: NatronDataset,
        val_dataset: NatronDataset,
        test_dataset: NatronDataset,
        batch_size: int = 32,
        num_workers: int = 4
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders from datasets.
        
        Args:
            train_dataset, val_dataset, test_dataset: NatronDataset instances
            batch_size: Batch size for training
            num_workers: Number of worker processes
            
        Returns:
            train_loader, val_loader, test_loader
        """
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
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
        
        return train_loader, val_loader, test_loader
