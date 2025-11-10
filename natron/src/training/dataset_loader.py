"""
Dataset Loader for Natron Training

This module provides PyTorch Dataset and DataLoader classes
for loading sequences of market data for transformer training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NatronDataset(Dataset):
    """
    Dataset class for Natron transformer training.
    
    Creates sequences of fixed length from time series data.
    """
    
    def __init__(self,
                 df: pd.DataFrame,
                 feature_cols: list,
                 seq_len: int = 96,
                 target_col: str = 'target',
                 regime_col: str = 'regime',
                 create_context_target: bool = True):
        """
        Initialize dataset.
        
        Args:
            df: DataFrame with features and targets
            feature_cols: List of feature column names
            seq_len: Sequence length (number of candles)
            target_col: Name of forecast target column
            regime_col: Name of regime column
            create_context_target: Whether to create context target from regime
        """
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.seq_len = seq_len
        self.target_col = target_col
        self.regime_col = regime_col
        
        # Extract features and targets
        self.features = self.df[feature_cols].values.astype(np.float32)
        
        # Regime targets
        if regime_col in self.df.columns:
            self.regime_targets = self.df[regime_col].values.astype(np.int64)
        else:
            logger.warning(f"Regime column '{regime_col}' not found, using zeros")
            self.regime_targets = np.zeros(len(self.df), dtype=np.int64)
        
        # Forecast targets
        if target_col in self.df.columns:
            self.forecast_targets = self.df[target_col].values.astype(np.int64)
        else:
            logger.warning(f"Target column '{target_col}' not found, using zeros")
            self.forecast_targets = np.zeros(len(self.df), dtype=np.int64)
        
        # Create context targets (|bull_p - bear_p|)
        if create_context_target:
            self.context_targets = self._create_context_targets()
        else:
            self.context_targets = np.zeros(len(self.df), dtype=np.float32)
        
        # Create valid indices (sequences that don't go out of bounds)
        self.valid_indices = []
        for i in range(len(self.df) - seq_len + 1):
            self.valid_indices.append(i)
        
        logger.info(f"Created dataset with {len(self.valid_indices)} valid sequences")
        logger.info(f"Feature dimension: {len(feature_cols)}")
        logger.info(f"Sequence length: {seq_len}")
    
    def _create_context_targets(self) -> np.ndarray:
        """
        Create context strength targets from regime labels.
        
        Context strength = |P(bull) - P(bear)|
        where P(bull) = (BULL_STRONG + BULL_WEAK) / total
        and P(bear) = (BEAR_STRONG + BEAR_WEAK) / total
        
        For single labels, we use a sliding window to estimate probabilities.
        
        Returns:
            Array of context strength values (0-1)
        """
        context_targets = np.zeros(len(self.df), dtype=np.float32)
        window = 20  # Window for probability estimation
        
        for i in range(len(self.df)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(self.df), i + window // 2 + 1)
            
            window_regimes = self.regime_targets[start_idx:end_idx]
            
            # Count bull and bear regimes
            bull_count = np.sum((window_regimes == 0) | (window_regimes == 1))  # BULL_STRONG, BULL_WEAK
            bear_count = np.sum((window_regimes == 2) | (window_regimes == 3))  # BEAR_STRONG, BEAR_WEAK
            
            total = len(window_regimes)
            if total > 0:
                bull_p = bull_count / total
                bear_p = bear_count / total
                context_strength = abs(bull_p - bear_p)
            else:
                context_strength = 0.0
            
            context_targets[i] = context_strength
        
        return context_targets
    
    def __len__(self) -> int:
        """Return number of valid sequences."""
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sequence sample.
        
        Args:
            idx: Index into valid_indices
            
        Returns:
            Dictionary with features and targets
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.seq_len
        
        # Extract sequence
        sequence = self.features[start_idx:end_idx]  # [seq_len, num_features]
        
        # Targets are for the last timestep in the sequence
        target_idx = end_idx - 1
        
        regime_target = self.regime_targets[target_idx]
        forecast_target = self.forecast_targets[target_idx]
        context_target = self.context_targets[target_idx]
        
        return {
            'features': torch.tensor(sequence, dtype=torch.float32),
            'regime_target': torch.tensor(regime_target, dtype=torch.long),
            'forecast_target': torch.tensor(forecast_target, dtype=torch.long),
            'context_target': torch.tensor(context_target, dtype=torch.float32)
        }


def create_dataloaders(df: pd.DataFrame,
                       feature_cols: list,
                       train_ratio: float = 0.7,
                       val_ratio: float = 0.15,
                       seq_len: int = 96,
                       batch_size: int = 32,
                       num_workers: int = 4,
                       shuffle_train: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/validation/test dataloaders.
    
    Args:
        df: Full DataFrame
        feature_cols: List of feature column names
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        seq_len: Sequence length
        batch_size: Batch size
        num_workers: Number of worker processes
        shuffle_train: Whether to shuffle training data
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Split data chronologically
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    df_train = df.iloc[:n_train].copy()
    df_val = df.iloc[n_train:n_train + n_val].copy()
    df_test = df.iloc[n_train + n_val:].copy()
    
    logger.info(f"Data split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    
    # Create datasets
    train_dataset = NatronDataset(df_train, feature_cols, seq_len)
    val_dataset = NatronDataset(df_val, feature_cols, seq_len)
    test_dataset = NatronDataset(df_test, feature_cols, seq_len)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
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
