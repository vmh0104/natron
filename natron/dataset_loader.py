"""
Natron Dataset Loader
PyTorch Dataset and DataLoader for sequence-based training.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NatronDataset(Dataset):
    """
    Dataset for Natron transformer model.
    Creates sequences of candles with multi-head targets.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        sequence_length: int = 96,
        forecast_horizon: int = 5,
        feature_columns: Optional[List[str]] = None,
        normalize: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            df: Processed DataFrame with features and regime labels
            sequence_length: Number of candles per sequence
            forecast_horizon: Number of candles ahead to forecast
            feature_columns: List of feature column names (if None, auto-detect)
            normalize: Whether to normalize features
        """
        self.df = df.copy()
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        
        # Identify feature columns (exclude metadata columns)
        exclude_cols = ['time', 'regime', 'regime_label', 'open', 'high', 'low', 'close', 'volume']
        if feature_columns is None:
            self.feature_columns = [col for col in self.df.columns 
                                   if col not in exclude_cols]
        else:
            self.feature_columns = feature_columns
        
        logger.info(f"Using {len(self.feature_columns)} features")
        
        # Extract feature matrix
        self.features = self.df[self.feature_columns].values.astype(np.float32)
        
        # Normalize features
        if normalize:
            self.feature_mean = self.features.mean(axis=0, keepdims=True)
            self.feature_std = self.features.std(axis=0, keepdims=True) + 1e-8
            self.features = (self.features - self.feature_mean) / self.feature_std
        else:
            self.feature_mean = None
            self.feature_std = None
        
        # Extract targets
        self.regime_labels = self.df['regime_label'].values.astype(np.int64)
        
        # Calculate context strength (|bull_p - bear_p|)
        # Approximate: if regime is BULL_*, context = 1.0, if BEAR_*, context = 1.0, else lower
        self.context_strength = self._calculate_context_strength()
        
        # Calculate forecast direction (up/down over forecast_horizon)
        self.forecast_labels = self._calculate_forecast_labels()
        
        # Valid indices (sequences that don't go out of bounds)
        self.valid_indices = self._get_valid_indices()
        
        logger.info(f"Created dataset with {len(self.valid_indices)} valid sequences")
    
    def _calculate_context_strength(self) -> np.ndarray:
        """Calculate context strength from regime labels."""
        strength = np.zeros(len(self.df))
        
        # Strong regimes have high context strength
        regime_map = {
            0: 1.0,  # BULL_STRONG
            1: 0.6,  # BULL_WEAK
            2: 1.0,  # BEAR_STRONG
            3: 0.6,  # BEAR_WEAK
            4: 0.3,  # RANGE
            5: 0.8   # VOLATILE
        }
        
        for idx, regime in enumerate(self.regime_labels):
            strength[idx] = regime_map.get(regime, 0.5)
        
        return strength.astype(np.float32)
    
    def _calculate_forecast_labels(self) -> np.ndarray:
        """
        Calculate forecast direction labels.
        Returns 1 if price goes up over forecast_horizon, 0 otherwise.
        """
        if 'close' not in self.df.columns:
            # Fallback: use returns if close not available
            returns = self.df.get('returns', pd.Series(0, index=self.df.index))
            future_returns = returns.shift(-self.forecast_horizon)
            labels = (future_returns > 0).astype(int).values
        else:
            current_price = self.df['close'].values
            future_price = self.df['close'].shift(-self.forecast_horizon).values
            
            # Handle NaN at the end
            labels = np.zeros(len(self.df), dtype=np.int64)
            valid_mask = ~np.isnan(future_price)
            labels[valid_mask] = (future_price[valid_mask] > current_price[valid_mask]).astype(np.int64)
        
        return labels
    
    def _get_valid_indices(self) -> List[int]:
        """Get indices that can form valid sequences."""
        valid = []
        max_idx = len(self.df) - self.sequence_length - self.forecast_horizon
        
        for i in range(max_idx):
            # Check if forecast label is valid
            if not np.isnan(self.forecast_labels[i + self.sequence_length - 1]):
                valid.append(i)
        
        return valid
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sequence sample.
        
        Returns:
            Dictionary with:
            - 'features': (sequence_length, num_features) input sequence
            - 'regime': regime label (scalar)
            - 'context': context strength (scalar)
            - 'forecast': forecast label (scalar)
        """
        start_idx = self.valid_indices[idx]
        end_idx = start_idx + self.sequence_length
        
        # Extract sequence
        sequence = self.features[start_idx:end_idx]  # (seq_len, features)
        
        # Extract targets (use last candle in sequence)
        target_idx = end_idx - 1
        regime_label = self.regime_labels[target_idx]
        context_strength = self.context_strength[target_idx]
        forecast_label = self.forecast_labels[target_idx]
        
        return {
            'features': torch.from_numpy(sequence),
            'regime': torch.tensor(regime_label, dtype=torch.long),
            'context': torch.tensor(context_strength, dtype=torch.float32),
            'forecast': torch.tensor(forecast_label, dtype=torch.long)
        }


def create_dataloaders(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    sequence_length: int = 96,
    forecast_horizon: int = 5,
    batch_size: int = 32,
    num_workers: int = 4,
    feature_columns: Optional[List[str]] = None
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.
    
    Args:
        df: Processed DataFrame
        train_ratio: Ratio of data for training
        val_ratio: Ratio of data for validation
        sequence_length: Sequence length
        forecast_horizon: Forecast horizon
        batch_size: Batch size
        num_workers: Number of data loader workers
        feature_columns: Optional list of feature columns
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Split data chronologically
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_val = df.iloc[n_train:n_train + n_val].reset_index(drop=True)
    df_test = df.iloc[n_train + n_val:].reset_index(drop=True)
    
    logger.info(f"Data split: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")
    
    # Create datasets
    train_dataset = NatronDataset(
        df_train, sequence_length, forecast_horizon, feature_columns, normalize=True
    )
    val_dataset = NatronDataset(
        df_val, sequence_length, forecast_horizon, feature_columns, normalize=False
    )
    test_dataset = NatronDataset(
        df_test, sequence_length, forecast_horizon, feature_columns, normalize=False
    )
    
    # Use train dataset normalization stats for val/test
    if train_dataset.feature_mean is not None:
        val_dataset.feature_mean = train_dataset.feature_mean
        val_dataset.feature_std = train_dataset.feature_std
        test_dataset.feature_mean = train_dataset.feature_mean
        test_dataset.feature_std = train_dataset.feature_std
    
    # Create dataloaders
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
