"""
Natron DatasetLoader - Create sequences of 96 consecutive candles for Transformer training.

Input: Features DataFrame (N, 100) + Labels
Output: Sequences (N-96, 96, 100) with corresponding labels
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Optional


class NatronDataset(Dataset):
    """Dataset for Natron Transformer model."""
    
    def __init__(
        self,
        features: np.ndarray,
        labels: Optional[dict] = None,
        sequence_length: int = 96,
        mode: str = 'supervised'
    ):
        """
        Args:
            features: Feature array of shape (N, feature_dim)
            labels: Dict with keys ['buy', 'sell', 'direction', 'regime'] or None for pretraining
            sequence_length: Length of input sequences (default 96)
            mode: 'pretrain', 'supervised', or 'rl'
        """
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        self.mode = mode
        
        # Calculate valid samples
        self.n_samples = len(features) - sequence_length + 1
        
        if self.n_samples <= 0:
            raise ValueError(f"Not enough data. Need at least {sequence_length} samples.")
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a sequence and its labels.
        
        Returns:
            Dictionary with 'features' and optionally 'labels'
        """
        # Extract sequence
        seq = self.features[idx:idx + self.sequence_length]
        
        # Convert to tensor
        seq_tensor = torch.FloatTensor(seq)
        
        result = {'features': seq_tensor}
        
        # Add labels if in supervised mode
        if self.mode == 'supervised' and self.labels is not None:
            label_idx = idx + self.sequence_length - 1  # Label at end of sequence
            
            result['buy'] = torch.FloatTensor([self.labels['buy'][label_idx]])
            result['sell'] = torch.FloatTensor([self.labels['sell'][label_idx]])
            result['direction'] = torch.LongTensor([self.labels['direction'][label_idx]])
            result['regime'] = torch.LongTensor([self.labels['regime'][label_idx]])
        
        # For pretraining, return masked version
        elif self.mode == 'pretrain':
            # Create masked sequence (random masking)
            masked_seq, mask = self._create_masked_sequence(seq_tensor)
            result['masked_features'] = masked_seq
            result['mask'] = mask
            result['original_features'] = seq_tensor
        
        return result
    
    def _create_masked_sequence(self, seq: torch.Tensor, mask_ratio: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
        """Create masked sequence for pretraining."""
        masked_seq = seq.clone()
        mask = torch.zeros(seq.shape[0], dtype=torch.bool)
        
        # Randomly mask tokens
        n_mask = int(seq.shape[0] * mask_ratio)
        mask_indices = torch.randperm(seq.shape[0])[:n_mask]
        mask[mask_indices] = True
        
        # Replace masked tokens with zeros (or random noise)
        masked_seq[mask_indices] = 0
        
        return masked_seq, mask


class SequenceCreator:
    """Helper class to create sequences from DataFrame."""
    
    def __init__(self, sequence_length: int = 96):
        """
        Args:
            sequence_length: Number of consecutive candles per sequence
        """
        self.sequence_length = sequence_length
    
    def create_sequences(
        self,
        features_df: pd.DataFrame,
        labels: Optional[dict] = None
    ) -> Tuple[np.ndarray, Optional[dict]]:
        """
        Create sequences from features and labels.
        
        Args:
            features_df: DataFrame with features (N, feature_dim)
            labels: Optional dict with label Series
        
        Returns:
            Tuple of (features_array, labels_dict)
        """
        # Convert features to numpy
        features_array = features_df.values.astype(np.float32)
        
        # Handle labels
        labels_dict = None
        if labels is not None:
            labels_dict = {}
            for key, label_series in labels.items():
                # Align labels with sequence end positions
                label_values = label_series.values
                labels_dict[key] = label_values
        
        return features_array, labels_dict
    
    def create_dataloaders(
        self,
        features_array: np.ndarray,
        labels_dict: Optional[dict] = None,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        batch_size: int = 32,
        mode: str = 'supervised',
        shuffle: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train/val/test dataloaders.
        
        Args:
            features_array: Feature array (N, feature_dim)
            labels_dict: Optional labels dict
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            batch_size: Batch size
            mode: 'pretrain', 'supervised', or 'rl'
            shuffle: Whether to shuffle training data
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        n_samples = len(features_array) - self.sequence_length + 1
        
        # Split indices
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        # Split features
        train_features = features_array[:train_end + self.sequence_length - 1]
        val_features = features_array[train_end:val_end + self.sequence_length - 1]
        test_features = features_array[val_end:]
        
        # Split labels if provided
        train_labels = None
        val_labels = None
        test_labels = None
        
        if labels_dict is not None:
            train_labels = {k: v[:train_end + self.sequence_length - 1] for k, v in labels_dict.items()}
            val_labels = {k: v[train_end:val_end + self.sequence_length - 1] for k, v in labels_dict.items()}
            test_labels = {k: v[val_end:] for k, v in labels_dict.items()}
        
        # Create datasets
        train_dataset = NatronDataset(train_features, train_labels, self.sequence_length, mode)
        val_dataset = NatronDataset(val_features, val_labels, self.sequence_length, mode)
        test_dataset = NatronDataset(test_features, test_labels, self.sequence_length, mode)
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=2,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
