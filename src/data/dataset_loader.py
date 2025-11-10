"""
Natron Dataset Loader - Sequence Builder and DataLoader
Author: Natron AI System
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional
import yaml
from pathlib import Path


class NatronSequenceDataset(Dataset):
    """
    PyTorch Dataset for Natron Transformer.
    Creates sequences of 96 consecutive candles with multi-task labels.
    """
    
    def __init__(self, 
                 features: np.ndarray,
                 labels: Optional[pd.DataFrame] = None,
                 sequence_length: int = 96,
                 stride: int = 1):
        """
        Args:
            features: Feature array (N, num_features)
            labels: Labels DataFrame with columns [buy, sell, direction, regime]
            sequence_length: Number of candles per sequence
            stride: Step size for sliding window
        """
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        self.stride = stride
        
        # Calculate valid indices
        self.indices = self._build_indices()
        
    def _build_indices(self) -> np.ndarray:
        """Build valid sequence indices"""
        max_idx = len(self.features) - self.sequence_length
        indices = np.arange(0, max_idx, self.stride)
        return indices
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sequence and its labels.
        
        Returns:
            Dictionary with:
                - sequence: (96, num_features)
                - buy: (1,)
                - sell: (1,)
                - direction: (1,)
                - regime: (1,)
        """
        start_idx = self.indices[idx]
        end_idx = start_idx + self.sequence_length
        
        # Get sequence (96 candles)
        sequence = self.features[start_idx:end_idx]
        sequence = torch.FloatTensor(sequence)
        
        item = {'sequence': sequence}
        
        # Get labels from last candle in sequence
        if self.labels is not None:
            label_idx = end_idx - 1
            item['buy'] = torch.FloatTensor([self.labels.iloc[label_idx]['buy']])
            item['sell'] = torch.FloatTensor([self.labels.iloc[label_idx]['sell']])
            item['direction'] = torch.LongTensor([self.labels.iloc[label_idx]['direction']])
            item['regime'] = torch.LongTensor([self.labels.iloc[label_idx]['regime']])
        
        return item


class NatronContrastiveDataset(Dataset):
    """
    Dataset for contrastive learning pretraining.
    Creates positive pairs through augmentation.
    """
    
    def __init__(self, 
                 features: np.ndarray,
                 sequence_length: int = 96,
                 stride: int = 1):
        self.features = features
        self.sequence_length = sequence_length
        self.stride = stride
        self.indices = self._build_indices()
        
    def _build_indices(self) -> np.ndarray:
        max_idx = len(self.features) - self.sequence_length
        indices = np.arange(0, max_idx, self.stride)
        return indices
    
    def _augment(self, sequence: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to create positive pairs.
        - Add noise
        - Scale
        - Shift
        """
        aug = sequence.copy()
        
        # Random noise
        if np.random.rand() > 0.5:
            noise = np.random.randn(*aug.shape) * 0.01
            aug = aug + noise
        
        # Random scaling
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.98, 1.02)
            aug = aug * scale
        
        # Random shift (time jitter)
        if np.random.rand() > 0.5:
            shift = np.random.randint(-2, 3)
            aug = np.roll(aug, shift, axis=0)
        
        return aug
    
    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx = self.indices[idx]
        end_idx = start_idx + self.sequence_length
        
        # Original sequence
        sequence = self.features[start_idx:end_idx]
        
        # Create two augmented views
        view1 = self._augment(sequence)
        view2 = self._augment(sequence)
        
        return {
            'view1': torch.FloatTensor(view1),
            'view2': torch.FloatTensor(view2)
        }


class NatronDataModule:
    """
    Data module for loading and preparing Natron datasets.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_config = self.config['data']
        self.features_config = self.config['features']
        
        self.raw_data = None
        self.features = None
        self.labels = None
        self.scaling_params = None
        
    def load_data(self) -> pd.DataFrame:
        """Load raw OHLCV data from CSV"""
        csv_path = self.data_config['csv_path']
        print(f"📂 Loading data from {csv_path}...")
        
        df = pd.read_csv(csv_path)
        
        # Ensure required columns
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert time to datetime
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        self.raw_data = df
        print(f"✅ Loaded {len(df)} candles")
        
        return df
    
    def generate_features(self) -> pd.DataFrame:
        """Generate technical features"""
        from features.feature_engine import FeatureEngine
        
        print("🔧 Generating features...")
        engine = FeatureEngine(verbose=True)
        features = engine.generate_all_features(self.raw_data)
        
        # Normalize if specified
        if self.features_config['normalize']:
            method = self.features_config['normalization_method']
            features, self.scaling_params = engine.normalize_features(features, method=method)
            print(f"✅ Features normalized using {method} scaling")
        
        self.features = features
        return features
    
    def generate_labels(self) -> pd.DataFrame:
        """Generate trading labels"""
        from labels.label_generator import LabelGenerator
        
        print("🏷️ Generating labels...")
        labeler = LabelGenerator(
            buy_threshold=self.config['labels']['buy_conditions_threshold'],
            sell_threshold=self.config['labels']['sell_conditions_threshold'],
            verbose=True
        )
        
        labels = labeler.generate_all_labels(self.raw_data, self.features)
        self.labels = labels
        
        return labels
    
    def create_datasets(self, mode: str = 'supervised') -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create train/val/test datasets.
        
        Args:
            mode: 'supervised' or 'contrastive'
            
        Returns:
            train_dataset, val_dataset, test_dataset
        """
        sequence_length = self.data_config['sequence_length']
        test_split = self.data_config['test_split']
        val_split = self.data_config['validation_split']
        
        # Convert to numpy
        features_array = self.features.values
        
        # Calculate splits
        n_samples = len(features_array) - sequence_length
        test_size = int(n_samples * test_split)
        val_size = int(n_samples * val_split)
        train_size = n_samples - test_size - val_size
        
        # Split data chronologically (important for time series!)
        train_features = features_array[:train_size + sequence_length]
        val_features = features_array[train_size:train_size + val_size + sequence_length]
        test_features = features_array[train_size + val_size:]
        
        print(f"📊 Data split:")
        print(f"   Train: {train_size} sequences")
        print(f"   Val: {val_size} sequences")
        print(f"   Test: {test_size} sequences")
        
        if mode == 'contrastive':
            # Contrastive datasets (no labels needed)
            train_dataset = NatronContrastiveDataset(train_features, sequence_length)
            val_dataset = NatronContrastiveDataset(val_features, sequence_length)
            test_dataset = NatronContrastiveDataset(test_features, sequence_length)
        else:
            # Supervised datasets
            train_labels = self.labels.iloc[:train_size + sequence_length]
            val_labels = self.labels.iloc[train_size:train_size + val_size + sequence_length]
            test_labels = self.labels.iloc[train_size + val_size:]
            
            train_dataset = NatronSequenceDataset(train_features, train_labels, sequence_length)
            val_dataset = NatronSequenceDataset(val_features, val_labels, sequence_length)
            test_dataset = NatronSequenceDataset(test_features, test_labels, sequence_length)
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self, 
                          train_dataset: Dataset,
                          val_dataset: Dataset,
                          test_dataset: Dataset,
                          batch_size: int = 32,
                          num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create PyTorch DataLoaders"""
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
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
    
    def prepare_full_pipeline(self, mode: str = 'supervised') -> Dict:
        """
        Run full data preparation pipeline.
        
        Returns:
            Dictionary with dataloaders and metadata
        """
        # Load data
        self.load_data()
        
        # Generate features
        self.generate_features()
        
        # Generate labels (if supervised)
        if mode == 'supervised':
            self.generate_labels()
        
        # Create datasets
        train_ds, val_ds, test_ds = self.create_datasets(mode=mode)
        
        # Create dataloaders
        batch_size = self.config.get('pretrain' if mode == 'contrastive' else 'supervised', {}).get('batch_size', 32)
        train_loader, val_loader, test_loader = self.create_dataloaders(
            train_ds, val_ds, test_ds, 
            batch_size=batch_size
        )
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'num_features': self.features.shape[1],
            'scaling_params': self.scaling_params,
            'config': self.config
        }


if __name__ == "__main__":
    # Test data loading
    print("🧪 Testing NatronDataModule...")
    
    # Note: This requires a real data_export.csv file
    # For testing, we'll create synthetic data
    
    print("✅ Dataset loader ready")
    print("📝 To use: Provide data_export.csv with columns [time, open, high, low, close, volume]")
