"""
Natron Transformer - Dataset Loader Module
Creates sequences and PyTorch datasets from OHLCV data
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional
import yaml
from pathlib import Path


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for sequence data.
    Each sample: (96, 100) feature sequence + labels
    """
    
    def __init__(
        self,
        sequences: np.ndarray,
        buy_labels: np.ndarray,
        sell_labels: np.ndarray,
        direction_labels: np.ndarray,
        regime_labels: np.ndarray
    ):
        """
        Args:
            sequences: (N, 96, 100) array of feature sequences
            buy_labels: (N,) array of buy signals
            sell_labels: (N,) array of sell signals
            direction_labels: (N,) array of direction labels
            regime_labels: (N,) array of regime labels
        """
        self.sequences = torch.FloatTensor(sequences)
        self.buy_labels = torch.FloatTensor(buy_labels)
        self.sell_labels = torch.FloatTensor(sell_labels)
        self.direction_labels = torch.LongTensor(direction_labels)
        self.regime_labels = torch.LongTensor(regime_labels)
        
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple:
        return (
            self.sequences[idx],
            self.buy_labels[idx],
            self.sell_labels[idx],
            self.direction_labels[idx],
            self.regime_labels[idx]
        )


class NatronDataLoader:
    """
    Main data loader for Natron Transformer.
    Handles data loading, feature engineering, labeling, and sequence creation.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.sequence_length = self.config['data']['sequence_length']
        self.train_split = self.config['data']['train_split']
        self.val_split = self.config['data']['val_split']
        self.test_split = self.config['data']['test_split']
        
        self.df = None
        self.features = None
        self.labels = None
        
    def load_and_prepare(self, csv_path: Optional[str] = None) -> Dict:
        """
        Complete data pipeline: load → feature engineering → labeling → sequences
        
        Args:
            csv_path: Path to CSV file (overrides config)
            
        Returns:
            Dictionary with train/val/test DataLoaders
        """
        # 1. Load raw data
        if csv_path is None:
            csv_path = self.config['data']['csv_path']
        
        print(f"📂 Loading data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # Validate columns
        required_cols = ['time'] + self.config['data']['ohlcv_cols']
        for col in required_cols:
            if col not in self.df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        print(f"  Loaded {len(self.df)} rows")
        
        # 2. Feature engineering
        print(f"🔧 Generating features...")
        from feature_engine import FeatureEngine
        engine = FeatureEngine(self.config)
        self.features = engine.generate_all_features(self.df)
        
        # 3. Generate labels
        print(f"🏷️  Generating labels...")
        from labeling import LabelGenerator
        labeler = LabelGenerator(self.config)
        buy, sell, direction, regime = labeler.generate_all_labels(self.df, self.features)
        
        self.labels = {
            'buy': buy.values,
            'sell': sell.values,
            'direction': direction.values,
            'regime': regime.values
        }
        
        # 4. Create sequences
        print(f"📦 Creating sequences (length={self.sequence_length})...")
        sequences, labels_dict = self._create_sequences()
        
        # 5. Split into train/val/test
        print(f"✂️  Splitting data (train={self.train_split}, val={self.val_split}, test={self.test_split})...")
        train_data, val_data, test_data = self._split_data(sequences, labels_dict)
        
        # 6. Create PyTorch datasets and dataloaders
        print(f"🔄 Creating DataLoaders...")
        dataloaders = self._create_dataloaders(train_data, val_data, test_data)
        
        print(f"✅ Data preparation complete!")
        print(f"  Train samples: {len(train_data[0])}")
        print(f"  Val samples: {len(val_data[0])}")
        print(f"  Test samples: {len(test_data[0])}")
        
        return dataloaders
    
    def _create_sequences(self) -> Tuple[np.ndarray, Dict]:
        """
        Create sliding window sequences.
        
        Returns:
            Tuple of (sequences, labels_dict)
            - sequences: (N, 96, 100) array
            - labels_dict: Dict with buy/sell/direction/regime arrays
        """
        feature_array = self.features.values
        n_samples = len(feature_array) - self.sequence_length
        n_features = feature_array.shape[1]
        
        # Initialize arrays
        sequences = np.zeros((n_samples, self.sequence_length, n_features))
        buy_labels = np.zeros(n_samples)
        sell_labels = np.zeros(n_samples)
        direction_labels = np.zeros(n_samples)
        regime_labels = np.zeros(n_samples)
        
        # Create sequences
        for i in range(n_samples):
            # Input: 96 consecutive candles
            sequences[i] = feature_array[i:i+self.sequence_length]
            
            # Labels: at the end of the sequence (candle 95)
            label_idx = i + self.sequence_length - 1
            buy_labels[i] = self.labels['buy'][label_idx]
            sell_labels[i] = self.labels['sell'][label_idx]
            direction_labels[i] = self.labels['direction'][label_idx]
            regime_labels[i] = self.labels['regime'][label_idx]
        
        labels_dict = {
            'buy': buy_labels,
            'sell': sell_labels,
            'direction': direction_labels,
            'regime': regime_labels
        }
        
        return sequences, labels_dict
    
    def _split_data(
        self, 
        sequences: np.ndarray, 
        labels_dict: Dict
    ) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Split data into train/val/test sets.
        Uses temporal split (not random) to prevent data leakage.
        
        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        n_samples = len(sequences)
        
        # Calculate split indices
        train_end = int(n_samples * self.train_split)
        val_end = int(n_samples * (self.train_split + self.val_split))
        
        # Split sequences
        train_seq = sequences[:train_end]
        val_seq = sequences[train_end:val_end]
        test_seq = sequences[val_end:]
        
        # Split labels
        train_labels = {k: v[:train_end] for k, v in labels_dict.items()}
        val_labels = {k: v[train_end:val_end] for k, v in labels_dict.items()}
        test_labels = {k: v[val_end:] for k, v in labels_dict.items()}
        
        # Package into tuples
        train_data = (
            train_seq,
            train_labels['buy'],
            train_labels['sell'],
            train_labels['direction'],
            train_labels['regime']
        )
        
        val_data = (
            val_seq,
            val_labels['buy'],
            val_labels['sell'],
            val_labels['direction'],
            val_labels['regime']
        )
        
        test_data = (
            test_seq,
            test_labels['buy'],
            test_labels['sell'],
            test_labels['direction'],
            test_labels['regime']
        )
        
        return train_data, val_data, test_data
    
    def _create_dataloaders(
        self,
        train_data: Tuple,
        val_data: Tuple,
        test_data: Tuple
    ) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders.
        
        Returns:
            Dictionary with 'train', 'val', 'test' DataLoaders
        """
        # Create datasets
        train_dataset = SequenceDataset(*train_data)
        val_dataset = SequenceDataset(*val_data)
        test_dataset = SequenceDataset(*test_data)
        
        # Get batch sizes from config
        pretrain_batch = self.config['training']['pretrain']['batch_size']
        supervised_batch = self.config['training']['supervised']['batch_size']
        
        # Create dataloaders
        dataloaders = {
            'train': DataLoader(
                train_dataset,
                batch_size=supervised_batch,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            ),
            'val': DataLoader(
                val_dataset,
                batch_size=supervised_batch,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            ),
            'test': DataLoader(
                test_dataset,
                batch_size=supervised_batch,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            ),
            'pretrain': DataLoader(
                train_dataset,
                batch_size=pretrain_batch,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )
        }
        
        return dataloaders
    
    def normalize_features(self, method: str = 'zscore'):
        """
        Normalize features (optional, applied before sequence creation).
        
        Args:
            method: 'zscore' or 'minmax'
        """
        if self.features is None:
            raise ValueError("Features not generated yet. Call load_and_prepare first.")
        
        if method == 'zscore':
            # Z-score normalization
            mean = self.features.mean()
            std = self.features.std()
            self.features = (self.features - mean) / (std + 1e-8)
        
        elif method == 'minmax':
            # Min-max normalization
            min_val = self.features.min()
            max_val = self.features.max()
            self.features = (self.features - min_val) / (max_val - min_val + 1e-8)
        
        print(f"✅ Features normalized using {method}")


class PretrainDataset(Dataset):
    """
    Dataset for unsupervised pretraining.
    Returns sequences without labels.
    """
    
    def __init__(self, sequences: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.sequences[idx]


def create_pretrain_loader(
    sequences: np.ndarray,
    batch_size: int = 256,
    shuffle: bool = True
) -> DataLoader:
    """
    Create DataLoader for pretraining phase.
    
    Args:
        sequences: (N, 96, 100) array of sequences
        batch_size: Batch size
        shuffle: Whether to shuffle
        
    Returns:
        DataLoader for pretraining
    """
    dataset = PretrainDataset(sequences)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )


if __name__ == "__main__":
    # Test data loading pipeline
    print("🧪 Testing Natron Data Loader...")
    
    # Create sample CSV data
    print("\n📝 Creating sample data...")
    np.random.seed(42)
    n_samples = 1000
    dates = pd.date_range('2024-01-01', periods=n_samples, freq='15min')
    
    returns = np.random.randn(n_samples) * 0.01
    prices = 100 * np.exp(np.cumsum(returns))
    
    sample_df = pd.DataFrame({
        'time': dates,
        'open': prices * (1 + np.random.randn(n_samples) * 0.001),
        'high': prices * (1 + abs(np.random.randn(n_samples)) * 0.002),
        'low': prices * (1 - abs(np.random.randn(n_samples)) * 0.002),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    sample_df['high'] = sample_df[['open', 'high', 'close']].max(axis=1)
    sample_df['low'] = sample_df[['open', 'low', 'close']].min(axis=1)
    
    # Save to CSV
    sample_csv = "/workspace/data/sample_data.csv"
    sample_df.to_csv(sample_csv, index=False)
    print(f"  Saved to {sample_csv}")
    
    # Test data loader
    print("\n🔄 Testing data loading pipeline...")
    loader = NatronDataLoader("/workspace/config.yaml")
    dataloaders = loader.load_and_prepare(sample_csv)
    
    # Test one batch
    print("\n🧪 Testing batch retrieval...")
    train_loader = dataloaders['train']
    batch = next(iter(train_loader))
    
    sequences, buy, sell, direction, regime = batch
    print(f"  Sequence shape: {sequences.shape}")
    print(f"  Buy shape: {buy.shape}")
    print(f"  Sell shape: {sell.shape}")
    print(f"  Direction shape: {direction.shape}")
    print(f"  Regime shape: {regime.shape}")
    
    print("\n✅ Data loader test successful!")
