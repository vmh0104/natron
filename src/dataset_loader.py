"""
Dataset Loader - Sequence Construction for Natron Transformer
Creates 96-candle sequences with multi-task labels

Input: (N, 100) features
Output: (N-96, 96, 100) sequences with labels
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional
from sklearn.preprocessing import StandardScaler
import pickle


class TradingSequenceDataset(Dataset):
    """PyTorch Dataset for trading sequences"""
    
    def __init__(
        self,
        features: np.ndarray,
        labels: Dict[str, np.ndarray],
        sequence_length: int = 96,
        transform: Optional[StandardScaler] = None
    ):
        """
        Initialize dataset
        
        Args:
            features: Array of shape (N, num_features)
            labels: Dict with keys: 'buy', 'sell', 'direction', 'regime'
            sequence_length: Length of input sequences
            transform: StandardScaler for normalization
        """
        self.features = features
        self.labels = labels
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Calculate valid sequence indices
        self.num_sequences = len(features) - sequence_length
        
        if self.num_sequences <= 0:
            raise ValueError(f"Not enough data for sequences. Need at least {sequence_length} samples.")
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Get a single sequence and its labels"""
        # Extract sequence (96 consecutive candles)
        sequence = self.features[idx:idx + self.sequence_length].copy()
        
        # Apply normalization if available
        if self.transform is not None:
            sequence = self.transform.transform(sequence)
        
        # Get label at the end of sequence
        label_idx = idx + self.sequence_length - 1
        
        labels_dict = {
            'buy': torch.tensor(self.labels['buy'][label_idx], dtype=torch.float32),
            'sell': torch.tensor(self.labels['sell'][label_idx], dtype=torch.float32),
            'direction': torch.tensor(self.labels['direction'][label_idx], dtype=torch.long),
            'regime': torch.tensor(self.labels['regime'][label_idx], dtype=torch.long)
        }
        
        # Convert sequence to tensor
        sequence_tensor = torch.tensor(sequence, dtype=torch.float32)
        
        return sequence_tensor, labels_dict


class DatasetManager:
    """Manage dataset creation, splitting, and loading"""
    
    def __init__(self, config: Dict):
        """
        Initialize dataset manager
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.scaler = None
        self.feature_names = None
        
    def prepare_dataset(
        self,
        df: pd.DataFrame,
        save_scaler: bool = True,
        scaler_path: str = "models/scaler.pkl"
    ) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Prepare dataset: extract features and labels, fit scaler
        
        Args:
            df: DataFrame with features and labels
            save_scaler: Whether to save the scaler
            scaler_path: Path to save scaler
            
        Returns:
            Tuple of (processed_df, scaler)
        """
        print("📦 Preparing dataset...")
        
        # Identify feature columns (exclude time, OHLCV, and labels)
        exclude_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 
                       'buy', 'sell', 'direction', 'regime']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        
        print(f"✅ Feature columns: {len(feature_cols)}")
        
        # Fit scaler on features
        self.scaler = StandardScaler()
        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])
        
        if save_scaler:
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"✅ Scaler saved to {scaler_path}")
        
        return df, self.scaler
    
    def create_datasets(
        self,
        df: pd.DataFrame,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15
    ) -> Tuple[TradingSequenceDataset, TradingSequenceDataset, TradingSequenceDataset]:
        """
        Create train, validation, and test datasets
        
        Args:
            df: DataFrame with normalized features and labels
            train_split: Proportion for training
            val_split: Proportion for validation
            test_split: Proportion for testing
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        print("\n🔪 Splitting dataset...")
        
        # Get feature and label columns
        exclude_cols = ['time', 'open', 'high', 'low', 'close', 'volume', 
                       'buy', 'sell', 'direction', 'regime']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Extract features and labels
        features = df[feature_cols].values
        labels = {
            'buy': df['buy'].values,
            'sell': df['sell'].values,
            'direction': df['direction'].values,
            'regime': df['regime'].values
        }
        
        # Calculate split indices (time-series split, no shuffle)
        n = len(df)
        train_idx = int(n * train_split)
        val_idx = int(n * (train_split + val_split))
        
        print(f"Total samples: {n}")
        print(f"Train: 0 to {train_idx} ({train_split*100:.0f}%)")
        print(f"Val: {train_idx} to {val_idx} ({val_split*100:.0f}%)")
        print(f"Test: {val_idx} to {n} ({test_split*100:.0f}%)")
        
        # Create datasets
        sequence_length = self.config['data']['sequence_length']
        
        train_dataset = TradingSequenceDataset(
            features[:train_idx],
            {k: v[:train_idx] for k, v in labels.items()},
            sequence_length=sequence_length
        )
        
        val_dataset = TradingSequenceDataset(
            features[train_idx:val_idx],
            {k: v[train_idx:val_idx] for k, v in labels.items()},
            sequence_length=sequence_length
        )
        
        test_dataset = TradingSequenceDataset(
            features[val_idx:],
            {k: v[val_idx:] for k, v in labels.items()},
            sequence_length=sequence_length
        )
        
        print(f"\n✅ Train sequences: {len(train_dataset)}")
        print(f"✅ Val sequences: {len(val_dataset)}")
        print(f"✅ Test sequences: {len(test_dataset)}")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(
        self,
        train_dataset: TradingSequenceDataset,
        val_dataset: TradingSequenceDataset,
        test_dataset: TradingSequenceDataset,
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders
        
        Args:
            train_dataset, val_dataset, test_dataset: Datasets
            batch_size: Batch size
            num_workers: Number of workers for data loading
            pin_memory: Pin memory for faster GPU transfer
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        print("\n🔄 Creating DataLoaders...")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,  # Shuffle for training
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
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
        
        print(f"✅ Train batches: {len(train_loader)}")
        print(f"✅ Val batches: {len(val_loader)}")
        print(f"✅ Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def load_and_prepare_data(
        self,
        csv_path: str,
        feature_engine,
        label_generator
    ) -> Tuple[pd.DataFrame, StandardScaler]:
        """
        Complete pipeline: load CSV -> generate features -> generate labels -> normalize
        
        Args:
            csv_path: Path to data_export.csv
            feature_engine: FeatureEngine instance
            label_generator: LabelGenerator instance
            
        Returns:
            Tuple of (processed_df, scaler)
        """
        print(f"📂 Loading data from {csv_path}...")
        
        # Load raw OHLCV data
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} rows")
        
        # Ensure required columns exist
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Generate features
        df = feature_engine.generate_all_features(df)
        
        # Generate labels
        df = label_generator.generate_labels(df)
        
        # Prepare dataset (normalize features)
        df, scaler = self.prepare_dataset(df)
        
        return df, scaler


def create_inference_sequence(
    recent_data: pd.DataFrame,
    feature_engine,
    scaler: StandardScaler,
    sequence_length: int = 96
) -> torch.Tensor:
    """
    Create a single sequence for inference from recent OHLCV data
    
    Args:
        recent_data: DataFrame with last 96+ rows of OHLCV data
        feature_engine: FeatureEngine instance
        scaler: Fitted StandardScaler
        sequence_length: Sequence length (96)
        
    Returns:
        Tensor of shape (1, 96, num_features)
    """
    if len(recent_data) < sequence_length:
        raise ValueError(f"Need at least {sequence_length} candles, got {len(recent_data)}")
    
    # Generate features
    df = feature_engine.generate_all_features(recent_data)
    
    # Get feature columns
    exclude_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Take last sequence_length rows
    features = df[feature_cols].iloc[-sequence_length:].values
    
    # Normalize
    features_normalized = scaler.transform(features)
    
    # Convert to tensor with batch dimension
    sequence_tensor = torch.tensor(features_normalized, dtype=torch.float32).unsqueeze(0)
    
    return sequence_tensor


if __name__ == "__main__":
    # Test dataset creation
    import yaml
    from feature_engine import FeatureEngine
    from label_generator import LabelGenerator
    
    print("Testing Dataset Loader...")
    
    # Load config
    with open('/workspace/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=2000, freq='15min')
    np.random.seed(42)
    
    sample_df = pd.DataFrame({
        'time': dates,
        'open': 100 + np.random.randn(2000).cumsum() * 0.5,
        'high': 101 + np.random.randn(2000).cumsum() * 0.5,
        'low': 99 + np.random.randn(2000).cumsum() * 0.5,
        'close': 100 + np.random.randn(2000).cumsum() * 0.5,
        'volume': np.random.randint(1000, 10000, 2000)
    })
    
    # Generate features and labels
    feature_engine = FeatureEngine(verbose=False)
    label_generator = LabelGenerator()
    
    df = feature_engine.generate_all_features(sample_df)
    df = label_generator.generate_labels(df, verbose=False)
    
    # Create dataset manager
    dataset_manager = DatasetManager(config)
    df, scaler = dataset_manager.prepare_dataset(df, save_scaler=False)
    
    # Create datasets
    train_ds, val_ds, test_ds = dataset_manager.create_datasets(df)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = dataset_manager.create_dataloaders(
        train_ds, val_ds, test_ds, batch_size=32, num_workers=0
    )
    
    # Test one batch
    sequences, labels = next(iter(train_loader))
    print(f"\n✅ Batch sequences shape: {sequences.shape}")
    print(f"✅ Buy labels shape: {labels['buy'].shape}")
    print(f"✅ Regime labels shape: {labels['regime'].shape}")
