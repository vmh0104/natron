"""
Natron V2 Dataset Loader and Sequence Creator
Handles data loading, preprocessing, and sequence generation
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict
import pickle


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for sequential OHLCV data with multiple labels.
    """
    
    def __init__(self, sequences: np.ndarray, labels: Dict[str, np.ndarray], mode: str = 'train'):
        """
        Args:
            sequences: numpy array of shape (N, seq_len, features)
            labels: dict with keys ['buy', 'sell', 'direction', 'regime']
            mode: 'train', 'val', or 'test'
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = {
            'buy': torch.FloatTensor(labels['buy']),
            'sell': torch.FloatTensor(labels['sell']),
            'direction': torch.LongTensor(labels['direction']),
            'regime': torch.LongTensor(labels['regime'])
        }
        self.mode = mode
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        labels = {
            'buy': self.labels['buy'][idx],
            'sell': self.labels['sell'][idx],
            'direction': self.labels['direction'][idx],
            'regime': self.labels['regime'][idx]
        }
        return sequence, labels


class PretrainDataset(Dataset):
    """
    Dataset for unsupervised pretraining (masked modeling, contrastive learning)
    """
    
    def __init__(self, sequences: np.ndarray, mask_ratio: float = 0.15):
        """
        Args:
            sequences: numpy array of shape (N, seq_len, features)
            mask_ratio: ratio of sequence to mask for reconstruction
        """
        self.sequences = torch.FloatTensor(sequences)
        self.mask_ratio = mask_ratio
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx].clone()
        
        # Create masked version for reconstruction
        seq_len = sequence.shape[0]
        mask_length = int(seq_len * self.mask_ratio)
        mask_indices = torch.randperm(seq_len)[:mask_length]
        
        masked_sequence = sequence.clone()
        masked_sequence[mask_indices] = 0  # Mask with zeros
        
        # Create mask indicator
        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[mask_indices] = True
        
        return {
            'original': sequence,
            'masked': masked_sequence,
            'mask': mask,
            'idx': idx
        }


class DataProcessor:
    """
    Main data processing pipeline for Natron V2.
    Handles loading, feature extraction, labeling, and sequence creation.
    """
    
    def __init__(self, config: dict):
        self.config = config
        self.sequence_length = config['data']['sequence_length']
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def load_and_process(self, csv_path: str) -> Tuple[Dict, Dict, Dict]:
        """
        Complete pipeline: load CSV -> features -> labels -> sequences
        
        Returns:
            train_data, val_data, test_data (each is a dict with sequences and labels)
        """
        from src.features.feature_engine import FeatureEngine
        from src.labels.label_generator import LabelGenerator
        
        print("Loading data...")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} candles")
        
        # Generate features
        print("Generating features...")
        feature_engine = FeatureEngine()
        features_df = feature_engine.generate_features(df)
        self.feature_names = feature_engine.get_feature_names()
        print(f"Generated {len(self.feature_names)} features")
        
        # Generate labels
        print("Generating labels...")
        label_generator = LabelGenerator()
        buy_labels, sell_labels, direction_labels, regime_labels = label_generator.generate_labels(features_df)
        
        # Print regime distribution
        regime_dist = label_generator.get_regime_distribution(regime_labels)
        print("\nRegime Distribution:")
        for regime_name, stats in regime_dist.items():
            print(f"  {regime_name}: {stats['count']} ({stats['percentage']:.2f}%)")
        
        # Extract feature matrix
        feature_matrix = features_df[self.feature_names].values
        
        # Normalize features
        print("\nNormalizing features...")
        feature_matrix = self.scaler.fit_transform(feature_matrix)
        
        # Create sequences
        print("Creating sequences...")
        sequences, labels_dict = self._create_sequences(
            feature_matrix,
            buy_labels.values,
            sell_labels.values,
            direction_labels.values,
            regime_labels.values
        )
        
        print(f"Created {len(sequences)} sequences of shape {sequences.shape}")
        
        # Split data
        train_data, val_data, test_data = self._split_data(sequences, labels_dict)
        
        print(f"\nData split:")
        print(f"  Train: {len(train_data['sequences'])} sequences")
        print(f"  Val: {len(val_data['sequences'])} sequences")
        print(f"  Test: {len(test_data['sequences'])} sequences")
        
        return train_data, val_data, test_data
    
    def _create_sequences(self, features: np.ndarray, buy: np.ndarray, sell: np.ndarray,
                         direction: np.ndarray, regime: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Create overlapping sequences of length seq_len.
        Each sequence predicts the label at the last time step.
        """
        seq_len = self.sequence_length
        num_sequences = len(features) - seq_len
        
        sequences = np.zeros((num_sequences, seq_len, features.shape[1]))
        labels = {
            'buy': np.zeros(num_sequences),
            'sell': np.zeros(num_sequences),
            'direction': np.zeros(num_sequences),
            'regime': np.zeros(num_sequences)
        }
        
        for i in range(num_sequences):
            sequences[i] = features[i:i + seq_len]
            labels['buy'][i] = buy[i + seq_len - 1]
            labels['sell'][i] = sell[i + seq_len - 1]
            labels['direction'][i] = direction[i + seq_len - 1]
            labels['regime'][i] = regime[i + seq_len - 1]
        
        return sequences, labels
    
    def _split_data(self, sequences: np.ndarray, labels: Dict) -> Tuple[Dict, Dict, Dict]:
        """
        Split data into train/val/test sets (chronological split for time series).
        """
        n = len(sequences)
        train_size = int(n * self.config['data']['train_split'])
        val_size = int(n * self.config['data']['val_split'])
        
        train_idx = train_size
        val_idx = train_size + val_size
        
        train_data = {
            'sequences': sequences[:train_idx],
            'labels': {k: v[:train_idx] for k, v in labels.items()}
        }
        
        val_data = {
            'sequences': sequences[train_idx:val_idx],
            'labels': {k: v[train_idx:val_idx] for k, v in labels.items()}
        }
        
        test_data = {
            'sequences': sequences[val_idx:],
            'labels': {k: v[val_idx:] for k, v in labels.items()}
        }
        
        return train_data, val_data, test_data
    
    def create_dataloaders(self, train_data: Dict, val_data: Dict, test_data: Dict,
                          batch_size: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for train/val/test.
        """
        train_dataset = SequenceDataset(train_data['sequences'], train_data['labels'], 'train')
        val_dataset = SequenceDataset(val_data['sequences'], val_data['labels'], 'val')
        test_dataset = SequenceDataset(test_data['sequences'], test_data['labels'], 'test')
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, val_loader, test_loader
    
    def create_pretrain_dataloader(self, train_data: Dict, batch_size: int,
                                   mask_ratio: float = 0.15) -> DataLoader:
        """
        Create DataLoader for pretraining phase.
        """
        pretrain_dataset = PretrainDataset(train_data['sequences'], mask_ratio)
        pretrain_loader = DataLoader(pretrain_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=2)
        return pretrain_loader
    
    def save_scaler(self, path: str):
        """Save the fitted scaler for inference"""
        with open(path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to {path}")
    
    def load_scaler(self, path: str):
        """Load a fitted scaler"""
        with open(path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Scaler loaded from {path}")
    
    def preprocess_inference_data(self, ohlcv_sequence: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single 96-candle OHLCV sequence for inference.
        
        Args:
            ohlcv_sequence: numpy array of shape (96, 6) [time, open, high, low, close, volume]
            
        Returns:
            torch tensor of shape (1, 96, num_features)
        """
        from src.features.feature_engine import FeatureEngine
        
        # Create DataFrame
        df = pd.DataFrame(ohlcv_sequence, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        
        # Generate features
        feature_engine = FeatureEngine()
        features_df = feature_engine.generate_features(df)
        
        # Extract feature matrix
        if self.feature_names is None:
            self.feature_names = feature_engine.get_feature_names()
        
        feature_matrix = features_df[self.feature_names].values
        
        # Normalize
        feature_matrix = self.scaler.transform(feature_matrix)
        
        # Convert to tensor (1, seq_len, features)
        tensor = torch.FloatTensor(feature_matrix).unsqueeze(0)
        
        return tensor
