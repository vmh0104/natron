"""
Dataset construction utilities for Natron V2.

Creates engineered feature tensors, generates labels, and produces
chronologically split sequence datasets ready for PyTorch training.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.data import DataLoader, Dataset

from feature_engine import FeatureEngineer, FeatureEngineerConfig
from labeling import LabelGenerator, LabelGeneratorConfig


@dataclass
class NatronDataConfig:
    """Data pipeline configuration."""

    csv_path: str = "data/data_export.csv"
    cache_dir: str = "artifacts/cache"
    sequence_length: int = 96
    train_split: float = 0.7
    val_split: float = 0.15
    feature_normalization: str = "robust"  # or "standard" / "none"
    num_workers: int = 8
    batch_size: int = 64
    timezone: str = "UTC"
    dropna: bool = True
    min_history: int = 200


class NatronSequenceDataset(Dataset):
    """PyTorch dataset containing sequence tensors and multi-task labels."""

    def __init__(
        self,
        sequences: np.ndarray,
        labels: Dict[str, np.ndarray],
        timestamps: np.ndarray,
        feature_names: List[str],
    ) -> None:
        self.sequences = sequences.astype(np.float32)
        self.labels = labels
        self.timestamps = timestamps
        self.feature_names = feature_names

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        seq = torch.from_numpy(self.sequences[idx])
        sample = {
            "sequence": seq,
            "buy": torch.tensor(self.labels["buy"][idx], dtype=torch.float32),
            "sell": torch.tensor(self.labels["sell"][idx], dtype=torch.float32),
            "direction": torch.tensor(self.labels["direction"][idx], dtype=torch.long),
            "regime": torch.tensor(self.labels["regime"][idx], dtype=torch.long),
            "direction_score": torch.tensor(self.labels["direction_score"][idx], dtype=torch.float32),
            "timestamp": self.timestamps[idx],
        }
        return sample


class NatronDatasetBuilder:
    """Loads data, creates features/labels, and builds datasets."""

    def __init__(
        self,
        data_config: NatronDataConfig,
        feature_config: Optional[FeatureEngineerConfig] = None,
        label_config: Optional[LabelGeneratorConfig] = None,
    ) -> None:
        self.data_config = data_config
        self.feature_engineer = FeatureEngineer(feature_config or FeatureEngineerConfig())
        self.label_generator = LabelGenerator(label_config or LabelGeneratorConfig())
        self.scaler: Optional[object] = None
        os.makedirs(self.data_config.cache_dir, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def build(self) -> Tuple[NatronSequenceDataset, NatronSequenceDataset, NatronSequenceDataset, Dict[str, np.ndarray]]:
        df = self._load_ohlcv()
        features = self.feature_engineer.transform(df)
        labels = self.label_generator.generate(df, features)

        if self.data_config.dropna:
            valid_mask = ~features.isnull().any(axis=1)
            df, features, labels = df[valid_mask], features[valid_mask], labels[valid_mask]
            df.reset_index(drop=True, inplace=True)
            features.reset_index(drop=True, inplace=True)
            labels.reset_index(drop=True, inplace=True)

        features = self._normalise_features(features)
        sequences, label_dict, timestamps = self._build_sequences(df, features, labels)

        splits = self._chronological_split(len(sequences))
        train_set = NatronSequenceDataset(
            sequences[splits["train"]],
            {k: v[splits["train"]] for k, v in label_dict.items()},
            timestamps[splits["train"]],
            feature_names=list(features.columns),
        )
        val_set = NatronSequenceDataset(
            sequences[splits["val"]],
            {k: v[splits["val"]] for k, v in label_dict.items()},
            timestamps[splits["val"]],
            feature_names=list(features.columns),
        )
        test_set = NatronSequenceDataset(
            sequences[splits["test"]],
            {k: v[splits["test"]] for k, v in label_dict.items()},
            timestamps[splits["test"]],
            feature_names=list(features.columns),
        )

        metadata = {
            "feature_names": np.array(features.columns),
            "scaler_state": self._serialise_scaler(),
            "config": {
                "data": asdict(self.data_config),
            },
        }
        return train_set, val_set, test_set, metadata

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _load_ohlcv(self) -> pd.DataFrame:
        if not os.path.exists(self.data_config.csv_path):
            raise FileNotFoundError(f"CSV file not found at {self.data_config.csv_path}")

        df = pd.read_csv(self.data_config.csv_path)
        if "time" not in df.columns:
            raise ValueError("CSV must contain a 'time' column.")
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.sort_values("time").reset_index(drop=True)
        if len(df) < self.data_config.min_history:
            raise ValueError(
                f"Insufficient data rows ({len(df)}). Need at least {self.data_config.min_history} rows."
            )
        return df

    def _normalise_features(self, features: pd.DataFrame) -> pd.DataFrame:
        mode = self.data_config.feature_normalization.lower()
        if mode == "none":
            return features

        scaler_cls = RobustScaler if mode == "robust" else StandardScaler
        self.scaler = scaler_cls()
        scaled = self.scaler.fit_transform(features.values)
        return pd.DataFrame(scaled, columns=features.columns)

    def _build_sequences(
        self, df: pd.DataFrame, features: pd.DataFrame, labels: pd.DataFrame
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
        seq_len = self.data_config.sequence_length
        feature_array = features.to_numpy(dtype=np.float32)
        label_array = labels.to_numpy()

        sequences: List[np.ndarray] = []
        buy, sell, direction, regime = [], [], [], []
        direction_score = []
        timestamps = []

        for end_idx in range(seq_len - 1, len(features)):
            window = feature_array[end_idx - seq_len + 1 : end_idx + 1]
            sequences.append(window)
            buy.append(label_array[end_idx, 0])
            sell.append(label_array[end_idx, 1])
            direction.append(label_array[end_idx, 2])
            regime.append(label_array[end_idx, 3])
            direction_score.append(label_array[end_idx, 6])
            timestamps.append(df.loc[end_idx, "time"].value)

        sequences_np = np.stack(sequences)
        label_dict = {
            "buy": np.array(buy, dtype=np.float32),
            "sell": np.array(sell, dtype=np.float32),
            "direction": np.array(direction, dtype=np.int64),
            "regime": np.array(regime, dtype=np.int64),
            "direction_score": np.array(direction_score, dtype=np.float32),
        }
        timestamps_np = np.array(timestamps, dtype=np.int64)
        return sequences_np, label_dict, timestamps_np

    def _chronological_split(self, n_samples: int) -> Dict[str, slice]:
        train_end = int(n_samples * self.data_config.train_split)
        val_end = train_end + int(n_samples * self.data_config.val_split)
        val_end = min(val_end, n_samples - 1)
        return {
            "train": slice(0, train_end),
            "val": slice(train_end, val_end),
            "test": slice(val_end, n_samples),
        }

    def _serialise_scaler(self) -> Optional[str]:
        if self.scaler is None:
            return None
        state = {k: getattr(self.scaler, k).tolist() for k in dir(self.scaler) if k.endswith("_") and isinstance(getattr(self.scaler, k), np.ndarray)}
        return json.dumps(state)


def create_dataloaders(
    train_set: NatronSequenceDataset,
    val_set: NatronSequenceDataset,
    test_set: NatronSequenceDataset,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Convenience function returning PyTorch DataLoaders."""

    def collate(batch):
        sequences = torch.stack([item["sequence"] for item in batch])
        buy = torch.stack([item["buy"] for item in batch])
        sell = torch.stack([item["sell"] for item in batch])
        direction = torch.stack([item["direction"] for item in batch])
        regime = torch.stack([item["regime"] for item in batch])
        direction_score = torch.stack([item["direction_score"] for item in batch])
        timestamps = torch.tensor([item["timestamp"] for item in batch], dtype=torch.int64)
        return {
            "sequences": sequences,
            "buy": buy,
            "sell": sell,
            "direction": direction,
            "regime": regime,
            "direction_score": direction_score,
            "timestamp": timestamps,
        }

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    return train_loader, val_loader, test_loader
