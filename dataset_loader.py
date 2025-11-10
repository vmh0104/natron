"""
Natron dataset pipeline: CSV ingestion → feature engineering → label generation → sequence datasets.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from pandas.api.types import is_datetime64_any_dtype
from torch.utils.data import DataLoader, Dataset

from feature_engineering import FeatureConfig, FeatureEngineer
from labeling import LabelConfig, LabelGenerator

try:
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "scikit-learn is required for scaling. Install via `pip install scikit-learn`."
    ) from exc

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    csv_path: str = "data/data_export.csv"
    cache_dir: str = "data/cache"
    sequence_length: int = 96
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    normalize: bool = True
    scaler: str = "robust"  # options: robust, standard, minmax, none
    num_workers: int = 8
    batch_size: int = 64
    pin_memory: bool = True
    drop_last: bool = False
    augment: bool = True

    def validate(self) -> None:
        total = self.train_split + self.val_split + self.test_split
        if not np.isclose(total, 1.0):
            raise ValueError("Train/val/test splits must sum to 1.0")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be >= 2")


class NatronSequenceDataset(Dataset):
    """Sliding-window sequence dataset with multi-task labels."""

    def __init__(
        self,
        features: np.ndarray,
        labels: DataFrame,
        timestamps: pd.Series,
        sequence_length: int,
        stage: str = "train",
    ):
        if len(features) != len(labels):
            raise ValueError("Features and labels must share the same number of rows.")
        if len(features) < sequence_length:
            raise ValueError("Not enough rows to create sequences.")

        self.sequence_length = sequence_length
        self.stage = stage
        self.features = features.astype(np.float32)
        self.labels = labels.reset_index(drop=True)
        self.timestamps = timestamps.reset_index(drop=True)
        self.size = len(self.features) - self.sequence_length + 1

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= self.size:
            raise IndexError(idx)
        start = idx
        end = idx + self.sequence_length
        seq = self.features[start:end]
        label_idx = end - 1

        sample = {
            "inputs": torch.from_numpy(seq),
            "buy": torch.tensor(self.labels.loc[label_idx, "buy"], dtype=torch.float32),
            "sell": torch.tensor(self.labels.loc[label_idx, "sell"], dtype=torch.float32),
            "direction": torch.tensor(self.labels.loc[label_idx, "direction"], dtype=torch.long),
            "regime": torch.tensor(self.labels.loc[label_idx, "regime"], dtype=torch.long),
            "timestamp": torch.tensor(self._timestamp_to_float(label_idx), dtype=torch.float64),
        }
        return sample

    def _timestamp_to_float(self, idx: int) -> float:
        ts = self.timestamps.iloc[idx]
        if pd.isna(ts):
            return float("nan")
        if isinstance(ts, (np.datetime64, pd.Timestamp)):
            return pd.Timestamp(ts).value / 1e9  # seconds
        return float(ts)


class NatronDataModule:
    """High-level orchestrator for preparing Natron datasets."""

    def __init__(
        self,
        data_config: Optional[DataConfig] = None,
        feature_config: Optional[FeatureConfig] = None,
        label_config: Optional[LabelConfig] = None,
    ):
        self.config = data_config or DataConfig()
        self.feature_engineer = FeatureEngineer(feature_config or FeatureConfig())
        self.label_generator = LabelGenerator(label_config or LabelConfig())
        self.scaler = None
        self.datasets: Dict[str, NatronSequenceDataset] = {}
        self.feature_columns: List[str] = []
        self.label_columns: List[str] = []

    # -- Public API --
    def prepare(self, force: bool = False) -> None:
        """Load data, engineer features, create labels, fit scaler, and split datasets."""
        self.config.validate()
        cache_path = self._cache_path("prepared_dataset.npz")

        if cache_path.exists() and not force:
            logger.info("Loading cached dataset artifacts from %s", cache_path)
            self._load_cache(cache_path)
            return

        df = self._load_csv(self.config.csv_path)
        features = self.feature_engineer.transform(df)
        labels = self.label_generator.generate(df, features)

        common_index = features.index.intersection(labels.index)
        features = features.loc[common_index]
        labels = labels.loc[common_index]
        timestamps = df.loc[common_index, "time"] if "time" in df.columns else pd.Series(common_index)
        self.feature_columns = features.columns.tolist()
        self.label_columns = labels.columns.tolist()

        train_features, val_features, test_features, train_labels, val_labels, test_labels, train_times, val_times, test_times = self._split(
            features, labels, timestamps
        )

        if self.config.normalize:
            self.scaler = self._init_scaler(self.config.scaler)
            train_arr = self.scaler.fit_transform(train_features)
            val_arr = self.scaler.transform(val_features)
            test_arr = self.scaler.transform(test_features)
        else:
            self.scaler = None
            train_arr, val_arr, test_arr = train_features.values, val_features.values, test_features.values

        self.datasets["train"] = NatronSequenceDataset(
            train_arr, train_labels, train_times, self.config.sequence_length, stage="train"
        )
        self.datasets["val"] = NatronSequenceDataset(
            val_arr, val_labels, val_times, self.config.sequence_length, stage="val"
        )
        self.datasets["test"] = NatronSequenceDataset(
            test_arr, test_labels, test_times, self.config.sequence_length, stage="test"
        )

        self._save_cache(
            cache_path,
            {
                "train_features": train_arr,
                "val_features": val_arr,
                "test_features": test_arr,
                "train_labels": train_labels,
                "val_labels": val_labels,
                "test_labels": test_labels,
                "train_times": train_times,
                "val_times": val_times,
                "test_times": test_times,
                "scaler": self._serialize_scaler(self.scaler),
                "config": asdict(self.config),
            },
        )

    def dataloaders(self) -> Dict[str, DataLoader]:
        if not self.datasets:
            raise RuntimeError("Datasets not prepared. Call `prepare()` first.")
        cfg = self.config
        return {
            split: DataLoader(
                dataset,
                batch_size=cfg.batch_size,
                shuffle=(split == "train"),
                num_workers=cfg.num_workers,
                pin_memory=cfg.pin_memory,
                drop_last=cfg.drop_last,
            )
            for split, dataset in self.datasets.items()
        }

    # -- Helpers --
    def _load_csv(self, path: str) -> DataFrame:
        csv_path = Path(path)
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        df = pd.read_csv(csv_path)
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
            missing = df["time"].isna().sum()
            if missing:
                logger.warning(
                    "Detected %d rows with non-parsable 'time'; assigning synthetic timestamps.",
                    missing,
                )
                base_time = pd.Timestamp.utcnow().normalize()
                df.loc[df["time"].isna(), "time"] = pd.date_range(
                    start=base_time, periods=missing, freq="min"
                ).tz_localize("UTC")
        else:
            logger.warning("Column 'time' missing in CSV; generating synthetic timeline.")
            base_time = pd.Timestamp.utcnow().normalize()
            df["time"] = pd.date_range(start=base_time, periods=len(df), freq="min", tz="UTC")

        if not is_datetime64_any_dtype(df["time"]):
            df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

        if df["time"].isna().any():
            missing = df["time"].isna().sum()
            base_time = pd.Timestamp.utcnow().normalize()
            df.loc[df["time"].isna(), "time"] = pd.date_range(
                start=base_time, periods=missing, freq="min", tz="UTC"
            )

        if getattr(df["time"].dt, "tz", None) is None:
            df["time"] = df["time"].dt.tz_localize("UTC")
        else:
            df["time"] = df["time"].dt.tz_convert("UTC")

        df = df.sort_values("time").reset_index(drop=True)
        return df

    def _split(
        self,
        features: DataFrame,
        labels: DataFrame,
        timestamps: pd.Series,
    ) -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, DataFrame, pd.Series, pd.Series, pd.Series]:
        n = len(features)
        train_end = int(n * self.config.train_split)
        val_end = train_end + int(n * self.config.val_split)

        train_slice = (0, train_end)
        val_slice = (train_end, val_end)
        test_slice = (val_end, n)

        train_slice = self._ensure_min_block(train_slice, n)
        val_slice = self._ensure_min_block(val_slice, n)
        test_slice = self._ensure_min_block(test_slice, n)

        train_features = features.iloc[train_slice[0] : train_slice[1]]
        val_features = features.iloc[val_slice[0] : val_slice[1]]
        test_features = features.iloc[test_slice[0] : test_slice[1]]

        train_labels = labels.iloc[train_slice[0] : train_slice[1]]
        val_labels = labels.iloc[val_slice[0] : val_slice[1]]
        test_labels = labels.iloc[test_slice[0] : test_slice[1]]

        train_times = timestamps.iloc[train_slice[0] : train_slice[1]]
        val_times = timestamps.iloc[val_slice[0] : val_slice[1]]
        test_times = timestamps.iloc[test_slice[0] : test_slice[1]]

        return (
            train_features,
            val_features,
            test_features,
            train_labels,
            val_labels,
            test_labels,
            train_times,
            val_times,
            test_times,
        )

    def _init_scaler(self, scaler_type: str):
        scaler_type = scaler_type.lower()
        if scaler_type == "standard":
            return StandardScaler()
        if scaler_type == "minmax":
            return MinMaxScaler()
        if scaler_type == "robust":
            return RobustScaler()
        if scaler_type in {"none", "identity"}:
            return None
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    def _cache_path(self, filename: str) -> Path:
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / filename

    def _save_cache(self, path: Path, payload: Dict) -> None:
        scaler_state = json.dumps(payload["scaler"]) if payload["scaler"] is not None else ""

        np.savez_compressed(
            path,
            train_features=payload["train_features"],
            val_features=payload["val_features"],
            test_features=payload["test_features"],
            train_labels=payload["train_labels"].to_dict(orient="list"),
            val_labels=payload["val_labels"].to_dict(orient="list"),
            test_labels=payload["test_labels"].to_dict(orient="list"),
            train_times=self._times_to_int(payload["train_times"]),
            val_times=self._times_to_int(payload["val_times"]),
            test_times=self._times_to_int(payload["test_times"]),
            scaler=scaler_state,
            config=json.dumps(payload["config"]),
        )
        logger.info("Saved dataset cache to %s", path)

    def _load_cache(self, path: Path) -> None:
        archive = np.load(path, allow_pickle=True)
        train_features = archive["train_features"]
        val_features = archive["val_features"]
        test_features = archive["test_features"]
        train_labels = pd.DataFrame(archive["train_labels"].item())
        val_labels = pd.DataFrame(archive["val_labels"].item())
        test_labels = pd.DataFrame(archive["test_labels"].item())

        train_times = pd.to_datetime(archive["train_times"], unit="ns")
        val_times = pd.to_datetime(archive["val_times"], unit="ns")
        test_times = pd.to_datetime(archive["test_times"], unit="ns")

        scaler_blob = archive["scaler"].item()
        if scaler_blob:
            scaler_state = json.loads(scaler_blob)
            self.scaler = self._init_scaler(self.config.scaler)
            self.scaler.__dict__.update(scaler_state)
        else:
            self.scaler = None

        self.datasets["train"] = NatronSequenceDataset(
            train_features, train_labels, train_times, self.config.sequence_length, stage="train"
        )
        self.datasets["val"] = NatronSequenceDataset(
            val_features, val_labels, val_times, self.config.sequence_length, stage="val"
        )
        self.datasets["test"] = NatronSequenceDataset(
            test_features, test_labels, test_times, self.config.sequence_length, stage="test"
        )

    def _serialize_scaler(self, scaler) -> Optional[Dict]:
        if scaler is None:
            return None
        state = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in scaler.__dict__.items()
            if not k.startswith("_")
        }
        return state

    def _times_to_int(self, series: pd.Series) -> np.ndarray:
        if series.empty:
            return np.array([], dtype="int64")
        ts = pd.to_datetime(series)
        return ts.view("int64")

    def _ensure_min_block(self, span: Tuple[int, int], total: int) -> Tuple[int, int]:
        start, end = span
        end = max(end, start)
        if end - start >= self.config.sequence_length:
            return start, end
        needed = self.config.sequence_length - (end - start)
        start = max(0, start - needed)
        end = min(total, max(end, start + self.config.sequence_length))
        return start, end


__all__ = ["NatronDataModule", "NatronSequenceDataset", "DataConfig"]
