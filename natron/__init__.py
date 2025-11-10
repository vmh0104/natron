"""
Natron package initialization.

Exposes high-level helpers for loading datasets, models, and training utilities.
"""

from .dataset_loader import NatronDatasetConfig, NatronFeatureEngineer, NatronSequenceDataset
from .model_natron import NatronTransformer, NatronInferenceWrapper, NatronModelConfig
from .losses import (
    MaskedModelingLoss,
    NTXentContrastiveLoss,
    MultiTaskLoss,
    LossWeights,
)
from .rl import MarketEnvironment, MarketEnvConfig, PPOAgent, PPOConfig

__all__ = [
    "NatronDatasetConfig",
    "NatronFeatureEngineer",
    "NatronSequenceDataset",
    "NatronTransformer",
    "NatronInferenceWrapper",
    "NatronModelConfig",
    "MaskedModelingLoss",
    "NTXentContrastiveLoss",
    "MultiTaskLoss",
    "LossWeights",
    "MarketEnvironment",
    "MarketEnvConfig",
    "PPOAgent",
    "PPOConfig",
]
