"""
Unsupervised view augmentations for Natron V2.
"""

from __future__ import annotations

import torch


def augment_batch(
    batch: torch.Tensor,
    dropout_prob: float = 0.1,
    jitter_std: float = 0.01,
    scaling_std: float = 0.05,
) -> torch.Tensor:
    """
    Apply stochastic augmentations: feature dropout, additive jitter, and scaling.
    """
    noise = torch.randn_like(batch) * jitter_std
    dropout_mask = (torch.rand_like(batch) > dropout_prob).float()
    scaling = torch.randn(batch.size(0), 1, 1, device=batch.device) * scaling_std + 1.0
    augmented = batch * dropout_mask * scaling + noise
    return augmented
