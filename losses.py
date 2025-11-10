"""
Natron loss functions for pretraining and multi-task supervised learning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    masked: float = 0.6
    contrastive: float = 0.4
    buy: float = 1.0
    sell: float = 1.0
    direction: float = 0.8
    regime: float = 1.2


def masked_reconstruction_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Args:
        reconstruction: (batch, seq, features) predicted tokens
        target: (batch, seq, features) ground truth tokens
        mask: (batch, seq) boolean tensor of masked positions (True=masked)
    """
    mask = mask.float().unsqueeze(-1)
    diff = (reconstruction - target) ** 2 * mask
    denom = mask.sum().clamp(min=1.0)
    return diff.sum() / denom


def info_nce_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    Computes the NT-Xent / InfoNCE loss between two batches of projections.
    """
    z_i = F.normalize(z_i, dim=-1)
    z_j = F.normalize(z_j, dim=-1)

    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = torch.matmul(representations, representations.T)
    batch_size = z_i.size(0)

    labels = torch.arange(batch_size, device=z_i.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    mask = torch.eye(2 * batch_size, device=z_i.device, dtype=torch.bool)
    similarity_matrix = similarity_matrix / temperature
    similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))

    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


def multi_task_loss(
    outputs: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    weights: LossWeights,
) -> Dict[str, torch.Tensor]:
    buy_loss = F.binary_cross_entropy_with_logits(outputs["buy_logits"], targets["buy"])
    sell_loss = F.binary_cross_entropy_with_logits(outputs["sell_logits"], targets["sell"])
    direction_loss = F.cross_entropy(outputs["direction_logits"], targets["direction"])
    regime_loss = F.cross_entropy(outputs["regime_logits"], targets["regime"])

    total = (
        weights.buy * buy_loss
        + weights.sell * sell_loss
        + weights.direction * direction_loss
        + weights.regime * regime_loss
    )

    return {
        "total": total,
        "buy": buy_loss.detach(),
        "sell": sell_loss.detach(),
        "direction": direction_loss.detach(),
        "regime": regime_loss.detach(),
    }


def aggregate_losses(loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    return sum(loss_dict.values())


__all__ = [
    "LossWeights",
    "masked_reconstruction_loss",
    "info_nce_loss",
    "multi_task_loss",
    "aggregate_losses",
]
