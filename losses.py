"""
Loss functions for Natron V2 multi-stage training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    """Mean-squared error computed only on masked tokens."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == 2:
            mask = mask.unsqueeze(-1)
        masked_pred = pred[mask]
        masked_target = target[mask]
        loss = F.mse_loss(masked_pred, masked_target, reduction="mean")
        return loss


class InfoNCELoss(nn.Module):
    """Standard InfoNCE contrastive objective."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        z_i = F.normalize(z_i, dim=-1)
        z_j = F.normalize(z_j, dim=-1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity = torch.matmul(representations, representations.T) / self.temperature

        batch_size = z_i.size(0)
        labels = torch.arange(batch_size, device=z_i.device)
        labels = torch.cat([labels + batch_size, labels])

        mask = torch.eye(2 * batch_size, device=z_i.device).bool()
        similarity = similarity.masked_fill(mask, float("-inf"))

        loss = F.cross_entropy(similarity, labels)
        return loss


@dataclass
class MultiTaskLossOutput:
    total: torch.Tensor
    buy: torch.Tensor
    sell: torch.Tensor
    direction: torch.Tensor
    regime: torch.Tensor


class MultiTaskLoss(nn.Module):
    """Weighted combination of task-specific losses."""

    def __init__(self, weights: Dict[str, float]) -> None:
        super().__init__()
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss()
        self.ce_direction = nn.CrossEntropyLoss()
        self.ce_regime = nn.CrossEntropyLoss()

    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> MultiTaskLossOutput:
        buy_loss = self.bce(outputs["buy_logits"].squeeze(-1), targets["buy"])
        sell_loss = self.bce(outputs["sell_logits"].squeeze(-1), targets["sell"])
        direction_loss = self.ce_direction(outputs["direction_logits"], targets["direction"])
        regime_loss = self.ce_regime(outputs["regime_logits"], targets["regime"])

        total = (
            self.weights.get("buy", 1.0) * buy_loss
            + self.weights.get("sell", 1.0) * sell_loss
            + self.weights.get("direction", 1.0) * direction_loss
            + self.weights.get("regime", 1.0) * regime_loss
        )

        return MultiTaskLossOutput(total=total, buy=buy_loss, sell=sell_loss, direction=direction_loss, regime=regime_loss)
