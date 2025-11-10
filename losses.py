"""
Loss utilities for Natron Transformer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    return F.mse_loss(pred, target)


def info_nce_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Compute symmetric InfoNCE loss for two augmented batches.
    """
    if z_i.shape != z_j.shape:
        raise ValueError("Latent batches must share shape.")

    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)
    batch_size = z_i.shape[0]

    representations = torch.cat([z_i, z_j], dim=0)
    similarity_matrix = F.cosine_similarity(
        representations.unsqueeze(1), representations.unsqueeze(0), dim=-1
    )

    labels = torch.arange(batch_size, device=z_i.device)
    labels = torch.cat([labels + batch_size, labels], dim=0)

    mask = torch.eye(2 * batch_size, device=z_i.device).bool()
    logits = similarity_matrix / temperature
    logits = logits.masked_fill(mask, float("-inf"))

    loss = F.cross_entropy(logits, labels)
    return loss


@dataclass
class LossWeights:
    buy: float = 0.8
    sell: float = 0.8
    direction: float = 1.0
    regime: float = 1.2
    reconstruction: float = 1.0
    contrastive: float = 1.0


@dataclass
class LossConfig:
    weights: LossWeights = field(default_factory=LossWeights)
    focal_gamma: float = 2.0
    focal_alpha: float = 0.25
    regime_class_weights: Optional[Tuple[float, ...]] = None
    label_smoothing: float = 0.0


class MultiTaskLoss(nn.Module):
    """Aggregate multi-task losses with configurable weights and focal modulation."""

    def __init__(self, config: Optional[LossConfig] = None):
        super().__init__()
        self.config = config or LossConfig()
        if self.config.regime_class_weights:
            weights_tensor = torch.tensor(self.config.regime_class_weights, dtype=torch.float32)
        else:
            weights_tensor = None
        self.register_buffer("regime_weights", weights_tensor, persistent=False)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses: Dict[str, torch.Tensor] = {}
        weights = self.config.weights

        # Buy / Sell with focal BCE
        losses["buy_loss"] = self._focal_bce(outputs["buy_logits"], targets["buy"])
        losses["sell_loss"] = self._focal_bce(outputs["sell_logits"], targets["sell"])

        # Direction cross-entropy
        losses["direction_loss"] = F.cross_entropy(
            outputs["direction_logits"], targets["direction"]
        )

        # Regime cross-entropy with optional class weights
        weight_tensor = self.regime_weights
        if weight_tensor is not None:
            weight_tensor = weight_tensor.to(outputs["regime_logits"].device)
        losses["regime_loss"] = F.cross_entropy(
            outputs["regime_logits"], targets["regime"], weight=weight_tensor
        )

        total = (
            weights.buy * losses["buy_loss"]
            + weights.sell * losses["sell_loss"]
            + weights.direction * losses["direction_loss"]
            + weights.regime * losses["regime_loss"]
        )
        losses["total_supervised"] = total
        return losses

    def _focal_bce(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_factor = (1 - pt) ** self.config.focal_gamma
        if self.config.focal_alpha is not None:
            alpha_t = self.config.focal_alpha * targets + (1 - self.config.focal_alpha) * (1 - targets)
            focal_factor = focal_factor * alpha_t
        loss = (focal_factor * bce).mean()
        return loss


__all__ = ["masked_mse_loss", "info_nce_loss", "MultiTaskLoss", "LossConfig", "LossWeights"]
