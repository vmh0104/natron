"""
Loss functions for Natron training phases.

- Masked modeling reconstruction loss
- NT-Xent contrastive loss
- Multi-task supervised loss with configurable weights
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class LossWeights:
    """Weighting factors for supervised multi-task loss."""

    buy: float = 1.0
    sell: float = 1.0
    direction: float = 1.5
    regime: float = 2.0
    entropy_regularization: float = 0.01


class MaskedModelingLoss(nn.Module):
    """Smooth-L1 reconstruction loss with optional cosine similarity term."""

    def __init__(self, alpha: float = 0.7) -> None:
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.SmoothL1Loss(reduction="none")

    def forward(
        self,
        reconstruction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        reconstruction:
            Predicted features (batch, seq_len, feature_dim)
        target:
            Ground-truth features (batch, seq_len, feature_dim)
        mask:
            Boolean mask of positions that were masked (batch, seq_len)
        """
        mask = mask.unsqueeze(-1).expand_as(target)
        masked_recon = reconstruction[mask]
        masked_target = target[mask]

        if masked_recon.numel() == 0:
            return torch.zeros((), device=reconstruction.device)

        l1_loss = self.l1(masked_recon, masked_target).mean()
        cosine_loss = 1.0 - F.cosine_similarity(masked_recon, masked_target, dim=-1).mean()
        return self.alpha * l1_loss + (1 - self.alpha) * cosine_loss


class NTXentContrastiveLoss(nn.Module):
    """
    Normalized temperature-scaled cross entropy loss (SimCLR-style).
    """

    def __init__(self, temperature: float = 0.2) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, zi: torch.Tensor, zj: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        zi, zj:
            Normalized projection vectors of shape (batch, projection_dim)
        """
        batch_size = zi.size(0)
        if batch_size < 2:
            raise ValueError("Contrastive loss requires batch size >= 2.")

        representations = torch.cat([zi, zj], dim=0)
        similarity = F.cosine_similarity(
            representations.unsqueeze(1),
            representations.unsqueeze(0),
            dim=-1,
        )

        logits = similarity / self.temperature
        labels = torch.arange(batch_size, device=zi.device)
        labels = torch.cat([labels + batch_size, labels])

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=zi.device)
        logits = logits.masked_fill(mask, float("-inf"))

        return F.cross_entropy(logits, labels)


class MultiTaskLoss(nn.Module):
    """Aggregate loss over buy/sell/direction/regime outputs."""

    def __init__(self, weights: LossWeights) -> None:
        super().__init__()
        self.weights = weights
        self.bce = nn.BCEWithLogitsLoss()
        self.ce_direction = nn.CrossEntropyLoss()
        self.ce_regime = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        buy_loss = self.bce(outputs["buy_logits"], targets["buy"])
        sell_loss = self.bce(outputs["sell_logits"], targets["sell"])
        direction_loss = self.ce_direction(outputs["direction_logits"], targets["direction"])
        regime_loss = self.ce_regime(outputs["regime_logits"], targets["regime"])

        total = (
            self.weights.buy * buy_loss
            + self.weights.sell * sell_loss
            + self.weights.direction * direction_loss
            + self.weights.regime * regime_loss
        )

        if self.weights.entropy_regularization > 0:
            direction_probs = F.softmax(outputs["direction_logits"], dim=-1)
            regime_probs = F.softmax(outputs["regime_logits"], dim=-1)
            direction_entropy = -(direction_probs * torch.log(direction_probs + 1e-9)).sum(dim=-1).mean()
            regime_entropy = -(regime_probs * torch.log(regime_probs + 1e-9)).sum(dim=-1).mean()
            entropy_penalty = direction_entropy + regime_entropy
            total = total + self.weights.entropy_regularization * (-entropy_penalty)

        return {
            "total_loss": total,
            "buy_loss": buy_loss.detach(),
            "sell_loss": sell_loss.detach(),
            "direction_loss": direction_loss.detach(),
            "regime_loss": regime_loss.detach(),
        }
