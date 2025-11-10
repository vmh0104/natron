"""
Metrics helpers for Natron V2.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn import metrics
import torch


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    return tensor


def compute_multitask_metrics(outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compute ROC-AUC/F1 for buy/sell and accuracy for direction/regime."""
    buy_probs = torch.sigmoid(outputs["buy_logits"]).squeeze(-1)
    sell_probs = torch.sigmoid(outputs["sell_logits"]).squeeze(-1)
    direction_logits = outputs["direction_logits"]
    regime_logits = outputs["regime_logits"]

    buy_targets = targets["buy"].float()
    sell_targets = targets["sell"].float()
    direction_targets = targets["direction"].long()
    regime_targets = targets["regime"].long()

    metrics_dict = {}

    for prefix, probs, y_true in [
        ("buy", buy_probs, buy_targets),
        ("sell", sell_probs, sell_targets),
    ]:
        metrics_dict[f"{prefix}_auc"] = _safe_auc(y_true, probs)
        preds = (probs > 0.5).float()
        metrics_dict[f"{prefix}_f1"] = _safe_f1(y_true, preds)

    direction_pred = torch.argmax(direction_logits, dim=-1)
    regime_pred = torch.argmax(regime_logits, dim=-1)
    metrics_dict["direction_acc"] = (direction_pred == direction_targets).float().mean().item()
    metrics_dict["regime_acc"] = (regime_pred == regime_targets).float().mean().item()
    return metrics_dict


def _safe_auc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    y_true_np = _to_numpy(y_true)
    y_score_np = _to_numpy(y_score)
    if len(np.unique(y_true_np)) < 2:
        return 0.5
    return float(metrics.roc_auc_score(y_true_np, y_score_np))


def _safe_f1(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    y_true_np = _to_numpy(y_true)
    y_pred_np = _to_numpy(y_pred)
    return float(metrics.f1_score(y_true_np, y_pred_np, zero_division=0))
