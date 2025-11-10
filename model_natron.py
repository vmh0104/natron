"""
Natron Transformer model definitions.

This module defines the multi-phase Natron architecture composed of
an encoder-only Transformer backbone, multi-task prediction heads,
masked-token reconstruction head, and contrastive projection head.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NatronTransformerConfig:
    input_dim: int = 100
    d_model: int = 256
    n_heads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_positions: int = 256
    pooling: str = "cls"  # or "mean"
    projection_dim: int = 128


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, length: int, device: Optional[torch.device] = None) -> torch.Tensor:
        return self.pe[:, :length].to(device=device)


class NatronEncoder(nn.Module):
    def __init__(self, config: NatronTransformerConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.pos_encoder = SinusoidalPositionalEncoding(config.d_model, max_len=config.max_positions + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            norm_first=True,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Tensor of shape (batch, seq_len, input_dim)
            padding_mask: Optional bool tensor of shape (batch, seq_len) where True indicates padding positions.

        Returns:
            token_embeddings: (batch, seq_len (+1 if cls), d_model)
            pooled_rep: (batch, d_model)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device

        tokens = self.input_proj(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        pos_enc = self.pos_encoder(length=tokens.size(1), device=device)
        tokens = tokens + pos_enc
        tokens = self.layer_norm(tokens)

        if padding_mask is not None:
            padding_mask = F.pad(padding_mask, (1, 0), value=False)

        encoded = self.encoder(tokens, src_key_padding_mask=padding_mask)

        if self.config.pooling == "mean":
            if padding_mask is not None:
                lengths = (~padding_mask).sum(dim=1, keepdim=True).clamp(min=1)
                pooled = (encoded * (~padding_mask).unsqueeze(-1)).sum(dim=1) / lengths
            else:
                pooled = encoded.mean(dim=1)
        else:
            pooled = encoded[:, 0]

        return encoded, pooled


class NatronTaskHeads(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        hidden = max(d_model // 2, 64)
        self.buy_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.sell_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.direction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )
        self.regime_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 6),
        )

    def forward(self, pooled: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "buy_logits": self.buy_head(pooled).squeeze(-1),
            "sell_logits": self.sell_head(pooled).squeeze(-1),
            "direction_logits": self.direction_head(pooled),
            "regime_logits": self.regime_head(pooled),
        }


class MaskedModelingHead(nn.Module):
    def __init__(self, d_model: int, input_dim: int, dropout: float = 0.1):
        super().__init__()
        hidden = max(d_model, 128)
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        # token_embeddings expected without CLS token
        return self.net(token_embeddings)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(in_dim, proj_dim),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        projected = self.mlp(pooled)
        return F.normalize(projected, dim=-1)


class NatronModel(nn.Module):
    """
    Complete Natron architecture with shared encoder and multi-task heads.
    """

    def __init__(self, config: NatronTransformerConfig):
        super().__init__()
        self.config = config
        self.encoder = NatronEncoder(config)
        self.task_heads = NatronTaskHeads(config.d_model, dropout=config.dropout)
        self.masked_head = MaskedModelingHead(config.d_model, config.input_dim, dropout=config.dropout)
        self.projection_head = ProjectionHead(config.d_model, config.projection_dim, dropout=config.dropout)

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        padding_mask: Optional[torch.Tensor] = None,
        return_token_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        encoded, pooled = self.encoder(inputs, padding_mask)
        tokens_wo_cls = encoded[:, 1:, :]
        outputs = self.task_heads(pooled)
        outputs.update(
            {
                "token_embeddings": tokens_wo_cls if return_token_embeddings else None,
                "pooled": pooled,
                "reconstruction": self.masked_head(tokens_wo_cls),
                "projection": self.projection_head(pooled),
            }
        )
        return outputs

    def encode(self, inputs: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        _, pooled = self.encoder(inputs, padding_mask)
        return pooled


def init_model_from_config(config: Dict) -> NatronModel:
    model_cfg = config.get("model", {})
    transformer_config = NatronTransformerConfig(
        input_dim=model_cfg.get("input_dim", 100),
        d_model=model_cfg.get("d_model", 256),
        n_heads=model_cfg.get("n_heads", 8),
        num_layers=model_cfg.get("num_layers", 6),
        dim_feedforward=model_cfg.get("dim_feedforward", 512),
        dropout=model_cfg.get("dropout", 0.1),
        max_positions=model_cfg.get("max_positions", 256),
        pooling=model_cfg.get("pooling", "cls"),
        projection_dim=model_cfg.get("projection_dim", config.get("pretraining", {}).get("projection_dim", 128)),
    )
    return NatronModel(transformer_config)


__all__ = [
    "NatronTransformerConfig",
    "NatronEncoder",
    "NatronModel",
    "init_model_from_config",
]
