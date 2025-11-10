"""
Natron Transformer architecture definitions.

Implements the shared encoder, projection heads, and multi-task prediction
layers used throughout pretraining, supervised learning, and inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(slots=True)
class NatronModelConfig:
    """Lightweight container for Natron Transformer hyperparameters."""

    feature_dim: int = 100
    d_model: int = 256
    num_layers: int = 6
    nhead: int = 8
    dim_feedforward: int = 512
    dropout: float = 0.1
    max_sequence_length: int = 256
    activation: str = "gelu"
    use_cls_token: bool = True


class NatronTransformer(nn.Module):
    """
    Transformer encoder with task-specific heads for Natron multi-task learning.

    Outputs buy/sell logits, direction logits (2 classes), regime logits (6 classes),
    and reconstruction logits for masked modeling.
    """

    def __init__(self, config: NatronModelConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.feature_dim, config.d_model)
        self.input_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model)) if config.use_cls_token else None
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, config.max_sequence_length + (1 if config.use_cls_token else 0), config.d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.post_norm = nn.LayerNorm(config.d_model)

        combined_dim = config.d_model * (2 if config.use_cls_token else 1)
        self.context_projection = nn.Sequential(
            nn.Linear(combined_dim, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
        )

        head_hidden = max(config.d_model // 2, 64)
        self.buy_head = nn.Sequential(
            nn.Linear(config.d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(head_hidden, 1),
        )
        self.sell_head = nn.Sequential(
            nn.Linear(config.d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(head_hidden, 1),
        )
        self.direction_head = nn.Sequential(
            nn.Linear(config.d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(head_hidden, 2),
        )
        self.regime_head = nn.Sequential(
            nn.Linear(config.d_model, head_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(head_hidden, 6),
        )

        self.reconstruction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.feature_dim),
        )
        self.projection_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, 128),
        )

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_projection.weight)
        nn.init.zeros_(self.input_projection.bias)
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings, mean=0.0, std=0.02)

    def forward(
        self,
        sequences: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for multi-task predictions and sequence reconstructions.

        Parameters
        ----------
        sequences:
            Tensor of shape (batch, seq_len, feature_dim)
        src_key_padding_mask:
            Optional boolean mask of shape (batch, seq_len)
        """
        batch, seq_len, _ = sequences.shape
        if seq_len + (1 if self.cls_token is not None else 0) > self.position_embeddings.size(1):
            raise ValueError("Sequence length exceeds configured maximum positional encoding length.")

        x = self.input_projection(sequences)
        x = self.input_norm(x)

        if self.cls_token is not None:
            cls_tokens = self.cls_token.expand(batch, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            pos_emb = self.position_embeddings[:, : seq_len + 1, :]
        else:
            pos_emb = self.position_embeddings[:, :seq_len, :]

        x = x + pos_emb
        x = self.dropout(x)

        encoder_mask = None
        if src_key_padding_mask is not None:
            if self.cls_token is not None:
                pad_mask = torch.zeros(
                    (src_key_padding_mask.size(0), 1),
                    dtype=src_key_padding_mask.dtype,
                    device=src_key_padding_mask.device,
                )
                encoder_mask = torch.cat([pad_mask, src_key_padding_mask], dim=1)
            else:
                encoder_mask = src_key_padding_mask

        x = self.encoder(x, src_key_padding_mask=encoder_mask)
        x = self.post_norm(x)

        if self.cls_token is not None:
            cls_state = x[:, 0]
            sequence_state = x[:, 1:]
            pooled = torch.cat([cls_state, sequence_state.mean(dim=1)], dim=-1)
        else:
            cls_state = sequence_state = x
            pooled = x.mean(dim=1)

        context = self.context_projection(pooled)

        reconstructions = self.reconstruction_head(sequence_state)
        projection = F.normalize(self.projection_head(context), dim=-1)

        return {
            "sequence_state": sequence_state,
            "cls_state": cls_state,
            "context": context,
            "projection": projection,
            "reconstruction": reconstructions,
            "buy_logits": self.buy_head(context).squeeze(-1),
            "sell_logits": self.sell_head(context).squeeze(-1),
            "direction_logits": self.direction_head(context),
            "regime_logits": self.regime_head(context),
        }

    @torch.no_grad()
    def encode(self, sequences: torch.Tensor) -> torch.Tensor:
        """Return pooled representations for downstream tasks."""
        outputs = self.forward(sequences)
        return outputs["context"]


class NatronInferenceWrapper:
    """
    Utility wrapper handling feature normalization and probability decoding.
    """

    def __init__(
        self,
        model: NatronTransformer,
        feature_mean: torch.Tensor,
        feature_std: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.register_feature_stats(feature_mean, feature_std)

    def register_feature_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        self.feature_mean = mean.to(self.device)
        self.feature_std = std.to(self.device)

    @torch.no_grad()
    def __call__(self, sequence: torch.Tensor) -> Dict[str, float]:
        """
        Run inference on a single (96, 100) feature tensor.
        """
        self.model.eval()
        sequence = sequence.to(self.device)
        normalized = (sequence - self.feature_mean) / (self.feature_std + 1e-6)
        normalized = normalized.unsqueeze(0)  # batch dimension
        outputs = self.model(normalized)
        buy_prob = torch.sigmoid(outputs["buy_logits"]).item()
        sell_prob = torch.sigmoid(outputs["sell_logits"]).item()
        direction_probs = torch.softmax(outputs["direction_logits"], dim=-1).squeeze(0)
        regime_probs = torch.softmax(outputs["regime_logits"], dim=-1).squeeze(0)
        confidence = float(torch.max(regime_probs))
        regime_idx = int(torch.argmax(regime_probs))

        regimes = [
            "BULL_STRONG",
            "BULL_WEAK",
            "RANGE",
            "BEAR_WEAK",
            "BEAR_STRONG",
            "VOLATILE",
        ]

        return {
            "buy_prob": buy_prob,
            "sell_prob": sell_prob,
            "direction_up": direction_probs[1].item(),
            "regime": regimes[regime_idx],
            "confidence": confidence,
        }
