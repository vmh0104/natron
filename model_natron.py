"""
Natron Transformer model definition.

Includes:
    - Transformer encoder with learnable CLS token and rotary-style positional embeddings.
    - Masked reconstruction head for self-supervised pretraining.
    - Contrastive projection head for InfoNCE objectives.
    - Multi-task classification heads for buy/sell, direction, and regime targets.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    input_dim: int = 100
    d_model: int = 256
    n_heads: int = 8
    num_layers: int = 6
    dim_feedforward: int = 1024
    dropout: float = 0.1
    activation: str = "gelu"
    max_seq_len: int = 128
    projection_dim: int = 128
    masking_ratio: float = 0.2
    layer_norm_eps: float = 1e-5
    use_rotary: bool = False


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with optional learned scale."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        return x + self.alpha * self.pe[:, :length]


class FeedForward(nn.Module):
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float, activation: str):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation) if hasattr(F, activation) else F.gelu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Pre-norm Transformer encoder layer with optional rotary embeddings."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.n_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        self.ff = FeedForward(
            config.d_model,
            config.dim_feedforward,
            config.dropout,
            config.activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        attn_output, _ = self.self_attn(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask
        )
        x = residual + self.dropout(attn_output)

        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ff(x))
        return x


class MultiTaskHeads(nn.Module):
    """Projection and classification heads for Natron Transformer."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        d_model = config.d_model
        self.pre_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.buy_head = nn.Linear(d_model, 1)
        self.sell_head = nn.Linear(d_model, 1)
        self.direction_head = nn.Linear(d_model, 2)
        self.regime_head = nn.Linear(d_model, 6)

        self.reconstruction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, config.input_dim),
        )
        self.projection_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, config.projection_dim),
        )

    def forward(self, pooled: torch.Tensor, sequence: torch.Tensor) -> Dict[str, torch.Tensor]:
        conditioned = self.pre_head(pooled)
        logits_buy = self.buy_head(conditioned).squeeze(-1)
        logits_sell = self.sell_head(conditioned).squeeze(-1)
        direction_logits = self.direction_head(conditioned)
        regime_logits = self.regime_head(conditioned)
        reconstruction = self.reconstruction_head(sequence)
        projection = F.normalize(self.projection_head(conditioned), dim=-1)

        return {
            "buy_logits": logits_buy,
            "sell_logits": logits_sell,
            "direction_logits": direction_logits,
            "regime_logits": regime_logits,
            "reconstruction": reconstruction,
            "projection": projection,
        }


class NatronTransformer(nn.Module):
    """End-to-end Natron Transformer with pretraining and supervised heads."""

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.input_proj = nn.Linear(self.config.input_dim, self.config.d_model)
        self.layer_drop = nn.Dropout(self.config.dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.d_model))
        self.positional_encoding = PositionalEncoding(
            self.config.d_model, max_len=self.config.max_seq_len + 1
        )
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer(self.config) for _ in range(self.config.num_layers)]
        )
        self.norm = nn.LayerNorm(self.config.d_model, eps=self.config.layer_norm_eps)
        self.heads = MultiTaskHeads(self.config)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.normal_(self.cls_token, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

    def encode(
        self,
        inputs: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode the input sequence and return contextualized representations.

        Args:
            inputs: Tensor of shape (B, T, F)
            padding_mask: Bool tensor of shape (B, T) where True indicates padding.
            attn_mask: Optional attention mask for causal/probability attention.
        """
        batch_size, seq_len, _ = inputs.shape
        token_embeddings = self.input_proj(inputs)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, token_embeddings], dim=1)
        x = self.positional_encoding(x)
        x = self.layer_drop(x)

        if padding_mask is not None:
            padding_mask = F.pad(padding_mask, (1, 0), value=False)

        for layer in self.encoder_layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=padding_mask)

        x = self.norm(x)
        cls_rep = x[:, 0]
        sequence_rep = x[:, 1:]
        return {"cls": cls_rep, "sequence": sequence_rep}

    def forward(
        self,
        inputs: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        encodings = self.encode(inputs, padding_mask, attn_mask)
        heads = self.heads(encodings["cls"], encodings["sequence"])
        heads.update(
            {
                "cls_embedding": encodings["cls"],
                "sequence_embeddings": encodings["sequence"],
            }
        )
        return heads

    def masked_reconstruct(
        self,
        inputs: torch.Tensor,
        mask: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass tailored for masked token reconstruction.

        Args:
            inputs: Float tensor (B, T, F).
            mask: Bool tensor (B, T, F) indicating masked positions.
        """
        outputs = self.forward(inputs, padding_mask=padding_mask)
        reconstruction = outputs["reconstruction"]
        masked_pred = reconstruction[mask]
        masked_target = inputs[mask]
        return {"pred": masked_pred, "target": masked_target}

    def contrastive_projection(
        self,
        inputs: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.forward(inputs, padding_mask=padding_mask)["projection"]


__all__ = ["NatronTransformer", "ModelConfig"]
