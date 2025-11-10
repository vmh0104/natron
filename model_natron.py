"""
Natron V2 Transformer backbone and task heads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class NatronTransformerConfig:
    input_dim: int = 100
    d_model: int = 256
    n_heads: int = 8
    num_layers: int = 6
    mlp_ratio: int = 4
    dropout: float = 0.1
    activation: str = "gelu"
    max_seq_len: int = 256
    projection_dim: int = 128
    mask_ratio: float = 0.25


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class ProjectionHead(nn.Module):
    def __init__(self, d_model: int, projection_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Linear(d_model, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NatronTransformer(nn.Module):
    """Multi-task transformer with masked modelling and contrastive heads."""

    def __init__(self, config: NatronTransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.input_dim, config.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.mask_token_feat = nn.Parameter(torch.zeros(1, 1, config.input_dim))
        self.pos_encoding = SinusoidalPositionalEncoding(config.d_model, config.max_seq_len + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * config.mlp_ratio,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.post_norm = nn.LayerNorm(config.d_model)

        # Heads
        self.reconstruction_head = nn.Linear(config.d_model, config.input_dim)
        self.projection_head = ProjectionHead(config.d_model, config.projection_dim)

        self.buy_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))
        self.sell_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 1))
        self.direction_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 2))
        self.regime_head = nn.Sequential(nn.LayerNorm(config.d_model), nn.Linear(config.d_model, 6))

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.input_proj.weight, std=0.02)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.trunc_normal_(self.reconstruction_head.weight, std=0.02)
        nn.init.zeros_(self.reconstruction_head.bias)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.mask_token_feat, std=0.02)

    # ------------------------------------------------------------------ #
    # Forward paths
    # ------------------------------------------------------------------ #

    def encode(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode sequences and return (cls_output, token_embeddings).
        """
        batch_size, seq_len, _ = x.shape
        token_embeddings = self.input_proj(x)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, token_embeddings], dim=1)
        tokens = self.pos_encoding(tokens)

        if attn_mask is not None:
            # pad mask expects shape (batch, seq_len+1)
            attn_mask = torch.cat([torch.zeros(attn_mask.size(0), 1, device=x.device, dtype=attn_mask.dtype), attn_mask], dim=1)
        encoded = self.encoder(tokens, src_key_padding_mask=attn_mask)
        encoded = self.post_norm(encoded)

        cls_output = encoded[:, 0]
        token_output = encoded[:, 1:]
        return cls_output, token_output

    def forward_supervised(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        cls_output, _ = self.encode(x)
        return {
            "buy_logits": self.buy_head(cls_output),
            "sell_logits": self.sell_head(cls_output),
            "direction_logits": self.direction_head(cls_output),
            "regime_logits": self.regime_head(cls_output),
            "cls_output": cls_output,
        }

    def forward_pretrain(self, x: torch.Tensor, mask_ratio: Optional[float] = None) -> Dict[str, torch.Tensor]:
        mask_ratio = mask_ratio or self.config.mask_ratio
        masked_x, mask = self.apply_mask(x, mask_ratio)
        cls_output, token_output = self.encode(masked_x)
        recon = self.reconstruction_head(token_output)
        projection = self.projection_head(cls_output)
        return {
            "reconstruction": recon,
            "mask": mask,
            "projection": projection,
            "token_output": token_output,
            "cls_output": cls_output,
        }

    # ------------------------------------------------------------------ #
    # Utility
    # ------------------------------------------------------------------ #

    def apply_mask(self, x: torch.Tensor, mask_ratio: float) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        mask = torch.rand(batch_size, seq_len, device=x.device) < mask_ratio
        mask_token = self.mask_token_feat.expand(batch_size, seq_len, -1)
        masked = torch.where(mask.unsqueeze(-1), mask_token, x)
        return masked, mask

    def forward(self, x: torch.Tensor, task: str = "supervised", mask_ratio: Optional[float] = None) -> Dict[str, torch.Tensor]:
        if task == "supervised":
            return self.forward_supervised(x)
        if task == "pretrain":
            return self.forward_pretrain(x, mask_ratio=mask_ratio)
        raise ValueError(f"Unknown task '{task}'")

    def get_num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
