"""
Natron Transformer Model
Multi-task Transformer for financial trading predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerEncoder(nn.Module):
    """Transformer Encoder for sequence encoding"""
    
    def __init__(self, d_model: int = 128, nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x, mask: Optional[torch.Tensor] = None):
        return self.encoder(x, src_key_padding_mask=mask)


class NatronTransformer(nn.Module):
    """
    Multi-task Transformer for financial trading.
    
    Architecture:
    - Input: (batch, 96, 100) feature sequences
    - Encoder: Transformer layers
    - Heads: Buy, Sell, Direction, Regime
    """
    
    def __init__(self, input_dim: int = 100, sequence_length: int = 96,
                 d_model: int = 128, nhead: int = 8, num_layers: int = 6,
                 dim_feedforward: int = 512, dropout: float = 0.1,
                 freeze_encoder: bool = False):
        super().__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)
        
        # Transformer encoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Pooling (use CLS token or mean pooling)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Task heads
        self.buy_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.sell_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
            nn.LogSoftmax(dim=-1)
        )
        
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 6),
            nn.LogSoftmax(dim=-1)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, input_dim) feature sequences
            mask: (batch, seq_len) padding mask
        
        Returns:
            Dictionary with predictions for each task
        """
        # Project input
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # Encode sequence
        encoded = self.encoder(x, mask=mask)  # (batch, seq_len, d_model)
        
        # Pool sequence (mean pooling)
        # Transpose for pooling: (batch, d_model, seq_len)
        encoded_t = encoded.transpose(1, 2)
        pooled = self.pooling(encoded_t).squeeze(-1)  # (batch, d_model)
        
        # Task predictions
        buy_prob = self.buy_head(pooled).squeeze(-1)  # (batch,)
        sell_prob = self.sell_head(pooled).squeeze(-1)  # (batch,)
        direction_logits = self.direction_head(pooled)  # (batch, 2)
        regime_logits = self.regime_head(pooled)  # (batch, 6)
        
        return {
            'buy': buy_prob,
            'sell': sell_prob,
            'direction': direction_logits,
            'regime': regime_logits,
            'encoded': encoded  # For pretraining
        }


class NatronPretrainModel(nn.Module):
    """
    Pretraining model for masked reconstruction and contrastive learning.
    """
    
    def __init__(self, encoder: TransformerEncoder, d_model: int = 128,
                 input_dim: int = 100, mask_ratio: float = 0.15):
        super().__init__()
        self.encoder = encoder
        self.d_model = d_model
        self.input_dim = input_dim
        self.mask_ratio = mask_ratio
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Reconstruction head
        self.reconstruction_head = nn.Linear(d_model, input_dim)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                return_reconstruction: bool = True):
        """
        Forward pass for pretraining.
        
        Args:
            x: (batch, seq_len, input_dim) features
            mask: Optional padding mask
            return_reconstruction: Whether to return reconstruction
        
        Returns:
            Dictionary with reconstruction and/or projections
        """
        # Create random mask for masked modeling
        batch_size, seq_len, _ = x.shape
        mask_tokens = torch.rand(batch_size, seq_len, device=x.device) < self.mask_ratio
        
        # Mask input (replace with zeros)
        x_masked = x.clone()
        x_masked[mask_tokens] = 0
        
        # Project to d_model
        x_proj = self.input_projection(x_masked)
        
        # Encode
        encoded = self.encoder(x_proj, mask=mask)
        
        # Reconstruction
        reconstruction = None
        if return_reconstruction:
            reconstruction = self.reconstruction_head(encoded)
        
        # Projection for contrastive learning
        projection = self.projection_head(encoded.mean(dim=1))  # Mean pool
        
        return {
            'reconstruction': reconstruction,
            'projection': projection,
            'mask_tokens': mask_tokens
        }
