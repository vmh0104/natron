"""
NatronTransformer - Multi-Task Transformer Model for Financial Trading
Part of Natron Transformer Multi-Task Trading System
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class NatronTransformerEncoder(nn.Module):
    """
    Transformer encoder for learning market representations.
    Can be used for pretraining (masked modeling) or supervised learning.
    """
    
    def __init__(self,
                 feature_dim: int = 100,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 activation: str = "gelu"):
        super().__init__()
        
        self.d_model = d_model
        self.feature_dim = feature_dim
        
        # Input projection
        self.input_projection = nn.Linear(feature_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=200, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer norm
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, feature_dim)
            mask: Optional attention mask
            
        Returns:
            encoded: (batch_size, seq_len, d_model)
        """
        # Project input
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask if needed (for variable length sequences)
        if mask is not None:
            # Convert boolean mask to attention mask format
            # True positions will be masked
            src_key_padding_mask = ~mask  # Invert: True = pad
        else:
            src_key_padding_mask = None
        
        # Transformer encoding
        encoded = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Layer norm
        encoded = self.layer_norm(encoded)
        
        return encoded


class NatronTransformer(nn.Module):
    """
    Multi-task Transformer model for financial trading.
    Predicts: buy, sell, direction, regime simultaneously.
    """
    
    def __init__(self,
                 feature_dim: int = 100,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 use_pooling: bool = True):
        super().__init__()
        
        # Encoder
        self.encoder = NatronTransformerEncoder(
            feature_dim=feature_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation
        )
        
        self.use_pooling = use_pooling
        
        # Pooling layer (aggregate sequence representation)
        if use_pooling:
            self.pooling = nn.AdaptiveAvgPool1d(1)
            self.pooling_linear = nn.Linear(d_model, d_model)
        else:
            # Use last token
            self.pooling_linear = nn.Linear(d_model, d_model)
        
        # Task-specific heads
        hidden_dim = d_model
        
        # Buy/Sell heads (binary classification)
        self.buy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.sell_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Direction head (binary: up/down)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2),
            nn.LogSoftmax(dim=-1)
        )
        
        # Regime head (6 classes)
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 6),
            nn.LogSoftmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, feature_dim)
            mask: Optional attention mask
            
        Returns:
            Dictionary with predictions:
            - 'buy': (batch_size, 1)
            - 'sell': (batch_size, 1)
            - 'direction': (batch_size, 2) log probabilities
            - 'regime': (batch_size, 6) log probabilities
        """
        # Encode sequence
        encoded = self.encoder(x, mask)  # (batch_size, seq_len, d_model)
        
        # Aggregate sequence representation
        if self.use_pooling:
            # Adaptive average pooling
            encoded_t = encoded.transpose(1, 2)  # (batch_size, d_model, seq_len)
            pooled = self.pooling(encoded_t).squeeze(-1)  # (batch_size, d_model)
        else:
            # Use last token
            pooled = encoded[:, -1, :]  # (batch_size, d_model)
        
        # Project pooled representation
        pooled = self.pooling_linear(pooled)
        pooled = F.relu(pooled)
        
        # Task-specific predictions
        buy_prob = self.buy_head(pooled)
        sell_prob = self.sell_head(pooled)
        direction_logits = self.direction_head(pooled)
        regime_logits = self.regime_head(pooled)
        
        return {
            'buy': buy_prob,
            'sell': sell_prob,
            'direction': direction_logits,
            'regime': regime_logits
        }
    
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Extract encoded representations (for pretraining or feature extraction).
        """
        return self.encoder(x, mask)


class MaskedModelingHead(nn.Module):
    """
    Head for masked language modeling pretraining.
    Reconstructs masked feature tokens.
    """
    
    def __init__(self, d_model: int, feature_dim: int, dropout: float = 0.1):
        super().__init__()
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, feature_dim)
        )
    
    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoded: (batch_size, seq_len, d_model)
        Returns:
            reconstructed: (batch_size, seq_len, feature_dim)
        """
        return self.reconstruction_head(encoded)


class ContrastiveHead(nn.Module):
    """
    Head for contrastive learning pretraining.
    Projects representations to a normalized embedding space.
    """
    
    def __init__(self, d_model: int, projection_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim),
            nn.LayerNorm(projection_dim)
        )
    
    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoded: (batch_size, seq_len, d_model) or (batch_size, d_model)
        Returns:
            projected: Normalized embeddings
        """
        if len(encoded.shape) == 3:
            # Pool sequence
            encoded = encoded.mean(dim=1)  # (batch_size, d_model)
        
        projected = self.projection(encoded)
        # L2 normalize
        projected = F.normalize(projected, p=2, dim=-1)
        return projected
