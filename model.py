"""
Natron Transformer: Multi-task financial trading model
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class NatronEncoder(nn.Module):
    """Transformer encoder for market sequence encoding"""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 96
    ):
        super().__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            
        Returns:
            Tensor, shape [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # Transpose for transformer: [seq_len, batch_size, d_model]
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask if needed
        if mask is not None:
            # Transformer expects mask where True positions are ignored
            mask = mask.transpose(0, 1) if mask.dim() == 2 else mask
        
        # Apply transformer encoder
        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Transpose back: [batch_size, seq_len, d_model]
        x = x.transpose(0, 1)
        
        # Layer norm
        x = self.layer_norm(x)
        
        return x


class NatronModel(nn.Module):
    """Multi-task Transformer model for financial trading"""
    
    def __init__(
        self,
        input_dim: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        sequence_length: int = 96
    ):
        super().__init__()
        
        # Encoder
        self.encoder = NatronEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=sequence_length
        )
        
        # Pooling: use last timestep + global average pooling
        self.pooling = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Multi-task heads
        # Buy/Sell classification (binary)
        self.buy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.sell_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Direction prediction (up/down)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
            nn.Softmax(dim=-1)
        )
        
        # Regime classification (6 classes)
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 6),
            nn.Softmax(dim=-1)
        )
        
    def forward(
        self, 
        x: torch.Tensor,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary with predictions for each task
        """
        # Encode sequence
        encoded = self.encoder(x)  # [batch_size, seq_len, d_model]
        
        # Pooling: concatenate last timestep and global average
        last_timestep = encoded[:, -1, :]  # [batch_size, d_model]
        global_avg = encoded.mean(dim=1)  # [batch_size, d_model]
        pooled = torch.cat([last_timestep, global_avg], dim=-1)  # [batch_size, d_model * 2]
        pooled = self.pooling(pooled)  # [batch_size, d_model]
        
        # Multi-task predictions
        buy_prob = self.buy_head(pooled).squeeze(-1)  # [batch_size]
        sell_prob = self.sell_head(pooled).squeeze(-1)  # [batch_size]
        direction = self.direction_head(pooled)  # [batch_size, 2]
        regime = self.regime_head(pooled)  # [batch_size, 6]
        
        outputs = {
            'buy': buy_prob,
            'sell': sell_prob,
            'direction': direction,
            'regime': regime
        }
        
        if return_features:
            outputs['features'] = pooled
        
        return outputs


class MaskedModelingHead(nn.Module):
    """Head for masked token reconstruction (pretraining)"""
    
    def __init__(self, d_model: int, input_dim: int, dropout: float = 0.1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, input_dim)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)
