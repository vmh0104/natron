"""
Natron Transformer Model - Multi-task financial trading model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder for sequence modeling."""
    
    def __init__(self, 
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_seq_len: int = 96):
        super().__init__()
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(100, d_model)  # 100 features
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            mask: Optional attention mask
        """
        # Project input to d_model
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        # Create padding mask if needed (for masked modeling)
        if mask is not None:
            # Convert boolean mask to attention mask format
            # True values become 0 (attend), False become -inf (mask)
            attn_mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        else:
            attn_mask = None
        
        x = self.transformer_encoder(x, src_key_padding_mask=None if mask is None else ~mask)
        
        # Layer norm
        x = self.layer_norm(x)
        
        return x


class NatronModel(nn.Module):
    """Multi-task Transformer model for financial trading."""
    
    def __init__(self,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 num_features: int = 100,
                 max_seq_len: int = 96,
                 freeze_encoder: bool = False):
        super().__init__()
        
        # Encoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Pooling: use CLS token or mean pooling
        self.pooling = 'mean'  # Options: 'mean', 'last', 'cls'
        
        # Task-specific heads
        hidden_dim = d_model
        
        # Buy/Sell heads (binary classification)
        self.buy_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        self.sell_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Direction head (binary: up/down)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # Regime head (6 classes)
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 6),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            mask: Optional mask for pretraining
            
        Returns:
            Dictionary with predictions for each task
        """
        # Encode sequence
        encoded = self.encoder(x, mask)
        
        # Pooling: mean over sequence length
        if self.pooling == 'mean':
            pooled = encoded.mean(dim=1)  # (batch_size, d_model)
        elif self.pooling == 'last':
            pooled = encoded[:, -1, :]  # (batch_size, d_model)
        else:
            pooled = encoded.mean(dim=1)
        
        # Task-specific predictions
        outputs = {
            'buy': self.buy_head(pooled).squeeze(-1),
            'sell': self.sell_head(pooled).squeeze(-1),
            'direction': self.direction_head(pooled),
            'regime': self.regime_head(pooled)
        }
        
        return outputs
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation (for pretraining)."""
        encoded = self.encoder(x)
        return encoded.mean(dim=1)  # Mean pooling


class MaskedModelingHead(nn.Module):
    """Head for masked language modeling pretraining."""
    
    def __init__(self, d_model: int, num_features: int, dropout: float = 0.1):
        super().__init__()
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, num_features)
        )
    
    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """Reconstruct masked features."""
        return self.reconstruction_head(encoded)


def create_model(config: dict) -> NatronModel:
    """Factory function to create model from config."""
    return NatronModel(
        d_model=config.get('d_model', 128),
        nhead=config.get('nhead', 8),
        num_layers=config.get('num_layers', 6),
        dim_feedforward=config.get('dim_feedforward', 512),
        dropout=config.get('dropout', 0.1),
        num_features=config.get('num_features', 100),
        max_seq_len=config.get('max_seq_len', 96),
        freeze_encoder=config.get('freeze_encoder', False)
    )
