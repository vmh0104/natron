"""
Natron Transformer Model - Multi-Task Financial Trading Model

Architecture:
- Encoder: Transformer encoder with positional encoding
- Multi-head outputs: buy, sell, direction, regime
- Supports pretraining (masked modeling) and supervised fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict


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


class NatronEncoder(nn.Module):
    """Transformer encoder for market sequence understanding."""
    
    def __init__(
        self,
        input_dim: int = 100,
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
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            mask: Optional attention mask
        
        Returns:
            Encoded sequence of shape (batch_size, seq_len, d_model)
        """
        # Project input
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask if needed (for variable length sequences)
        src_mask = None
        if mask is not None:
            src_mask = mask
        
        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=src_mask)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        return x


class NatronModel(nn.Module):
    """Complete Natron Transformer model with multi-task heads."""
    
    def __init__(
        self,
        input_dim: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 96,
        freeze_encoder: bool = False
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
            max_seq_len=max_seq_len
        )
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Pooling: Use last token or mean pooling
        self.pooling = 'mean'  # 'last' or 'mean'
        
        # Task-specific heads
        # Buy/Sell heads (binary classification)
        self.buy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        self.sell_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Direction head (binary: up/down)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),
            nn.LogSoftmax(dim=-1)
        )
        
        # Regime head (6 classes)
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 6),
            nn.LogSoftmax(dim=-1)
        )
        
        # Reconstruction head for pretraining
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, input_dim)
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        mode: str = 'supervised',
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, input_dim)
            mode: 'pretrain', 'supervised', or 'inference'
            mask: Optional mask for pretraining
        
        Returns:
            Dictionary with predictions
        """
        # Encode sequence
        encoded = self.encoder(x, mask=mask)
        
        # Pooling: mean over sequence length
        if self.pooling == 'mean':
            pooled = encoded.mean(dim=1)  # (batch_size, d_model)
        else:  # last token
            pooled = encoded[:, -1, :]  # (batch_size, d_model)
        
        result = {}
        
        if mode == 'pretrain':
            # Reconstruction for masked tokens
            reconstructed = self.reconstruction_head(encoded)  # (batch_size, seq_len, input_dim)
            result['reconstructed'] = reconstructed
            result['encoded'] = encoded
        
        elif mode == 'supervised' or mode == 'inference':
            # Multi-task predictions
            result['buy'] = self.buy_head(pooled)
            result['sell'] = self.sell_head(pooled)
            result['direction'] = self.direction_head(pooled)
            result['regime'] = self.regime_head(pooled)
            result['encoded'] = encoded
        
        return result
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Convenience method for inference."""
        self.eval()
        with torch.no_grad():
            return self.forward(x, mode='inference')
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoded representation."""
        encoded = self.encoder(x)
        if self.pooling == 'mean':
            return encoded.mean(dim=1)
        else:
            return encoded[:, -1, :]


class ContrastiveEncoder(nn.Module):
    """Encoder for contrastive learning (SimCLR style)."""
    
    def __init__(self, base_encoder: NatronEncoder, projection_dim: int = 128):
        super().__init__()
        self.encoder = base_encoder
        self.projection_head = nn.Sequential(
            nn.Linear(base_encoder.d_model, base_encoder.d_model // 2),
            nn.ReLU(),
            nn.Linear(base_encoder.d_model // 2, projection_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Get contrastive projection (L2 normalized)."""
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        projection = self.projection_head(pooled)
        # L2 normalize
        return F.normalize(projection, p=2, dim=-1)


# Helper function to create model
def create_natron_model(
    input_dim: int = 100,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 6,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    max_seq_len: int = 96,
    freeze_encoder: bool = False,
    pretrained_path: Optional[str] = None
) -> NatronModel:
    """
    Create and optionally load pretrained Natron model.
    
    Args:
        input_dim: Input feature dimension
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of transformer layers
        dim_feedforward: Feedforward dimension
        dropout: Dropout rate
        max_seq_len: Maximum sequence length
        freeze_encoder: Whether to freeze encoder weights
        pretrained_path: Path to pretrained encoder weights
    
    Returns:
        NatronModel instance
    """
    model = NatronModel(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        max_seq_len=max_seq_len,
        freeze_encoder=freeze_encoder
    )
    
    if pretrained_path is not None:
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        if 'encoder_state_dict' in checkpoint:
            model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    
    return model
