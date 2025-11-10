"""
Natron Multi-Head Transformer Model

This module implements the Natron transformer model with three output heads:
1. Regime classification (6 classes)
2. Context strength scoring (|bull_p - bear_p|)
3. Forecast direction (up/down next N candles)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer input sequences.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [seq_len, batch_size, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class NatronTransformer(nn.Module):
    """
    Multi-head transformer model for Natron trading system.
    
    Architecture:
    - Transformer Encoder with positional encoding
    - Three output heads: regime, context, forecast
    """
    
    def __init__(self,
                 input_dim: int,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 seq_len: int = 96,
                 num_regime_classes: int = 6):
        """
        Initialize Natron transformer model.
        
        Args:
            input_dim: Number of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            seq_len: Input sequence length
            num_regime_classes: Number of regime classes (default 6)
        """
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        self.num_regime_classes = num_regime_classes
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        # Regime classification head (6 classes)
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_regime_classes)
        )
        
        # Context strength head (1 output: |bull_p - bear_p|)
        self.context_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Forecast direction head (2 classes: up/down)
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            
        Returns:
            Tuple of (regime_logits, context_score, forecast_logits)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Project input to d_model
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # Transpose for transformer (seq_len, batch_size, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask if needed
        if mask is None:
            # No padding mask (assume all sequences are valid)
            mask = None
        else:
            # Convert to transformer mask format
            mask = mask.bool()
        
        # Transformer encoder
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)  # [seq_len, batch_size, d_model]
        
        # Use last timestep for prediction (or mean pooling)
        # Option 1: Last timestep
        last_hidden = encoded[-1]  # [batch_size, d_model]
        
        # Option 2: Mean pooling (alternative)
        # last_hidden = encoded.mean(dim=0)  # [batch_size, d_model]
        
        # Regime classification head
        regime_logits = self.regime_head(last_hidden)  # [batch_size, num_regime_classes]
        
        # Context strength head
        context_score = self.context_head(last_hidden)  # [batch_size, 1]
        
        # Forecast direction head
        forecast_logits = self.forecast_head(last_hidden)  # [batch_size, 2]
        
        return regime_logits, context_score, forecast_logits


class NatronLoss(nn.Module):
    """
    Combined loss function for multi-head training.
    """
    
    def __init__(self,
                 regime_weight: float = 1.0,
                 context_weight: float = 0.5,
                 forecast_weight: float = 1.0):
        """
        Initialize loss function.
        
        Args:
            regime_weight: Weight for regime classification loss
            context_weight: Weight for context strength loss
            forecast_weight: Weight for forecast direction loss
        """
        super().__init__()
        self.regime_weight = regime_weight
        self.context_weight = context_weight
        self.forecast_weight = forecast_weight
        
        self.regime_loss = nn.CrossEntropyLoss()
        self.forecast_loss = nn.CrossEntropyLoss()
        self.context_loss = nn.MSELoss()
    
    def forward(self,
                regime_pred: torch.Tensor,
                context_pred: torch.Tensor,
                forecast_pred: torch.Tensor,
                regime_target: torch.Tensor,
                context_target: torch.Tensor,
                forecast_target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            regime_pred: Regime predictions [batch_size, num_classes]
            context_pred: Context predictions [batch_size, 1]
            forecast_pred: Forecast predictions [batch_size, 2]
            regime_target: Regime targets [batch_size]
            context_target: Context targets [batch_size, 1]
            forecast_target: Forecast targets [batch_size]
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Regime classification loss
        regime_loss = self.regime_loss(regime_pred, regime_target)
        
        # Forecast direction loss
        forecast_loss = self.forecast_loss(forecast_pred, forecast_target)
        
        # Context strength loss (MSE)
        context_loss = self.context_loss(context_pred.squeeze(), context_target.squeeze())
        
        # Combined loss
        total_loss = (
            self.regime_weight * regime_loss +
            self.context_weight * context_loss +
            self.forecast_weight * forecast_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'regime': regime_loss.item(),
            'context': context_loss.item(),
            'forecast': forecast_loss.item()
        }
        
        return total_loss, loss_dict
