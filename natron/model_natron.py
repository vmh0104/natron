"""
Natron Multi-Head Transformer Model
Three output heads: regime classification, context strength, and forecast direction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch, d_model)
        """
        x = x + self.pe[:x.size(0), :]
        return x


class NatronTransformer(nn.Module):
    """
    Natron Multi-Head Transformer Model.
    
    Architecture:
    - Input: sequence of candles (seq_len, batch, features)
    - Transformer Encoder with positional encoding
    - Three output heads:
      1. Regime classification (6 classes)
      2. Context strength (1 value: |bull_p - bear_p|)
      3. Forecast direction (2 classes: up/down)
    """
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 96,
        num_regime_classes: int = 6,
        num_forecast_classes: int = 2
    ):
        """
        Initialize Natron Transformer.
        
        Args:
            input_dim: Number of input features per candle
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            num_regime_classes: Number of regime classes (default 6)
            num_forecast_classes: Number of forecast classes (default 2)
        """
        super().__init__()
        
        self.d_model = d_model
        self.input_dim = input_dim
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        # Regime head: 6 classes
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_regime_classes)
        )
        
        # Context head: 1 value (strength score)
        self.context_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Forecast head: 2 classes (up/down)
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_forecast_classes)
        )
        
        # Pooling layer (to aggregate sequence)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, features)
            
        Returns:
            Dictionary with:
            - 'regime': (batch, num_regime_classes) logits
            - 'context': (batch, 1) context strength scores
            - 'forecast': (batch, num_forecast_classes) logits
        """
        batch_size, seq_len, features = x.shape
        
        # Project input to d_model
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Transpose for transformer: (seq_len, batch, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        encoded = self.transformer_encoder(x)  # (seq_len, batch, d_model)
        
        # Pool sequence: take mean over sequence dimension
        # Transpose back: (batch, seq_len, d_model)
        encoded = encoded.transpose(0, 1)
        
        # Global average pooling over sequence
        pooled = encoded.mean(dim=1)  # (batch, d_model)
        
        # Output heads
        regime_logits = self.regime_head(pooled)  # (batch, num_regime_classes)
        context_score = self.context_head(pooled)  # (batch, 1)
        forecast_logits = self.forecast_head(pooled)  # (batch, num_forecast_classes)
        
        return {
            'regime': regime_logits,
            'context': context_score,
            'forecast': forecast_logits
        }


class NatronLoss(nn.Module):
    """
    Combined loss function for multi-head outputs.
    Uses weighted cross-entropy for classification heads.
    """
    
    def __init__(
        self,
        regime_weight: float = 1.0,
        context_weight: float = 0.5,
        forecast_weight: float = 1.0
    ):
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
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined loss.
        
        Args:
            predictions: Dict with 'regime', 'context', 'forecast'
            targets: Dict with 'regime', 'context', 'forecast'
            
        Returns:
            Total loss and individual loss components
        """
        # Regime classification loss
        regime_loss = self.regime_loss(predictions['regime'], targets['regime'])
        
        # Forecast classification loss
        forecast_loss = self.forecast_loss(predictions['forecast'], targets['forecast'])
        
        # Context strength loss (MSE)
        context_loss = self.context_loss(
            predictions['context'].squeeze(),
            targets['context'].float()
        )
        
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
