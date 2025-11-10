"""
Natron Multi-Head Transformer Model
Three output heads: regime classification, context strength, and forecast direction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
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
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class NatronTransformer(nn.Module):
    """
    Natron Multi-Head Transformer Model with three output heads.
    """
    
    def __init__(self,
                 n_features: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_seq_len: int = 96,
                 n_regime_classes: int = 6):
        """
        Initialize Natron Transformer.
        
        Args:
            n_features: Number of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            n_regime_classes: Number of regime classes (default 6)
        """
        super().__init__()
        
        self.d_model = d_model
        self.n_features = n_features
        
        # Input projection
        self.input_projection = nn.Linear(n_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        # 1. Regime classification head (6 classes)
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, n_regime_classes)
        )
        
        # 2. Context strength head (1 output: |bull_p - bear_p|)
        self.context_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
        # 3. Forecast direction head (2 classes: down/up)
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, 2)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, seq_len, n_features)
            
        Returns:
            Tuple of (regime_logits, context_score, forecast_logits)
            - regime_logits: (batch_size, n_regime_classes)
            - context_score: (batch_size, 1)
            - forecast_logits: (batch_size, 2)
        """
        # Project input to d_model
        x = self.input_projection(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        encoded = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # Use last timestep for prediction
        last_hidden = encoded[:, -1, :]  # (batch_size, d_model)
        
        # Output heads
        regime_logits = self.regime_head(last_hidden)  # (batch_size, n_regime_classes)
        context_score = self.context_head(last_hidden)  # (batch_size, 1)
        forecast_logits = self.forecast_head(last_hidden)  # (batch_size, 2)
        
        return regime_logits, context_score, forecast_logits


class NatronLoss(nn.Module):
    """
    Combined loss function for Natron model.
    """
    
    def __init__(self,
                 regime_weight: float = 1.0,
                 context_weight: float = 0.5,
                 forecast_weight: float = 1.0,
                 class_weights: Optional[torch.Tensor] = None):
        """
        Initialize loss function.
        
        Args:
            regime_weight: Weight for regime classification loss
            context_weight: Weight for context strength loss
            forecast_weight: Weight for forecast direction loss
            class_weights: Optional class weights for regime classification
        """
        super().__init__()
        self.regime_weight = regime_weight
        self.context_weight = context_weight
        self.forecast_weight = forecast_weight
        
        self.regime_criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.context_criterion = nn.MSELoss()
        self.forecast_criterion = nn.CrossEntropyLoss()
    
    def forward(self,
                regime_logits: torch.Tensor,
                context_score: torch.Tensor,
                forecast_logits: torch.Tensor,
                regime_target: torch.Tensor,
                context_target: torch.Tensor,
                forecast_target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.
        
        Args:
            regime_logits: Regime classification logits
            context_score: Context strength predictions
            forecast_logits: Forecast direction logits
            regime_target: Regime labels
            context_target: Context strength targets
            forecast_target: Forecast direction labels
            
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        # Regime classification loss
        regime_loss = self.regime_criterion(regime_logits, regime_target)
        
        # Context strength loss (MSE)
        context_loss = self.context_criterion(context_score.squeeze(), context_target)
        
        # Forecast direction loss
        forecast_loss = self.forecast_criterion(forecast_logits, forecast_target)
        
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
