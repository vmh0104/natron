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
    """Positional encoding for transformer."""
    
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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class NatronTransformer(nn.Module):
    """
    Multi-head transformer model for Natron trading system.
    
    Architecture:
    - Transformer Encoder with positional encoding
    - Three output heads:
      1. Regime classification (6 classes)
      2. Context strength (1 value)
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
        activation: str = "gelu",
        regime_classes: int = 6,
        context_output_dim: int = 1,
        forecast_classes: int = 2,
        max_seq_len: int = 96
    ):
        """
        Initialize Natron Transformer model.
        
        Args:
            input_dim: Number of input features
            d_model: Embedding dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward network dimension
            dropout: Dropout rate
            activation: Activation function ('relu' or 'gelu')
            regime_classes: Number of regime classes (default: 6)
            context_output_dim: Dimension of context output (default: 1)
            forecast_classes: Number of forecast classes (default: 2)
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.regime_classes = regime_classes
        self.context_output_dim = context_output_dim
        self.forecast_classes = forecast_classes
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        # Regime classification head
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, regime_classes)
        )
        
        # Context strength head (sigmoid output)
        self.context_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, context_output_dim),
            nn.Sigmoid()
        )
        
        # Forecast direction head
        self.forecast_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, forecast_classes)
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
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            mask: Optional attention mask
            
        Returns:
            Tuple of (regime_logits, context_score, forecast_logits)
        """
        # Project input to model dimension
        x = self.input_projection(x)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        # Note: Transformer expects [seq_len, batch_size, d_model] for mask, but we use batch_first=True
        encoded = self.transformer_encoder(x, mask=mask)  # [batch_size, seq_len, d_model]
        
        # Use the last timestep for prediction
        last_hidden = encoded[:, -1, :]  # [batch_size, d_model]
        
        # Regime classification
        regime_logits = self.regime_head(last_hidden)  # [batch_size, regime_classes]
        
        # Context strength
        context_score = self.context_head(last_hidden)  # [batch_size, context_output_dim]
        
        # Forecast direction
        forecast_logits = self.forecast_head(last_hidden)  # [batch_size, forecast_classes]
        
        return regime_logits, context_score, forecast_logits
    
    def predict(self, x: torch.Tensor) -> dict:
        """
        Make predictions with model in eval mode.
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            
        Returns:
            Dictionary with predictions
        """
        self.eval()
        with torch.no_grad():
            regime_logits, context_score, forecast_logits = self.forward(x)
            
            # Get probabilities
            regime_probs = F.softmax(regime_logits, dim=-1)
            forecast_probs = F.softmax(forecast_logits, dim=-1)
            
            # Get predictions
            regime_pred = torch.argmax(regime_probs, dim=-1)
            forecast_pred = torch.argmax(forecast_probs, dim=-1)
            
            return {
                'regime_logits': regime_logits,
                'regime_probs': regime_probs,
                'regime_pred': regime_pred,
                'context_score': context_score,
                'forecast_logits': forecast_logits,
                'forecast_probs': forecast_probs,
                'forecast_pred': forecast_pred
            }
