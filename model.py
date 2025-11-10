"""
Natron Transformer Model - Multi-task financial trading model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from typing import Optional, Dict, Tuple


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
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class NatronTransformer(nn.Module):
    """
    Multi-task Transformer for financial trading
    Supports pretraining and supervised fine-tuning
    """
    
    def __init__(
        self,
        num_features: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 96,
        use_pretrain: bool = False
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.use_pretrain = use_pretrain
        
        # Input projection
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = TransformerEncoder(encoder_layers, num_layers)
        
        # Pretraining head (for masked modeling)
        if use_pretrain:
            self.pretrain_head = nn.Linear(d_model, num_features)
        
        # Multi-task heads
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
        
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
        
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 6)
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mode: str = 'supervised'
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor [batch_size, seq_len, num_features]
            mask: Optional mask for pretraining [batch_size, seq_len]
            mode: 'pretrain' or 'supervised'
        
        Returns:
            Dictionary with predictions
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)  # [B, L, d_model]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Create padding mask if needed
        src_mask = None
        if mask is not None:
            src_mask = mask.bool()
        
        # Transformer encoding
        encoded = self.transformer(x, src_key_padding_mask=src_mask)
        
        # Layer norm
        encoded = self.layer_norm(encoded)
        
        # Use last timestep for prediction
        last_hidden = encoded[:, -1, :]  # [B, d_model]
        
        outputs = {}
        
        if mode == 'pretrain':
            # Reconstruct masked features
            if mask is not None:
                masked_positions = mask
                masked_features = encoded[masked_positions]
                outputs['reconstruction'] = self.pretrain_head(masked_features)
            else:
                # Reconstruct all
                outputs['reconstruction'] = self.pretrain_head(encoded)
        
        elif mode == 'supervised':
            # Multi-task predictions
            outputs['buy'] = self.buy_head(last_hidden).squeeze(-1)
            outputs['sell'] = self.sell_head(last_hidden).squeeze(-1)
            outputs['direction'] = self.direction_head(last_hidden)
            outputs['regime'] = self.regime_head(last_hidden)
        
        # Also return hidden states for contrastive learning
        outputs['hidden_states'] = encoded
        
        return outputs


class ContrastiveHead(nn.Module):
    """Head for contrastive learning (SimCLR style)"""
    
    def __init__(self, d_model: int, projection_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project to contrastive space"""
        return F.normalize(self.projection(x), p=2, dim=-1)


class NatronPretrainModel(nn.Module):
    """Wrapper for pretraining with contrastive learning"""
    
    def __init__(self, base_model: NatronTransformer):
        super().__init__()
        self.base_model = base_model
        self.contrastive_head = ContrastiveHead(base_model.d_model)
    
    def forward(
        self, 
        x: torch.Tensor,
        x_aug: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward for pretraining
        
        Args:
            x: Original sequence
            x_aug: Augmented sequence (for contrastive learning)
            mask: Mask for masked modeling
        """
        outputs = self.base_model(x, mask=mask, mode='pretrain')
        
        # Contrastive projections
        hidden = outputs['hidden_states'][:, -1, :]  # Last timestep
        outputs['projection'] = self.contrastive_head(hidden)
        
        if x_aug is not None:
            outputs_aug = self.base_model(x_aug, mode='pretrain')
            hidden_aug = outputs_aug['hidden_states'][:, -1, :]
            outputs['projection_aug'] = self.contrastive_head(hidden_aug)
        
        return outputs


def create_model(
    num_features: int = 100,
    d_model: int = 256,
    nhead: int = 8,
    num_layers: int = 6,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    pretrain: bool = False
) -> nn.Module:
    """Factory function to create model"""
    base_model = NatronTransformer(
        num_features=num_features,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        use_pretrain=pretrain
    )
    
    if pretrain:
        return NatronPretrainModel(base_model)
    else:
        return base_model
