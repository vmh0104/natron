"""
Natron Transformer - Multi-task Financial Trading Model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple


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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class TransformerEncoder(nn.Module):
    """Transformer Encoder for sequence modeling"""
    
    def __init__(self, 
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 1024,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            (batch_size, seq_len, d_model)
        """
        return self.transformer(x, src_key_padding_mask=mask)


class NatronTransformer(nn.Module):
    """Multi-task Transformer for financial trading"""
    
    def __init__(self,
                 num_features: int = 100,
                 d_model: int = 256,
                 n_heads: int = 8,
                 n_layers: int = 6,
                 d_ff: int = 1024,
                 dropout: float = 0.1,
                 sequence_length: int = 96):
        super().__init__()
        
        self.d_model = d_model
        self.num_features = num_features
        
        # Input projection
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len=sequence_length)
        
        # Transformer encoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout
        )
        
        # Task heads
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
        
        # Reconstruction head for pretraining
        self.reconstruction_head = nn.Linear(d_model, num_features)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, 
                x: torch.Tensor,
                return_sequence: bool = False) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, num_features)
            return_sequence: If True, return full sequence outputs
        Returns:
            Dictionary with predictions
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to model dimension
        x = self.input_projection(x)  # (B, L, d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Encode
        encoded = self.encoder(x)  # (B, L, d_model)
        
        # Use last token for predictions (or mean pooling)
        if return_sequence:
            pooled = encoded.mean(dim=1)  # (B, d_model)
        else:
            pooled = encoded[:, -1, :]  # (B, d_model)
        
        # Task predictions
        buy_prob = self.buy_head(pooled).squeeze(-1)  # (B,)
        sell_prob = self.sell_head(pooled).squeeze(-1)  # (B,)
        direction_logits = self.direction_head(pooled)  # (B, 2)
        regime_logits = self.regime_head(pooled)  # (B, 6)
        
        return {
            'buy': buy_prob,
            'sell': sell_prob,
            'direction': direction_logits,
            'regime': regime_logits,
            'encoded': encoded  # For pretraining
        }
    
    def reconstruct(self, encoded: torch.Tensor) -> torch.Tensor:
        """Reconstruct features from encoded representation"""
        return self.reconstruction_head(encoded)


class PretrainingLoss(nn.Module):
    """Loss for masked modeling pretraining"""
    
    def __init__(self, mask_ratio: float = 0.15):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.mse_loss = nn.MSELoss()
    
    def forward(self, 
                model: NatronTransformer,
                x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Masked modeling: randomly mask tokens and reconstruct
        
        Args:
            model: NatronTransformer model
            x: (batch_size, seq_len, num_features)
        """
        batch_size, seq_len, num_features = x.shape
        
        # Create random mask
        num_masked = int(seq_len * self.mask_ratio)
        mask_indices = torch.randperm(seq_len)[:num_masked]
        
        # Create masked input
        x_masked = x.clone()
        mask_token = torch.zeros(num_features, device=x.device)
        for idx in mask_indices:
            x_masked[:, idx, :] = mask_token
        
        # Forward pass
        output = model(x_masked, return_sequence=True)
        encoded = output['encoded']
        
        # Reconstruct only masked positions
        reconstructed = model.reconstruct(encoded)  # (B, L, num_features)
        
        # Compute loss only on masked positions
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
        for idx in mask_indices:
            mask[:, idx] = True
        
        loss = self.mse_loss(
            reconstructed[mask],
            x[mask]
        )
        
        return loss, {'pretrain_loss': loss.item()}


class MultiTaskLoss(nn.Module):
    """Multi-task loss for supervised fine-tuning"""
    
    def __init__(self,
                 weight_buy: float = 1.0,
                 weight_sell: float = 1.0,
                 weight_direction: float = 1.0,
                 weight_regime: float = 1.0):
        super().__init__()
        self.weight_buy = weight_buy
        self.weight_sell = weight_sell
        self.weight_direction = weight_direction
        self.weight_regime = weight_regime
        
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self,
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Dict with 'buy', 'sell', 'direction', 'regime'
            targets: Dict with same keys
        """
        # Buy/Sell: Binary classification
        buy_loss = self.bce_loss(predictions['buy'], targets['buy'].float())
        sell_loss = self.bce_loss(predictions['sell'], targets['sell'].float())
        
        # Direction: Binary classification (2 classes)
        direction_loss = self.ce_loss(predictions['direction'], targets['direction'])
        
        # Regime: Multi-class classification (6 classes)
        regime_loss = self.ce_loss(predictions['regime'], targets['regime'])
        
        # Weighted sum
        total_loss = (
            self.weight_buy * buy_loss +
            self.weight_sell * sell_loss +
            self.weight_direction * direction_loss +
            self.weight_regime * regime_loss
        )
        
        losses = {
            'total_loss': total_loss.item(),
            'buy_loss': buy_loss.item(),
            'sell_loss': sell_loss.item(),
            'direction_loss': direction_loss.item(),
            'regime_loss': regime_loss.item()
        }
        
        return total_loss, losses
