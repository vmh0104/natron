"""
Natron Transformer - Multi-Task Financial Trading Model
Transformer architecture with multi-task heads for buy/sell, direction, and regime prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, Optional


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer"""
    
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


class TransformerEncoderLayer(nn.Module):
    """Standard Transformer Encoder Layer"""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = getattr(F, activation)
    
    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # Feedforward
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class NatronTransformer(nn.Module):
    """
    Natron Transformer Model for Multi-Task Financial Trading
    
    Architecture:
    - Input projection: (batch, 96, 100) -> (batch, 96, d_model)
    - Transformer Encoder: learns market representations
    - Multi-task heads: buy, sell, direction, regime
    """
    
    def __init__(
        self,
        num_features: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        sequence_length: int = 96,
        freeze_encoder: bool = False
    ):
        """
        Args:
            num_features: Number of input features per timestep
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            sequence_length: Input sequence length
            freeze_encoder: Whether to freeze encoder during fine-tuning
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.freeze_encoder = freeze_encoder
        
        # Input projection
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length + 10, dropout=dropout)
        
        # Transformer encoder layers
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Global pooling (use last timestep + mean pooling)
        self.pooling = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Multi-task heads
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
        
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)  # up/down
        )
        
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 6)  # 6 regime classes
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, x: torch.Tensor, return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: (batch_size, sequence_length, num_features)
            return_embeddings: Whether to return encoder embeddings
            
        Returns:
            Dictionary with predictions:
            - 'buy': (batch_size, 1)
            - 'sell': (batch_size, 1)
            - 'direction': (batch_size, 2) logits
            - 'regime': (batch_size, 6) logits
            - 'embeddings': (batch_size, sequence_length, d_model) if return_embeddings=True
        """
        batch_size = x.size(0)
        
        # Project input
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder layers
        if self.freeze_encoder:
            with torch.no_grad():
                for layer in self.transformer_encoder:
                    x = layer(x)
        else:
            for layer in self.transformer_encoder:
                x = layer(x)
        
        # Store embeddings if needed
        embeddings = x if return_embeddings else None
        
        # Global pooling: concatenate last timestep and mean
        last_timestep = x[:, -1, :]  # (batch, d_model)
        mean_pooled = x.mean(dim=1)  # (batch, d_model)
        pooled = torch.cat([last_timestep, mean_pooled], dim=1)  # (batch, d_model * 2)
        pooled = self.pooling(pooled)  # (batch, d_model)
        
        # Multi-task predictions
        buy_pred = self.buy_head(pooled)
        sell_pred = self.sell_head(pooled)
        direction_logits = self.direction_head(pooled)
        regime_logits = self.regime_head(pooled)
        
        outputs = {
            'buy': buy_pred,
            'sell': sell_pred,
            'direction': direction_logits,
            'regime': regime_logits
        }
        
        if return_embeddings:
            outputs['embeddings'] = embeddings
        
        return outputs
    
    def predict(self, x: torch.Tensor) -> Dict[str, any]:
        """
        Inference mode: returns human-readable predictions
        
        Args:
            x: (batch_size, sequence_length, num_features) or (sequence_length, num_features)
            
        Returns:
            Dictionary with predictions and probabilities
        """
        self.eval()
        
        # Handle single sample
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.forward(x)
        
        # Process outputs
        buy_prob = outputs['buy'].item() if x.size(0) == 1 else outputs['buy'].cpu().numpy()
        sell_prob = outputs['sell'].item() if x.size(0) == 1 else outputs['sell'].cpu().numpy()
        
        direction_probs = F.softmax(outputs['direction'], dim=-1)
        direction_up_prob = direction_probs[0, 1].item() if x.size(0) == 1 else direction_probs[:, 1].cpu().numpy()
        
        regime_probs = F.softmax(outputs['regime'], dim=-1)
        regime_idx = regime_probs.argmax(dim=-1).item() if x.size(0) == 1 else regime_probs.argmax(dim=-1).cpu().numpy()
        
        regime_names = ['BULL_STRONG', 'BULL_WEAK', 'RANGE', 'BEAR_WEAK', 'BEAR_STRONG', 'VOLATILE']
        regime_name = regime_names[regime_idx] if isinstance(regime_idx, (int, np.integer)) else [regime_names[i] for i in regime_idx]
        
        # Confidence (average of max probabilities)
        confidence = (buy_prob + (1 - sell_prob) + direction_up_prob + regime_probs.max(dim=-1)[0].item()) / 4
        
        result = {
            'buy_prob': float(buy_prob),
            'sell_prob': float(sell_prob),
            'direction_up': float(direction_up_prob),
            'regime': regime_name,
            'regime_probs': regime_probs[0].cpu().tolist() if x.size(0) == 1 else regime_probs.cpu().tolist(),
            'confidence': float(confidence)
        }
        
        return result


class NatronPretrainModel(nn.Module):
    """
    Pretraining model for Phase 1: Masked Modeling and Contrastive Learning
    """
    
    def __init__(
        self,
        num_features: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        sequence_length: int = 96,
        mask_ratio: float = 0.15
    ):
        super().__init__()
        
        self.d_model = d_model
        self.mask_ratio = mask_ratio
        
        # Encoder (same as main model)
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length + 10, dropout=dropout)
        
        self.transformer_encoder = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # Reconstruction head
        self.reconstruction_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, num_features)
        )
        
        # Contrastive projection head
        self.contrastive_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 128)  # Contrastive embedding dimension
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass for pretraining
        
        Args:
            x: (batch_size, sequence_length, num_features)
            mask: Optional boolean mask (batch_size, sequence_length)
            
        Returns:
            Dictionary with 'reconstruction' and 'contrastive_embedding'
        """
        batch_size, seq_len, num_features = x.size()
        
        # Create mask if not provided
        if mask is None:
            num_masked = int(seq_len * self.mask_ratio)
            mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
            for i in range(batch_size):
                masked_indices = torch.randperm(seq_len)[:num_masked]
                mask[i, masked_indices] = True
        
        # Mask input
        masked_x = x.clone()
        masked_x[mask] = 0  # Simple masking (could use learnable mask token)
        
        # Encode
        x_proj = self.input_projection(masked_x)
        x_proj = self.pos_encoder(x_proj)
        
        for layer in self.transformer_encoder:
            x_proj = layer(x_proj)
        
        # Reconstruction
        reconstruction = self.reconstruction_head(x_proj)
        
        # Contrastive embedding (use mean pooled)
        pooled = x_proj.mean(dim=1)
        contrastive_embedding = self.contrastive_head(pooled)
        contrastive_embedding = F.normalize(contrastive_embedding, p=2, dim=1)
        
        return {
            'reconstruction': reconstruction,
            'contrastive_embedding': contrastive_embedding,
            'mask': mask
        }
