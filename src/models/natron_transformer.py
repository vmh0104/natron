"""
Natron Transformer - Multi-Task Financial Trading Model
Author: Natron AI System
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class NatronTransformer(nn.Module):
    """
    Natron Transformer for Multi-Task Financial Trading.
    
    Architecture:
        Input (96, 100) → Embedding → Positional Encoding → 
        Transformer Encoder → Multi-Task Heads
    
    Outputs:
        - buy_prob: Buy signal probability
        - sell_prob: Sell signal probability
        - direction: Direction classification (up/down)
        - regime: Market regime (6 classes)
    """
    
    def __init__(self,
                 num_features: int = 100,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 activation: str = 'gelu',
                 max_seq_len: int = 96):
        super().__init__()
        
        self.num_features = num_features
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Input embedding
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Global pooling
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Task-specific heads
        # Buy head
        self.buy_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Sell head
        self.sell_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Direction head (binary classification)
        self.direction_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )
        
        # Regime head (6-class classification)
        self.regime_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 6)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            return_embeddings: If True, also return encoder embeddings
            
        Returns:
            Dictionary with predictions and optionally embeddings
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to d_model
        x = self.input_projection(x)  # (batch, seq, d_model)
        
        # Transpose for transformer (seq, batch, d_model)
        x = x.permute(1, 0, 2)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        encoded = self.transformer_encoder(x)  # (seq, batch, d_model)
        
        # Global average pooling over sequence
        # Transpose to (batch, d_model, seq)
        pooled = encoded.permute(1, 2, 0)
        pooled = self.pooling(pooled).squeeze(-1)  # (batch, d_model)
        
        # Multi-task predictions
        buy_prob = self.buy_head(pooled)  # (batch, 1)
        sell_prob = self.sell_head(pooled)  # (batch, 1)
        direction_logits = self.direction_head(pooled)  # (batch, 2)
        regime_logits = self.regime_head(pooled)  # (batch, 6)
        
        outputs = {
            'buy_prob': buy_prob.squeeze(-1),
            'sell_prob': sell_prob.squeeze(-1),
            'direction_logits': direction_logits,
            'regime_logits': regime_logits
        }
        
        if return_embeddings:
            outputs['embeddings'] = pooled
            outputs['sequence_embeddings'] = encoded.permute(1, 0, 2)  # (batch, seq, d_model)
        
        return outputs
    
    def get_encoder(self) -> nn.Module:
        """Get encoder part of the model (for pretraining)"""
        return nn.Sequential(
            self.input_projection,
            self.pos_encoder,
            self.transformer_encoder
        )
    
    def freeze_encoder(self):
        """Freeze encoder parameters"""
        for param in self.input_projection.parameters():
            param.requires_grad = False
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False
        print("🔒 Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters"""
        for param in self.input_projection.parameters():
            param.requires_grad = True
        for param in self.transformer_encoder.parameters():
            param.requires_grad = True
        print("🔓 Encoder unfrozen")


class NatronPretrainModel(nn.Module):
    """
    Natron model for unsupervised pretraining.
    Supports masked modeling and contrastive learning.
    """
    
    def __init__(self,
                 num_features: int = 100,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_encoder_layers: int = 6,
                 dim_feedforward: int = 1024,
                 dropout: float = 0.1,
                 activation: str = 'gelu',
                 max_seq_len: int = 96):
        super().__init__()
        
        self.num_features = num_features
        self.d_model = d_model
        
        # Encoder (shared with main model)
        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # Reconstruction head (for masked modeling)
        self.reconstruction_head = nn.Linear(d_model, num_features)
        
        # Projection head (for contrastive learning)
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Linear(512, 128)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward_encoder(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode input sequence.
        
        Args:
            x: Input tensor (batch, seq, features)
            mask: Optional mask for attention
            
        Returns:
            Encoded tensor (batch, seq, d_model)
        """
        # Project to d_model
        x = self.input_projection(x)
        
        # Transpose for transformer
        x = x.permute(1, 0, 2)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Encode
        encoded = self.transformer_encoder(x, src_key_padding_mask=mask)
        
        # Transpose back
        encoded = encoded.permute(1, 0, 2)
        
        return encoded
    
    def forward_masked(self, x: torch.Tensor, mask_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for masked modeling.
        
        Args:
            x: Input with some features masked
            mask_indices: Boolean tensor indicating masked positions
            
        Returns:
            Reconstructed features
        """
        encoded = self.forward_encoder(x)
        reconstructed = self.reconstruction_head(encoded)
        return reconstructed
    
    def forward_contrastive(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for contrastive learning.
        
        Args:
            x: Input tensor
            
        Returns:
            Projected embeddings for contrastive loss
        """
        # Encode
        encoded = self.forward_encoder(x)
        
        # Global average pooling
        pooled = encoded.mean(dim=1)  # (batch, d_model)
        
        # Project
        projected = self.projection_head(pooled)
        
        # L2 normalize
        projected = F.normalize(projected, p=2, dim=1)
        
        return projected
    
    def transfer_to_supervised(self) -> NatronTransformer:
        """
        Transfer pretrained encoder to supervised model.
        
        Returns:
            NatronTransformer with pretrained encoder
        """
        supervised_model = NatronTransformer(
            num_features=self.num_features,
            d_model=self.d_model
        )
        
        # Copy encoder weights
        supervised_model.input_projection.load_state_dict(self.input_projection.state_dict())
        supervised_model.pos_encoder.load_state_dict(self.pos_encoder.state_dict())
        supervised_model.transformer_encoder.load_state_dict(self.transformer_encoder.state_dict())
        
        print("✅ Transferred pretrained weights to supervised model")
        
        return supervised_model


if __name__ == "__main__":
    print("🧪 Testing Natron Transformer...")
    
    # Test supervised model
    model = NatronTransformer(
        num_features=100,
        d_model=256,
        nhead=8,
        num_encoder_layers=6
    )
    
    # Create dummy input
    batch_size = 8
    seq_len = 96
    num_features = 100
    
    x = torch.randn(batch_size, seq_len, num_features)
    
    # Forward pass
    outputs = model(x, return_embeddings=True)
    
    print(f"✅ Model created successfully")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"\n📊 Output shapes:")
    print(f"   buy_prob: {outputs['buy_prob'].shape}")
    print(f"   sell_prob: {outputs['sell_prob'].shape}")
    print(f"   direction_logits: {outputs['direction_logits'].shape}")
    print(f"   regime_logits: {outputs['regime_logits'].shape}")
    print(f"   embeddings: {outputs['embeddings'].shape}")
    
    # Test pretrain model
    print(f"\n🧪 Testing Pretrain Model...")
    pretrain_model = NatronPretrainModel(num_features=100, d_model=256)
    
    # Test contrastive
    projected = pretrain_model.forward_contrastive(x)
    print(f"   Contrastive projection: {projected.shape}")
    
    # Test masked
    reconstructed = pretrain_model.forward_masked(x, None)
    print(f"   Masked reconstruction: {reconstructed.shape}")
    
    print(f"\n✅ All tests passed!")
