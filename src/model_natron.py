"""
Natron Transformer - Multi-Task Financial Trading Model
Architecture: Transformer Encoder + Multi-Task Heads

Tasks:
- Buy prediction (binary classification)
- Sell prediction (binary classification)
- Direction prediction (binary classification)
- Regime classification (6 classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
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


class TaskHead(nn.Module):
    """Generic task-specific prediction head"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        output_dim: int,
        dropout: float = 0.1,
        activation: str = "relu"
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class NatronTransformer(nn.Module):
    """
    Natron Multi-Task Transformer Model
    
    Architecture:
    1. Input Projection (num_features -> d_model)
    2. Positional Encoding
    3. Transformer Encoder
    4. Multi-Task Heads:
       - Buy Head (sigmoid)
       - Sell Head (sigmoid)
       - Direction Head (softmax)
       - Regime Head (softmax)
    """
    
    def __init__(
        self,
        num_features: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        buy_head_dims: list = [128, 64],
        sell_head_dims: list = [128, 64],
        direction_head_dims: list = [128, 64],
        regime_head_dims: list = [128, 64],
        sequence_length: int = 96
    ):
        super().__init__()
        
        self.num_features = num_features
        self.d_model = d_model
        self.sequence_length = sequence_length
        
        # Input projection
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length, dropout=dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=False,  # (seq_len, batch, d_model)
            norm_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Global pooling
        self.pooling_type = "attention"  # Options: "last", "mean", "max", "attention"
        
        if self.pooling_type == "attention":
            self.attention_pooling = nn.Sequential(
                nn.Linear(d_model, d_model // 4),
                nn.Tanh(),
                nn.Linear(d_model // 4, 1)
            )
        
        # Task-specific heads
        self.buy_head = TaskHead(
            d_model, buy_head_dims, output_dim=1,
            dropout=dropout, activation=activation
        )
        
        self.sell_head = TaskHead(
            d_model, sell_head_dims, output_dim=1,
            dropout=dropout, activation=activation
        )
        
        self.direction_head = TaskHead(
            d_model, direction_head_dims, output_dim=2,  # Binary: up/down
            dropout=dropout, activation=activation
        )
        
        self.regime_head = TaskHead(
            d_model, regime_head_dims, output_dim=6,  # 6 regime classes
            dropout=dropout, activation=activation
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        x: torch.Tensor,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, num_features)
            return_embeddings: If True, return encoder embeddings
            
        Returns:
            Dictionary with task outputs:
            - buy_logits: (batch_size, 1)
            - sell_logits: (batch_size, 1)
            - direction_logits: (batch_size, 2)
            - regime_logits: (batch_size, 6)
            - embeddings: (batch_size, d_model) [optional]
        """
        batch_size, seq_len, _ = x.shape
        
        # Input projection: (batch, seq, features) -> (batch, seq, d_model)
        x = self.input_projection(x)
        
        # Transpose for transformer: (batch, seq, d_model) -> (seq, batch, d_model)
        x = x.transpose(0, 1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        encoded = self.transformer_encoder(x)  # (seq, batch, d_model)
        
        # Global pooling
        pooled = self._pool_sequence(encoded)  # (batch, d_model)
        
        # Task-specific predictions
        buy_logits = self.buy_head(pooled)  # (batch, 1)
        sell_logits = self.sell_head(pooled)  # (batch, 1)
        direction_logits = self.direction_head(pooled)  # (batch, 2)
        regime_logits = self.regime_head(pooled)  # (batch, 6)
        
        output = {
            'buy_logits': buy_logits,
            'sell_logits': sell_logits,
            'direction_logits': direction_logits,
            'regime_logits': regime_logits
        }
        
        if return_embeddings:
            output['embeddings'] = pooled
        
        return output
    
    def _pool_sequence(self, encoded: torch.Tensor) -> torch.Tensor:
        """
        Pool sequence to single vector
        
        Args:
            encoded: (seq_len, batch, d_model)
            
        Returns:
            pooled: (batch, d_model)
        """
        # Transpose to (batch, seq, d_model)
        encoded = encoded.transpose(0, 1)
        
        if self.pooling_type == "last":
            # Use last token
            pooled = encoded[:, -1, :]
        
        elif self.pooling_type == "mean":
            # Mean pooling
            pooled = encoded.mean(dim=1)
        
        elif self.pooling_type == "max":
            # Max pooling
            pooled = encoded.max(dim=1)[0]
        
        elif self.pooling_type == "attention":
            # Attention-based pooling
            attention_weights = self.attention_pooling(encoded)  # (batch, seq, 1)
            attention_weights = F.softmax(attention_weights, dim=1)
            pooled = (encoded * attention_weights).sum(dim=1)  # (batch, d_model)
        
        else:
            pooled = encoded[:, -1, :]  # Default to last
        
        return pooled
    
    def get_predictions(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get final predictions with probabilities
        
        Args:
            x: Input tensor (batch_size, seq_len, num_features)
            
        Returns:
            Dictionary with predictions:
            - buy_prob: (batch_size,)
            - sell_prob: (batch_size,)
            - direction_prob: (batch_size, 2)
            - regime_prob: (batch_size, 6)
            - direction_pred: (batch_size,)
            - regime_pred: (batch_size,)
        """
        with torch.no_grad():
            outputs = self.forward(x)
            
            predictions = {
                'buy_prob': torch.sigmoid(outputs['buy_logits']).squeeze(-1),
                'sell_prob': torch.sigmoid(outputs['sell_logits']).squeeze(-1),
                'direction_prob': F.softmax(outputs['direction_logits'], dim=-1),
                'regime_prob': F.softmax(outputs['regime_logits'], dim=-1),
                'direction_pred': outputs['direction_logits'].argmax(dim=-1),
                'regime_pred': outputs['regime_logits'].argmax(dim=-1)
            }
        
        return predictions
    
    def freeze_encoder(self):
        """Freeze transformer encoder weights"""
        for param in self.input_projection.parameters():
            param.requires_grad = False
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False
        print("🔒 Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze transformer encoder weights"""
        for param in self.input_projection.parameters():
            param.requires_grad = True
        for param in self.transformer_encoder.parameters():
            param.requires_grad = True
        print("🔓 Encoder unfrozen")


class NatronPretrainModel(nn.Module):
    """
    Natron Pretraining Model
    Tasks: Masked token reconstruction + Contrastive learning
    """
    
    def __init__(
        self,
        num_features: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        activation: str = "gelu",
        sequence_length: int = 96
    ):
        super().__init__()
        
        self.num_features = num_features
        self.d_model = d_model
        self.sequence_length = sequence_length
        
        # Input projection
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length, dropout=dropout)
        
        # Transformer encoder (shared backbone)
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
            num_layers=num_encoder_layers
        )
        
        # Reconstruction head
        self.reconstruction_head = nn.Linear(d_model, num_features)
        
        # Contrastive projection head
        self.contrastive_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 128)  # Projection dimension
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for pretraining
        
        Args:
            x: Input (batch, seq, features)
            mask: Boolean mask for masked tokens (batch, seq)
            
        Returns:
            - reconstructed: (batch, seq, features)
            - contrastive_emb: (batch, 128)
            - embeddings: (seq, batch, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        x = x.transpose(0, 1)  # (seq, batch, d_model)
        
        # Positional encoding
        x = self.pos_encoder(x)
        
        # Encode
        encoded = self.transformer_encoder(x)  # (seq, batch, d_model)
        
        # Reconstruction
        reconstructed = self.reconstruction_head(encoded)  # (seq, batch, features)
        reconstructed = reconstructed.transpose(0, 1)  # (batch, seq, features)
        
        # Contrastive embedding (use mean pooling)
        pooled = encoded.mean(dim=0)  # (batch, d_model)
        contrastive_emb = self.contrastive_head(pooled)  # (batch, 128)
        contrastive_emb = F.normalize(contrastive_emb, dim=-1)  # L2 normalize
        
        return {
            'reconstructed': reconstructed,
            'contrastive_emb': contrastive_emb,
            'embeddings': encoded
        }
    
    def transfer_to_natron(self, natron_model: NatronTransformer):
        """Transfer pretrained weights to NatronTransformer"""
        print("🔄 Transferring pretrained weights to Natron model...")
        
        # Transfer input projection
        natron_model.input_projection.load_state_dict(
            self.input_projection.state_dict()
        )
        
        # Transfer transformer encoder
        natron_model.transformer_encoder.load_state_dict(
            self.transformer_encoder.state_dict()
        )
        
        print("✅ Weight transfer complete!")


if __name__ == "__main__":
    # Test model
    print("Testing Natron Transformer...")
    
    # Model parameters
    batch_size = 16
    seq_len = 96
    num_features = 100
    
    # Create model
    model = NatronTransformer(
        num_features=num_features,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    x = torch.randn(batch_size, seq_len, num_features)
    outputs = model(x)
    
    print("\n✅ Model outputs:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")
    
    # Test predictions
    predictions = model.get_predictions(x)
    print("\n✅ Predictions:")
    for key, value in predictions.items():
        print(f"  {key}: {value.shape}")
    
    # Test pretrain model
    print("\n\nTesting Pretrain Model...")
    pretrain_model = NatronPretrainModel(
        num_features=num_features,
        d_model=256,
        nhead=8,
        num_encoder_layers=6
    )
    
    pretrain_outputs = pretrain_model(x)
    print("\n✅ Pretrain outputs:")
    for key, value in pretrain_outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
