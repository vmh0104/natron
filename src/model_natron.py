"""
Natron Transformer - Model Architecture
Multi-task Transformer for financial trading
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class NatronTransformer(nn.Module):
    """
    Multi-Task Transformer for Financial Trading
    
    Architecture:
    - Input: (batch, 96, 100) feature sequences
    - Embedding: Project to d_model dimensions
    - Transformer Encoder: Learn temporal patterns
    - Multi-Task Heads:
        * Buy head (sigmoid)
        * Sell head (sigmoid)
        * Direction head (2-class softmax)
        * Regime head (6-class softmax)
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Model dimensions
        self.input_dim = config['model']['input_dim']
        self.d_model = config['model']['d_model']
        self.nhead = config['model']['nhead']
        self.num_layers = config['model']['num_encoder_layers']
        self.dim_feedforward = config['model']['dim_feedforward']
        self.dropout = config['model']['dropout']
        self.max_seq_length = config['model']['max_seq_length']
        
        # Input embedding
        self.input_projection = nn.Linear(self.input_dim, self.d_model)
        self.pos_encoder = PositionalEncoding(self.d_model, self.max_seq_length, self.dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        # Task-specific heads
        self.buy_head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.sell_head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        self.direction_head = nn.Sequential(
            nn.Linear(self.d_model, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 2)  # 2 classes: down/up
        )
        
        self.regime_head = nn.Sequential(
            nn.Linear(self.d_model, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 6)  # 6 regime classes
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            Dictionary with predictions for each task
        """
        # Input projection
        x = self.input_projection(x)  # (batch, 96, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoding
        encoded = self.transformer_encoder(x)  # (batch, 96, d_model)
        
        # Global pooling across sequence
        pooled = encoded.mean(dim=1)  # (batch, d_model)
        
        # Multi-task predictions
        buy_prob = self.buy_head(pooled).squeeze(-1)  # (batch,)
        sell_prob = self.sell_head(pooled).squeeze(-1)  # (batch,)
        direction_logits = self.direction_head(pooled)  # (batch, 2)
        regime_logits = self.regime_head(pooled)  # (batch, 6)
        
        return {
            'buy': buy_prob,
            'sell': sell_prob,
            'direction': direction_logits,
            'regime': regime_logits,
            'encoded': encoded  # For pretraining
        }
    
    def get_encoder_output(self, x: torch.Tensor) -> torch.Tensor:
        """Get encoder output (for pretraining)"""
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)
        return encoded
    
    def freeze_encoder(self):
        """Freeze encoder parameters (for transfer learning)"""
        for param in self.input_projection.parameters():
            param.requires_grad = False
        for param in self.pos_encoder.parameters():
            param.requires_grad = False
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False
        print("🔒 Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters"""
        for param in self.input_projection.parameters():
            param.requires_grad = True
        for param in self.pos_encoder.parameters():
            param.requires_grad = True
        for param in self.transformer_encoder.parameters():
            param.requires_grad = True
        print("🔓 Encoder unfrozen")


class MaskedLanguageModel(nn.Module):
    """
    Masked Language Model for Pretraining
    Reconstructs masked tokens in the sequence
    """
    
    def __init__(self, encoder: NatronTransformer, input_dim: int):
        super().__init__()
        self.encoder = encoder
        self.reconstruction_head = nn.Linear(encoder.d_model, input_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input sequence (batch, seq_len, input_dim)
            mask: Boolean mask (batch, seq_len) - True for masked positions
            
        Returns:
            Reconstructed sequence (batch, seq_len, input_dim)
        """
        encoded = self.encoder.get_encoder_output(x)
        reconstructed = self.reconstruction_head(encoded)
        return reconstructed


class ContrastiveProjection(nn.Module):
    """
    Projection head for contrastive learning
    """
    
    def __init__(self, d_model: int, projection_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Encoded representation (batch, d_model)
            
        Returns:
            Projected representation (batch, projection_dim)
        """
        return F.normalize(self.projection(x), dim=-1)


def create_model(config: Dict, device: str = 'cuda') -> NatronTransformer:
    """
    Create Natron Transformer model
    
    Args:
        config: Configuration dictionary
        device: Device to place model on
        
    Returns:
        NatronTransformer model
    """
    model = NatronTransformer(config).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"🧠 Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Device: {device}")
    
    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    path: str
):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)
    print(f"💾 Checkpoint saved: {path}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    path: str,
    device: str = 'cuda'
) -> Tuple[int, float]:
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"📂 Checkpoint loaded: {path}")
    print(f"  Epoch: {epoch}, Loss: {loss:.4f}")
    
    return epoch, loss


if __name__ == "__main__":
    # Test model
    print("🧪 Testing Natron Transformer...")
    
    import yaml
    
    # Load config
    with open('/workspace/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    model = create_model(config, device)
    
    # Test forward pass
    batch_size = 16
    seq_len = 96
    input_dim = 100
    
    x = torch.randn(batch_size, seq_len, input_dim).to(device)
    
    print(f"\n🔄 Testing forward pass...")
    print(f"  Input shape: {x.shape}")
    
    with torch.no_grad():
        outputs = model(x)
    
    print(f"\n📊 Output shapes:")
    print(f"  Buy: {outputs['buy'].shape}")
    print(f"  Sell: {outputs['sell'].shape}")
    print(f"  Direction: {outputs['direction'].shape}")
    print(f"  Regime: {outputs['regime'].shape}")
    print(f"  Encoded: {outputs['encoded'].shape}")
    
    # Test encoder freezing
    print(f"\n🔒 Testing encoder freezing...")
    trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.freeze_encoder()
    trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params before: {trainable_before:,}")
    print(f"  Trainable params after: {trainable_after:,}")
    
    print("\n✅ Model test successful!")
