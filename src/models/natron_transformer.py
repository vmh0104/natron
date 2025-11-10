"""
Natron V2 Transformer Model
Multi-task Transformer for financial trading predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer.
    Injects information about position in sequence.
    """
    
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
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class NatronTransformer(nn.Module):
    """
    Multi-task Transformer model for financial trading.
    
    Architecture:
    - Input projection layer
    - Positional encoding
    - Transformer encoder
    - Multiple prediction heads:
        * Buy probability (sigmoid)
        * Sell probability (sigmoid)
        * Direction (2-class softmax)
        * Market regime (6-class softmax)
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        self.num_features = config['features']['num_features']
        self.d_model = config['model']['d_model']
        self.nhead = config['model']['nhead']
        self.num_layers = config['model']['num_encoder_layers']
        self.dim_feedforward = config['model']['dim_feedforward']
        self.dropout = config['model']['dropout']
        self.max_seq_length = config['model']['max_seq_length']
        
        # Input projection
        self.input_projection = nn.Linear(self.num_features, self.d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(self.d_model, self.max_seq_length, self.dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Global pooling (use last token + average)
        self.pooling = 'hybrid'  # 'last', 'mean', 'hybrid'
        
        # Prediction heads
        hidden_dim = self.d_model if self.pooling != 'hybrid' else self.d_model * 2
        
        # Buy head (binary sigmoid)
        self.buy_head = nn.Sequential(
            nn.Linear(hidden_dim, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 1)
        )
        
        # Sell head (binary sigmoid)
        self.sell_head = nn.Sequential(
            nn.Linear(hidden_dim, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 1)
        )
        
        # Direction head (2 classes)
        self.direction_head = nn.Sequential(
            nn.Linear(hidden_dim, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 2)
        )
        
        # Regime head (6 classes)
        self.regime_head = nn.Sequential(
            nn.Linear(hidden_dim, self.d_model // 2),
            nn.LayerNorm(self.d_model // 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model // 2, 6)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: torch.Tensor, return_embeddings: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, seq_len, num_features)
            return_embeddings: If True, return encoder embeddings
            
        Returns:
            Dictionary with predictions for each task
        """
        # Input projection: (batch, seq_len, num_features) -> (batch, seq_len, d_model)
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        encoder_output = self.transformer_encoder(x)  # (batch, seq_len, d_model)
        
        # Pooling
        if self.pooling == 'last':
            pooled = encoder_output[:, -1, :]  # Last token
        elif self.pooling == 'mean':
            pooled = encoder_output.mean(dim=1)  # Average pooling
        elif self.pooling == 'hybrid':
            last_token = encoder_output[:, -1, :]
            mean_pool = encoder_output.mean(dim=1)
            pooled = torch.cat([last_token, mean_pool], dim=-1)
        
        # Multi-task predictions
        buy_logits = self.buy_head(pooled).squeeze(-1)  # (batch,)
        sell_logits = self.sell_head(pooled).squeeze(-1)  # (batch,)
        direction_logits = self.direction_head(pooled)  # (batch, 2)
        regime_logits = self.regime_head(pooled)  # (batch, 6)
        
        outputs = {
            'buy_logits': buy_logits,
            'sell_logits': sell_logits,
            'direction_logits': direction_logits,
            'regime_logits': regime_logits
        }
        
        if return_embeddings:
            outputs['embeddings'] = encoder_output
            outputs['pooled'] = pooled
        
        return outputs
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict with probabilities (for inference).
        
        Returns:
            Dictionary with probabilities and class predictions
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            # Convert logits to probabilities
            buy_prob = torch.sigmoid(outputs['buy_logits'])
            sell_prob = torch.sigmoid(outputs['sell_logits'])
            direction_prob = F.softmax(outputs['direction_logits'], dim=-1)
            regime_prob = F.softmax(outputs['regime_logits'], dim=-1)
            
            # Get class predictions
            direction_pred = direction_prob.argmax(dim=-1)
            regime_pred = regime_prob.argmax(dim=-1)
            
            predictions = {
                'buy_prob': buy_prob,
                'sell_prob': sell_prob,
                'direction_prob': direction_prob[:, 1],  # Probability of up
                'direction_class': direction_pred,
                'regime_prob': regime_prob,
                'regime_class': regime_pred,
                'confidence': self._calculate_confidence(buy_prob, sell_prob, 
                                                        direction_prob, regime_prob)
            }
            
            return predictions
    
    def _calculate_confidence(self, buy_prob, sell_prob, direction_prob, regime_prob):
        """
        Calculate overall confidence score.
        Based on prediction certainty across all tasks.
        """
        # Direction confidence (max probability)
        direction_conf = direction_prob.max(dim=-1)[0]
        
        # Regime confidence (max probability)
        regime_conf = regime_prob.max(dim=-1)[0]
        
        # Buy/Sell confidence (distance from 0.5)
        buy_conf = torch.abs(buy_prob - 0.5) * 2
        sell_conf = torch.abs(sell_prob - 0.5) * 2
        
        # Weighted average
        confidence = (direction_conf * 0.3 + regime_conf * 0.4 + 
                     (buy_conf + sell_conf) / 2 * 0.3)
        
        return confidence
    
    def get_encoder(self):
        """Return encoder for pretraining"""
        return nn.Sequential(
            self.input_projection,
            self.pos_encoder,
            self.transformer_encoder
        )
    
    def freeze_encoder(self):
        """Freeze encoder weights (for fine-tuning)"""
        for param in self.input_projection.parameters():
            param.requires_grad = False
        for param in self.transformer_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder weights"""
        for param in self.input_projection.parameters():
            param.requires_grad = True
        for param in self.transformer_encoder.parameters():
            param.requires_grad = True


class PretrainTransformer(nn.Module):
    """
    Transformer for unsupervised pretraining.
    Supports masked modeling and contrastive learning.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.config = config
        self.num_features = config['features']['num_features']
        self.d_model = config['model']['d_model']
        
        # Shared encoder (same as main model)
        self.encoder = nn.Sequential()
        self.encoder.add_module('input_proj', nn.Linear(self.num_features, self.d_model))
        self.encoder.add_module('pos_enc', PositionalEncoding(self.d_model, 
                                                              config['model']['max_seq_length'],
                                                              config['model']['dropout']))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=config['model']['nhead'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout'],
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.encoder.add_module('transformer', 
                               nn.TransformerEncoder(encoder_layer, 
                                                    num_layers=config['model']['num_encoder_layers']))
        
        # Reconstruction head (for masked modeling)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.num_features)
        )
        
        # Projection head (for contrastive learning)
        self.projection_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 128)
        )
    
    def forward(self, x: torch.Tensor, mode: str = 'reconstruction') -> Dict[str, torch.Tensor]:
        """
        Forward pass for pretraining.
        
        Args:
            x: Input tensor (batch, seq_len, features)
            mode: 'reconstruction' or 'contrastive'
        """
        # Encode
        encoded = self.encoder(x)  # (batch, seq_len, d_model)
        
        if mode == 'reconstruction':
            # Reconstruct original input
            reconstructed = self.reconstruction_head(encoded)
            return {'reconstructed': reconstructed, 'encoded': encoded}
        
        elif mode == 'contrastive':
            # Global representation (mean pooling)
            pooled = encoded.mean(dim=1)  # (batch, d_model)
            projected = self.projection_head(pooled)  # (batch, 128)
            # L2 normalize for contrastive loss
            projected = F.normalize(projected, p=2, dim=-1)
            return {'projected': projected, 'encoded': encoded}
        
        return {'encoded': encoded}
    
    def transfer_weights_to(self, target_model: NatronTransformer):
        """
        Transfer pretrained encoder weights to supervised model.
        """
        # Transfer input projection
        target_model.input_projection.load_state_dict(
            self.encoder.input_proj.state_dict()
        )
        
        # Transfer transformer encoder
        target_model.transformer_encoder.load_state_dict(
            self.encoder.transformer.state_dict()
        )
        
        print("Pretrained weights transferred successfully!")
