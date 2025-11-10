"""
Natron Loss Functions
Multi-task loss combinations for training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
    Combined multi-task loss for Buy/Sell/Direction/Regime predictions.
    """
    
    def __init__(self, buy_weight: float = 1.0, sell_weight: float = 1.0,
                 direction_weight: float = 1.0, regime_weight: float = 1.0,
                 use_focal_loss: bool = False):
        super().__init__()
        self.buy_weight = buy_weight
        self.sell_weight = sell_weight
        self.direction_weight = direction_weight
        self.regime_weight = regime_weight
        self.use_focal_loss = use_focal_loss
        
        # Binary cross-entropy for buy/sell
        self.bce_loss = nn.BCELoss()
        
        # Cross-entropy for direction and regime
        self.ce_loss = nn.NLLLoss()
        
        # Focal loss (if enabled)
        if use_focal_loss:
            self.focal_loss_buy = FocalLoss(alpha=0.25, gamma=2.0)
            self.focal_loss_sell = FocalLoss(alpha=0.25, gamma=2.0)
    
    def forward(self, predictions: dict, targets: dict) -> dict:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Dict with 'buy', 'sell', 'direction', 'regime'
            targets: Dict with same keys
        
        Returns:
            Dict with individual and total losses
        """
        # Buy loss
        if self.use_focal_loss:
            buy_loss = self.focal_loss_buy(predictions['buy'], targets['buy'])
        else:
            buy_loss = self.bce_loss(predictions['buy'], targets['buy'])
        
        # Sell loss
        if self.use_focal_loss:
            sell_loss = self.focal_loss_sell(predictions['sell'], targets['sell'])
        else:
            sell_loss = self.bce_loss(predictions['sell'], targets['sell'])
        
        # Direction loss
        direction_loss = self.ce_loss(predictions['direction'], targets['direction'])
        
        # Regime loss
        regime_loss = self.ce_loss(predictions['regime'], targets['regime'])
        
        # Weighted combination
        total_loss = (
            self.buy_weight * buy_loss +
            self.sell_weight * sell_loss +
            self.direction_weight * direction_loss +
            self.regime_weight * regime_loss
        )
        
        return {
            'total': total_loss,
            'buy': buy_loss,
            'sell': sell_loss,
            'direction': direction_loss,
            'regime': regime_loss
        }


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


class PretrainLoss(nn.Module):
    """
    Loss for pretraining (masked reconstruction + contrastive).
    """
    
    def __init__(self, recon_weight: float = 1.0, contrastive_weight: float = 0.5,
                 temperature: float = 0.07):
        super().__init__()
        self.recon_weight = recon_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        self.mse_loss = nn.MSELoss()
    
    def forward(self, predictions: dict, targets: torch.Tensor) -> dict:
        """
        Compute pretraining loss.
        
        Args:
            predictions: Dict with 'reconstruction' and 'projection'
            targets: Original features (batch, seq_len, input_dim)
        
        Returns:
            Dict with reconstruction and contrastive losses
        """
        reconstruction = predictions['reconstruction']
        mask_tokens = predictions['mask_tokens']
        
        # Reconstruction loss (only on masked tokens)
        if reconstruction is not None:
            recon_loss = self.mse_loss(
                reconstruction[mask_tokens],
                targets[mask_tokens]
            )
        else:
            recon_loss = torch.tensor(0.0, device=targets.device)
        
        # Contrastive loss (InfoNCE)
        projection = predictions['projection']
        contrastive_loss = self._infonce_loss(projection)
        
        total_loss = self.recon_weight * recon_loss + self.contrastive_weight * contrastive_loss
        
        return {
            'total': total_loss,
            'reconstruction': recon_loss,
            'contrastive': contrastive_loss
        }
    
    def _infonce_loss(self, projections: torch.Tensor) -> torch.Tensor:
        """
        InfoNCE contrastive loss.
        
        Args:
            projections: (batch, d_model) projected features
        
        Returns:
            Contrastive loss scalar
        """
        batch_size = projections.shape[0]
        
        # Normalize projections
        projections = F.normalize(projections, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(projections, projections.T) / self.temperature
        
        # Positive pairs are diagonal (same sample augmented)
        # Negative pairs are off-diagonal
        labels = torch.arange(batch_size, device=projections.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss
