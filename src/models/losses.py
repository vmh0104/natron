"""
Natron V2 Loss Functions
Multi-task and pretraining loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for supervised training.
    Combines losses from buy/sell, direction, and regime predictions.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        # Loss weights from config
        self.buy_weight = config['loss_weights']['buy']
        self.sell_weight = config['loss_weights']['sell']
        self.direction_weight = config['loss_weights']['direction']
        self.regime_weight = config['loss_weights']['regime']
        
        # Individual loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        
        # Focal loss parameters (optional, for class imbalance)
        self.use_focal = True
        self.focal_alpha = 0.25
        self.focal_gamma = 2.0
    
    def forward(self, predictions: dict, labels: dict) -> dict:
        """
        Calculate multi-task loss.
        
        Args:
            predictions: dict with keys ['buy_logits', 'sell_logits', 
                         'direction_logits', 'regime_logits']
            labels: dict with keys ['buy', 'sell', 'direction', 'regime']
            
        Returns:
            dict with individual losses and total loss
        """
        # Buy loss (BCE)
        buy_loss = self.bce_loss(predictions['buy_logits'], labels['buy'])
        
        # Sell loss (BCE)
        sell_loss = self.bce_loss(predictions['sell_logits'], labels['sell'])
        
        # Direction loss (CE)
        direction_loss = self.ce_loss(predictions['direction_logits'], labels['direction'])
        
        # Regime loss (CE with optional focal loss)
        if self.use_focal:
            regime_loss = self.focal_loss(predictions['regime_logits'], labels['regime'])
        else:
            regime_loss = self.ce_loss(predictions['regime_logits'], labels['regime'])
        
        # Weighted total loss
        total_loss = (self.buy_weight * buy_loss +
                     self.sell_weight * sell_loss +
                     self.direction_weight * direction_loss +
                     self.regime_weight * regime_loss)
        
        return {
            'total': total_loss,
            'buy': buy_loss,
            'sell': sell_loss,
            'direction': direction_loss,
            'regime': regime_loss
        }
    
    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Focal loss for handling class imbalance.
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.focal_alpha * (1 - pt) ** self.focal_gamma * ce_loss
        return focal_loss.mean()


class ReconstructionLoss(nn.Module):
    """
    Reconstruction loss for masked modeling pretraining.
    Only computes loss on masked tokens.
    """
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            reconstructed: (batch, seq_len, features)
            original: (batch, seq_len, features)
            mask: (batch, seq_len) boolean mask indicating masked positions
        """
        # Compute MSE loss
        loss = self.mse_loss(reconstructed, original)  # (batch, seq_len, features)
        
        # Average over features
        loss = loss.mean(dim=-1)  # (batch, seq_len)
        
        # Only compute loss on masked positions
        mask = mask.float()
        masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        
        return masked_loss


class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for self-supervised pretraining.
    Creates positive pairs through data augmentation.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: Projections from augmented view 1 (batch, dim)
            z2: Projections from augmented view 2 (batch, dim)
            
        Both should be L2-normalized.
        """
        batch_size = z1.shape[0]
        
        # Concatenate augmentations
        z = torch.cat([z1, z2], dim=0)  # (2*batch, dim)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z, z.t()) / self.temperature  # (2*batch, 2*batch)
        
        # Create labels (positive pairs are separated by batch_size)
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels])  # (2*batch,)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim_matrix.masked_fill_(mask, float('-inf'))
        
        # InfoNCE loss
        loss = F.cross_entropy(sim_matrix, labels)
        
        return loss


class PretrainLoss(nn.Module):
    """
    Combined pretraining loss: reconstruction + contrastive.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        self.reconstruction_loss = ReconstructionLoss()
        self.contrastive_loss = ContrastiveLoss(
            temperature=config['pretrain']['contrastive_temperature']
        )
        
        # Loss weights
        self.recon_weight = 1.0
        self.contrastive_weight = 0.5
    
    def forward(self, outputs: dict, batch: dict, mode: str = 'reconstruction') -> dict:
        """
        Calculate pretraining loss.
        
        Args:
            outputs: Model outputs
            batch: Batch data
            mode: 'reconstruction' or 'contrastive'
        """
        if mode == 'reconstruction':
            loss = self.reconstruction_loss(
                outputs['reconstructed'],
                batch['original'],
                batch['mask']
            )
            return {'total': loss, 'reconstruction': loss}
        
        elif mode == 'contrastive':
            # Assume we have two augmented views
            loss = self.contrastive_loss(outputs['projected'], outputs['projected_aug'])
            return {'total': loss, 'contrastive': loss}
        
        return {'total': torch.tensor(0.0)}


class TradingLoss(nn.Module):
    """
    Custom trading-specific loss that considers market dynamics.
    Penalizes incorrect predictions more heavily during volatile periods.
    """
    
    def __init__(self, config: dict):
        super().__init__()
        self.base_loss = MultiTaskLoss(config)
    
    def forward(self, predictions: dict, labels: dict, 
                volatility: torch.Tensor = None) -> dict:
        """
        Trading loss with volatility weighting.
        
        Args:
            predictions: Model predictions
            labels: Ground truth labels
            volatility: Optional volatility measure (batch,)
        """
        # Get base multi-task loss
        losses = self.base_loss(predictions, labels)
        
        # If volatility is provided, weight the loss
        if volatility is not None:
            # Higher weight for volatile periods (harder to predict)
            volatility_weight = 1.0 + volatility
            losses['total'] = losses['total'] * volatility_weight.mean()
        
        return losses
