"""
Natron Loss Functions
Author: Natron AI System
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss for supervised training.
    Combines buy, sell, direction, and regime losses with weights.
    """
    
    def __init__(self, 
                 task_weights: Dict[str, float] = None,
                 class_weights: Dict[str, torch.Tensor] = None):
        """
        Args:
            task_weights: Weights for each task {buy, sell, direction, regime}
            class_weights: Class weights for imbalanced datasets
        """
        super().__init__()
        
        self.task_weights = task_weights or {
            'buy': 1.0,
            'sell': 1.0,
            'direction': 1.5,
            'regime': 1.2
        }
        
        # Loss functions
        self.bce_loss = nn.BCELoss()
        
        # Direction loss (binary classification)
        self.direction_loss_fn = nn.CrossEntropyLoss(
            weight=class_weights.get('direction') if class_weights else None
        )
        
        # Regime loss (6-class classification)
        self.regime_loss_fn = nn.CrossEntropyLoss(
            weight=class_weights.get('regime') if class_weights else None
        )
    
    def forward(self, 
                predictions: Dict[str, torch.Tensor],
                targets: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate multi-task loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
            
        Returns:
            Total loss and individual task losses
        """
        # Buy loss
        buy_loss = self.bce_loss(predictions['buy_prob'], targets['buy'].float())
        
        # Sell loss
        sell_loss = self.bce_loss(predictions['sell_prob'], targets['sell'].float())
        
        # Direction loss
        direction_loss = self.direction_loss_fn(
            predictions['direction_logits'], 
            targets['direction']
        )
        
        # Regime loss
        regime_loss = self.regime_loss_fn(
            predictions['regime_logits'],
            targets['regime']
        )
        
        # Weighted combination
        total_loss = (
            self.task_weights['buy'] * buy_loss +
            self.task_weights['sell'] * sell_loss +
            self.task_weights['direction'] * direction_loss +
            self.task_weights['regime'] * regime_loss
        )
        
        # Individual losses for logging
        losses = {
            'total': total_loss.item(),
            'buy': buy_loss.item(),
            'sell': sell_loss.item(),
            'direction': direction_loss.item(),
            'regime': regime_loss.item()
        }
        
        return total_loss, losses


class MaskedModelingLoss(nn.Module):
    """
    Loss for masked modeling pretraining.
    Reconstructs masked features.
    """
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(self,
                predictions: torch.Tensor,
                targets: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Calculate masked reconstruction loss.
        
        Args:
            predictions: Reconstructed features (batch, seq, features)
            targets: Original features
            mask: Boolean mask (True = masked position)
            
        Returns:
            MSE loss on masked positions only
        """
        # Only compute loss on masked positions
        masked_predictions = predictions[mask]
        masked_targets = targets[mask]
        
        loss = self.mse_loss(masked_predictions, masked_targets)
        
        return loss


class ContrastiveLoss(nn.Module):
    """
    NT-Xent (InfoNCE) loss for contrastive learning.
    Used in SimCLR-style pretraining.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Calculate contrastive loss between two views.
        
        Args:
            z_i: Embeddings from view 1 (batch, embedding_dim)
            z_j: Embeddings from view 2 (batch, embedding_dim)
            
        Returns:
            Contrastive loss
        """
        batch_size = z_i.shape[0]
        
        # Normalize embeddings
        z_i = F.normalize(z_i, p=2, dim=1)
        z_j = F.normalize(z_j, p=2, dim=1)
        
        # Concatenate both views
        representations = torch.cat([z_i, z_j], dim=0)  # (2*batch, dim)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(representations, representations.t())  # (2*batch, 2*batch)
        
        # Create positive pairs mask
        # Positive pairs are (i, i+batch) and (i+batch, i)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)
        mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = True
        mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = True
        
        # Remove self-similarities
        similarity_matrix = similarity_matrix[~torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)].reshape(2 * batch_size, -1)
        
        # Get positive pairs
        positives = similarity_matrix[mask[~torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)].reshape(2 * batch_size, -1)]
        positives = positives.reshape(2 * batch_size, 1)
        
        # Get negative pairs
        negatives = similarity_matrix[~mask[~torch.eye(2 * batch_size, dtype=torch.bool, device=z_i.device)].reshape(2 * batch_size, -1)]
        negatives = negatives.reshape(2 * batch_size, -1)
        
        # Compute logits
        logits = torch.cat([positives, negatives], dim=1) / self.temperature
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)
        
        # Cross entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Calculate focal loss.
        
        Args:
            inputs: Predictions (logits or probabilities)
            targets: Ground truth labels
            
        Returns:
            Focal loss
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class HybridPretrainLoss(nn.Module):
    """
    Hybrid loss combining masked modeling and contrastive learning.
    """
    
    def __init__(self, 
                 masked_weight: float = 0.5,
                 contrastive_weight: float = 0.5,
                 temperature: float = 0.07):
        super().__init__()
        
        self.masked_weight = masked_weight
        self.contrastive_weight = contrastive_weight
        
        self.masked_loss = MaskedModelingLoss()
        self.contrastive_loss = ContrastiveLoss(temperature)
    
    def forward(self,
                reconstructed: torch.Tensor,
                original: torch.Tensor,
                mask: torch.Tensor,
                z_i: torch.Tensor,
                z_j: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate hybrid pretraining loss.
        
        Returns:
            Total loss and individual components
        """
        # Masked modeling loss
        masked_l = self.masked_loss(reconstructed, original, mask)
        
        # Contrastive loss
        contrastive_l = self.contrastive_loss(z_i, z_j)
        
        # Combined loss
        total_loss = (
            self.masked_weight * masked_l +
            self.contrastive_weight * contrastive_l
        )
        
        losses = {
            'total': total_loss.item(),
            'masked': masked_l.item(),
            'contrastive': contrastive_l.item()
        }
        
        return total_loss, losses


if __name__ == "__main__":
    print("🧪 Testing Loss Functions...")
    
    # Test multi-task loss
    batch_size = 16
    
    predictions = {
        'buy_prob': torch.rand(batch_size),
        'sell_prob': torch.rand(batch_size),
        'direction_logits': torch.randn(batch_size, 2),
        'regime_logits': torch.randn(batch_size, 6)
    }
    
    targets = {
        'buy': torch.randint(0, 2, (batch_size,)),
        'sell': torch.randint(0, 2, (batch_size,)),
        'direction': torch.randint(0, 2, (batch_size,)),
        'regime': torch.randint(0, 6, (batch_size,))
    }
    
    loss_fn = MultiTaskLoss()
    total_loss, losses = loss_fn(predictions, targets)
    
    print(f"✅ Multi-task loss: {total_loss.item():.4f}")
    print(f"   Individual losses: {losses}")
    
    # Test contrastive loss
    z_i = torch.randn(batch_size, 128)
    z_j = torch.randn(batch_size, 128)
    
    contrastive_fn = ContrastiveLoss()
    contrastive_loss = contrastive_fn(z_i, z_j)
    
    print(f"\n✅ Contrastive loss: {contrastive_loss.item():.4f}")
    
    print(f"\n✅ All loss functions working correctly!")
