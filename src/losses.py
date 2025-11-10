"""
Loss Functions for Natron Transformer
- Multi-task weighted loss (supervised)
- Masked reconstruction loss (pretraining)
- Contrastive loss (pretraining)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class MultiTaskLoss(nn.Module):
    """
    Multi-task weighted loss for supervised training
    
    Tasks:
    - Buy: Binary cross-entropy
    - Sell: Binary cross-entropy
    - Direction: Cross-entropy (2 classes)
    - Regime: Cross-entropy (6 classes)
    """
    
    def __init__(
        self,
        buy_weight: float = 1.0,
        sell_weight: float = 1.0,
        direction_weight: float = 0.8,
        regime_weight: float = 0.6,
        class_weights: Dict = None
    ):
        super().__init__()
        
        self.buy_weight = buy_weight
        self.sell_weight = sell_weight
        self.direction_weight = direction_weight
        self.regime_weight = regime_weight
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        
        # Direction and regime with optional class weights
        direction_weights = None
        regime_weights = None
        
        if class_weights:
            if 'direction' in class_weights:
                direction_weights = torch.tensor(class_weights['direction'])
            if 'regime' in class_weights:
                regime_weights = torch.tensor(class_weights['regime'])
        
        self.direction_loss = nn.CrossEntropyLoss(weight=direction_weights)
        self.regime_loss = nn.CrossEntropyLoss(weight=regime_weights)
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Dict with 'buy_logits', 'sell_logits', 'direction_logits', 'regime_logits'
            targets: Dict with 'buy', 'sell', 'direction', 'regime'
            
        Returns:
            Dict with 'total_loss' and individual losses
        """
        # Buy loss
        buy_loss = self.bce_loss(
            predictions['buy_logits'].squeeze(-1),
            targets['buy']
        )
        
        # Sell loss
        sell_loss = self.bce_loss(
            predictions['sell_logits'].squeeze(-1),
            targets['sell']
        )
        
        # Direction loss
        direction_loss = self.direction_loss(
            predictions['direction_logits'],
            targets['direction']
        )
        
        # Regime loss
        regime_loss = self.regime_loss(
            predictions['regime_logits'],
            targets['regime']
        )
        
        # Weighted total loss
        total_loss = (
            self.buy_weight * buy_loss +
            self.sell_weight * sell_loss +
            self.direction_weight * direction_loss +
            self.regime_weight * regime_loss
        )
        
        return {
            'total_loss': total_loss,
            'buy_loss': buy_loss,
            'sell_loss': sell_loss,
            'direction_loss': direction_loss,
            'regime_loss': regime_loss
        }


class PretrainingLoss(nn.Module):
    """
    Combined loss for pretraining
    - Reconstruction loss (MSE)
    - Contrastive loss (InfoNCE)
    """
    
    def __init__(
        self,
        reconstruction_weight: float = 0.6,
        contrastive_weight: float = 0.4,
        temperature: float = 0.07
    ):
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        original: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute pretraining loss
        
        Args:
            predictions: Dict with 'reconstructed' and 'contrastive_emb'
            original: Original input (batch, seq, features)
            mask: Mask for reconstruction loss (batch, seq)
            
        Returns:
            Dict with losses
        """
        # Reconstruction loss (only on masked tokens if mask provided)
        if mask is not None:
            # Expand mask to match features dimension
            mask_expanded = mask.unsqueeze(-1).expand_as(original)
            
            masked_pred = predictions['reconstructed'][mask_expanded]
            masked_orig = original[mask_expanded]
            
            if len(masked_pred) > 0:
                reconstruction_loss = F.mse_loss(masked_pred, masked_orig)
            else:
                reconstruction_loss = torch.tensor(0.0, device=original.device)
        else:
            reconstruction_loss = self.mse_loss(
                predictions['reconstructed'],
                original
            )
        
        # Contrastive loss (InfoNCE)
        contrastive_loss = self._contrastive_loss(
            predictions['contrastive_emb']
        )
        
        # Total loss
        total_loss = (
            self.reconstruction_weight * reconstruction_loss +
            self.contrastive_weight * contrastive_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'contrastive_loss': contrastive_loss
        }
    
    def _contrastive_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute InfoNCE contrastive loss
        
        Args:
            embeddings: (batch_size, embedding_dim) - normalized embeddings
            
        Returns:
            Scalar loss
        """
        batch_size = embeddings.shape[0]
        
        if batch_size < 2:
            return torch.tensor(0.0, device=embeddings.device)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings, embeddings.T) / self.temperature
        
        # Create positive pairs mask (diagonal)
        # In a more sophisticated version, we could create augmented views
        # For now, we use a simplified version where we contrast each sample with all others
        
        # Mask out diagonal (self-similarity)
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        
        # Labels: for each sample, positive is itself (but we've masked it)
        # Simplified: maximize separation between different samples
        # Use softmax to create pseudo-labels
        
        # Alternative: SimCLR-style with augmentations would be better
        # For now, use a simplified contrastive objective
        
        # Compute log-softmax
        log_prob = F.log_softmax(similarity_matrix, dim=1)
        
        # Take mean of negative log-likelihood
        # This encourages diversity in embeddings
        loss = -log_prob.mean()
        
        return loss


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Useful when buy/sell signals are rare
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (batch_size,)
            targets: Binary labels (batch_size,)
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction='none'
        )
        
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        loss = alpha_weight * focal_weight * bce_loss
        
        return loss.mean()


class MultiTaskFocalLoss(nn.Module):
    """Multi-task loss with Focal Loss for buy/sell signals"""
    
    def __init__(
        self,
        buy_weight: float = 1.0,
        sell_weight: float = 1.0,
        direction_weight: float = 0.8,
        regime_weight: float = 0.6,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        
        self.buy_weight = buy_weight
        self.sell_weight = sell_weight
        self.direction_weight = direction_weight
        self.regime_weight = regime_weight
        
        self.buy_loss_fn = FocalLoss(focal_alpha, focal_gamma)
        self.sell_loss_fn = FocalLoss(focal_alpha, focal_gamma)
        self.direction_loss = nn.CrossEntropyLoss()
        self.regime_loss = nn.CrossEntropyLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss with Focal Loss"""
        
        buy_loss = self.buy_loss_fn(
            predictions['buy_logits'].squeeze(-1),
            targets['buy']
        )
        
        sell_loss = self.sell_loss_fn(
            predictions['sell_logits'].squeeze(-1),
            targets['sell']
        )
        
        direction_loss = self.direction_loss(
            predictions['direction_logits'],
            targets['direction']
        )
        
        regime_loss = self.regime_loss(
            predictions['regime_logits'],
            targets['regime']
        )
        
        total_loss = (
            self.buy_weight * buy_loss +
            self.sell_weight * sell_loss +
            self.direction_weight * direction_loss +
            self.regime_weight * regime_loss
        )
        
        return {
            'total_loss': total_loss,
            'buy_loss': buy_loss,
            'sell_loss': sell_loss,
            'direction_loss': direction_loss,
            'regime_loss': regime_loss
        }


def compute_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute accuracy metrics for all tasks
    
    Args:
        predictions: Model predictions
        targets: Ground truth labels
        
    Returns:
        Dict with accuracy metrics
    """
    metrics = {}
    
    # Buy accuracy (threshold at 0.5)
    buy_pred = (torch.sigmoid(predictions['buy_logits']) > 0.5).float().squeeze()
    metrics['buy_acc'] = (buy_pred == targets['buy']).float().mean().item()
    
    # Sell accuracy
    sell_pred = (torch.sigmoid(predictions['sell_logits']) > 0.5).float().squeeze()
    metrics['sell_acc'] = (sell_pred == targets['sell']).float().mean().item()
    
    # Direction accuracy
    direction_pred = predictions['direction_logits'].argmax(dim=-1)
    metrics['direction_acc'] = (direction_pred == targets['direction']).float().mean().item()
    
    # Regime accuracy
    regime_pred = predictions['regime_logits'].argmax(dim=-1)
    metrics['regime_acc'] = (regime_pred == targets['regime']).float().mean().item()
    
    # Average accuracy
    metrics['avg_acc'] = (
        metrics['buy_acc'] + metrics['sell_acc'] +
        metrics['direction_acc'] + metrics['regime_acc']
    ) / 4
    
    return metrics


if __name__ == "__main__":
    # Test losses
    print("Testing Loss Functions...")
    
    batch_size = 16
    seq_len = 96
    num_features = 100
    
    # Test multi-task loss
    print("\n1. Multi-Task Loss")
    loss_fn = MultiTaskLoss(
        buy_weight=1.0,
        sell_weight=1.0,
        direction_weight=0.8,
        regime_weight=0.6
    )
    
    predictions = {
        'buy_logits': torch.randn(batch_size, 1),
        'sell_logits': torch.randn(batch_size, 1),
        'direction_logits': torch.randn(batch_size, 2),
        'regime_logits': torch.randn(batch_size, 6)
    }
    
    targets = {
        'buy': torch.randint(0, 2, (batch_size,)).float(),
        'sell': torch.randint(0, 2, (batch_size,)).float(),
        'direction': torch.randint(0, 2, (batch_size,)),
        'regime': torch.randint(0, 6, (batch_size,))
    }
    
    losses = loss_fn(predictions, targets)
    print(f"Total Loss: {losses['total_loss'].item():.4f}")
    for key, value in losses.items():
        if key != 'total_loss':
            print(f"  {key}: {value.item():.4f}")
    
    # Test metrics
    metrics = compute_metrics(predictions, targets)
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Test pretraining loss
    print("\n2. Pretraining Loss")
    pretrain_loss_fn = PretrainingLoss(
        reconstruction_weight=0.6,
        contrastive_weight=0.4
    )
    
    original = torch.randn(batch_size, seq_len, num_features)
    pretrain_predictions = {
        'reconstructed': torch.randn(batch_size, seq_len, num_features),
        'contrastive_emb': F.normalize(torch.randn(batch_size, 128), dim=-1)
    }
    
    pretrain_losses = pretrain_loss_fn(pretrain_predictions, original)
    print(f"Total Loss: {pretrain_losses['total_loss'].item():.4f}")
    for key, value in pretrain_losses.items():
        if key != 'total_loss':
            print(f"  {key}: {value.item():.4f}")
