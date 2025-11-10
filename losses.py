"""
Loss Functions for Natron Transformer Multi-Task Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss combining buy, sell, direction, and regime predictions.
    """
    
    def __init__(
        self,
        buy_weight: float = 1.0,
        sell_weight: float = 1.0,
        direction_weight: float = 1.0,
        regime_weight: float = 1.0,
        class_weights: Optional[Dict[str, torch.Tensor]] = None
    ):
        """
        Args:
            buy_weight: Weight for buy signal loss
            sell_weight: Weight for sell signal loss
            direction_weight: Weight for direction prediction loss
            regime_weight: Weight for regime classification loss
            class_weights: Optional class weights for imbalanced datasets
        """
        super().__init__()
        
        self.buy_weight = buy_weight
        self.sell_weight = sell_weight
        self.direction_weight = direction_weight
        self.regime_weight = regime_weight
        
        # Loss functions
        self.buy_loss_fn = nn.BCELoss()
        self.sell_loss_fn = nn.BCELoss()
        self.direction_loss_fn = nn.CrossEntropyLoss(
            weight=class_weights.get('direction') if class_weights else None
        )
        self.regime_loss_fn = nn.CrossEntropyLoss(
            weight=class_weights.get('regime') if class_weights else None
        )
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Dict with 'buy', 'sell', 'direction', 'regime'
            targets: Dict with 'buy', 'sell', 'direction', 'regime'
            
        Returns:
            Dictionary with individual losses and total loss
        """
        # Buy loss
        buy_loss = self.buy_loss_fn(
            predictions['buy'].squeeze(),
            targets['buy'].squeeze()
        )
        
        # Sell loss
        sell_loss = self.sell_loss_fn(
            predictions['sell'].squeeze(),
            targets['sell'].squeeze()
        )
        
        # Direction loss
        direction_loss = self.direction_loss_fn(
            predictions['direction'],
            targets['direction'].squeeze()
        )
        
        # Regime loss
        regime_loss = self.regime_loss_fn(
            predictions['regime'],
            targets['regime'].squeeze()
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


class PretrainLoss(nn.Module):
    """
    Loss for Phase 1 Pretraining: Masked Modeling + Contrastive Learning
    """
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        temperature: float = 0.07
    ):
        """
        Args:
            reconstruction_weight: Weight for masked reconstruction loss
            contrastive_weight: Weight for contrastive loss
            temperature: Temperature for contrastive loss
        """
        super().__init__()
        
        self.reconstruction_weight = reconstruction_weight
        self.contrastive_weight = contrastive_weight
        self.temperature = temperature
        
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute pretraining loss
        
        Args:
            predictions: Dict with 'reconstruction' and 'contrastive_embedding'
            targets: Original input sequence (batch, seq_len, features)
            mask: Boolean mask indicating masked positions
            
        Returns:
            Dictionary with losses
        """
        reconstruction = predictions['reconstruction']
        contrastive_embedding = predictions['contrastive_embedding']
        
        # Reconstruction loss (only on masked positions)
        recon_loss = self.mse_loss(reconstruction, targets)
        recon_loss = (recon_loss * mask.unsqueeze(-1)).sum() / (mask.sum() + 1e-8)
        
        # Contrastive loss (InfoNCE)
        batch_size = contrastive_embedding.size(0)
        
        # Create positive pairs (augmented versions of same sample)
        # For simplicity, use different timesteps as positives
        # In practice, you'd use data augmentation
        
        # SimCLR-style contrastive loss
        # Normalize embeddings
        embeddings_norm = F.normalize(contrastive_embedding, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.T) / self.temperature
        
        # Create labels (diagonal = positive pairs)
        labels = torch.arange(batch_size, device=contrastive_embedding.device)
        
        # Cross-entropy loss
        contrastive_loss = F.cross_entropy(similarity_matrix, labels)
        
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.contrastive_weight * contrastive_loss
        )
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': recon_loss,
            'contrastive_loss': contrastive_loss
        }


class RLLoss(nn.Module):
    """
    Reinforcement Learning Loss (for Phase 3: PPO/SAC)
    """
    
    def __init__(self, clip_epsilon: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01):
        """
        Args:
            clip_epsilon: PPO clip parameter
            value_coef: Value function loss coefficient
            entropy_coef: Entropy bonus coefficient
        """
        super().__init__()
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
    
    def compute_ppo_loss(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        values: torch.Tensor,
        old_values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Compute PPO loss
        
        Args:
            old_log_probs: Log probabilities from old policy
            new_log_probs: Log probabilities from new policy
            advantages: Advantage estimates
            returns: Discounted returns
            values: Current value estimates
            old_values: Old value estimates
            
        Returns:
            Dictionary with PPO losses
        """
        # Policy ratio
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # Clipped surrogate objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss (clipped)
        value_clipped = old_values + torch.clamp(
            values - old_values,
            -self.clip_epsilon,
            self.clip_epsilon
        )
        value_loss1 = (values - returns).pow(2)
        value_loss2 = (value_clipped - returns).pow(2)
        value_loss = 0.5 * torch.max(value_loss1, value_loss2).mean()
        
        # Entropy bonus
        entropy = -(torch.exp(new_log_probs) * new_log_probs).sum(dim=-1).mean()
        
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        return {
            'total_loss': total_loss,
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy
        }
