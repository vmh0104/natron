"""
Natron Loss Functions - Multi-task, Contrastive, and RL Losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MultiTaskLoss(nn.Module):
    """Multi-task loss for supervised fine-tuning."""
    
    def __init__(
        self,
        buy_weight: float = 1.0,
        sell_weight: float = 1.0,
        direction_weight: float = 1.0,
        regime_weight: float = 1.0
    ):
        super().__init__()
        self.buy_weight = buy_weight
        self.sell_weight = sell_weight
        self.direction_weight = direction_weight
        self.regime_weight = regime_weight
        
        # Binary cross-entropy for buy/sell
        self.bce_loss = nn.BCELoss()
        
        # Cross-entropy for direction and regime
        self.ce_loss = nn.NLLLoss()
    
    def forward(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate multi-task loss.
        
        Args:
            predictions: Dict with 'buy', 'sell', 'direction', 'regime'
            targets: Dict with ground truth labels
        
        Returns:
            Dict with individual losses and total loss
        """
        losses = {}
        
        # Buy loss
        if 'buy' in predictions and 'buy' in targets:
            buy_pred = predictions['buy'].squeeze()
            buy_target = targets['buy'].squeeze()
            losses['buy'] = self.bce_loss(buy_pred, buy_target) * self.buy_weight
        
        # Sell loss
        if 'sell' in predictions and 'sell' in targets:
            sell_pred = predictions['sell'].squeeze()
            sell_target = targets['sell'].squeeze()
            losses['sell'] = self.bce_loss(sell_pred, sell_target) * self.sell_weight
        
        # Direction loss
        if 'direction' in predictions and 'direction' in targets:
            direction_pred = predictions['direction']
            direction_target = targets['direction'].squeeze()
            losses['direction'] = self.ce_loss(direction_pred, direction_target) * self.direction_weight
        
        # Regime loss
        if 'regime' in predictions and 'regime' in targets:
            regime_pred = predictions['regime']
            regime_target = targets['regime'].squeeze()
            losses['regime'] = self.ce_loss(regime_pred, regime_target) * self.regime_weight
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses


class MaskedModelingLoss(nn.Module):
    """Loss for masked token reconstruction (pretraining)."""
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate reconstruction loss only for masked tokens.
        
        Args:
            reconstructed: Reconstructed features (batch_size, seq_len, feature_dim)
            original: Original features (batch_size, seq_len, feature_dim)
            mask: Boolean mask indicating masked tokens (batch_size, seq_len)
        
        Returns:
            Scalar loss
        """
        # Expand mask to match feature dimensions
        mask_expanded = mask.unsqueeze(-1).expand_as(reconstructed)
        
        # Calculate MSE only for masked positions
        loss = self.mse_loss(reconstructed, original)
        masked_loss = loss * mask_expanded.float()
        
        if self.reduction == 'mean':
            # Average over masked tokens only
            n_masked = mask_expanded.sum().float()
            return masked_loss.sum() / (n_masked + 1e-8)
        else:
            return masked_loss.sum()


class ContrastiveLoss(nn.Module):
    """InfoNCE / SimCLR style contrastive loss."""
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Calculate contrastive loss between two augmented views.
        
        Args:
            z1: Projections from first augmentation (batch_size, projection_dim)
            z2: Projections from second augmentation (batch_size, projection_dim)
        
        Returns:
            Contrastive loss
        """
        batch_size = z1.size(0)
        
        # Normalize
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        # Concatenate all projections
        all_projections = torch.cat([z1, z2], dim=0)  # (2*batch_size, projection_dim)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(all_projections, all_projections.t()) / self.temperature
        
        # Create labels: positive pairs are (i, i+batch_size) and (i+batch_size, i)
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, device=z1.device, dtype=torch.bool)
        similarity_matrix.masked_fill_(mask, float('-inf'))
        
        # Cross-entropy loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class PPOActorCriticLoss(nn.Module):
    """PPO loss for reinforcement learning."""
    
    def __init__(
        self,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01
    ):
        super().__init__()
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
    
    def forward(
        self,
        old_log_probs: torch.Tensor,
        new_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        old_values: torch.Tensor,
        new_values: torch.Tensor,
        returns: torch.Tensor,
        action_probs: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate PPO loss.
        
        Args:
            old_log_probs: Log probabilities from old policy
            new_log_probs: Log probabilities from new policy
            advantages: Advantage estimates
            old_values: Value estimates from old policy
            new_values: Value estimates from new policy
            returns: Actual returns
            action_probs: Action probabilities for entropy calculation
        
        Returns:
            Dict with policy loss, value loss, entropy, and total loss
        """
        # Policy loss (clipped)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = F.mse_loss(new_values, returns)
        
        # Entropy bonus
        entropy = -(action_probs * torch.log(action_probs + 1e-8)).sum(dim=-1).mean()
        
        # Total loss
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        
        return {
            'policy_loss': policy_loss,
            'value_loss': value_loss,
            'entropy': entropy,
            'total_loss': total_loss
        }


class TradingReward:
    """Reward function for RL trading."""
    
    def __init__(
        self,
        profit_coef: float = 1.0,
        turnover_penalty: float = 0.01,
        drawdown_penalty: float = 0.1
    ):
        self.profit_coef = profit_coef
        self.turnover_penalty = turnover_penalty
        self.drawdown_penalty = drawdown_penalty
    
    def calculate_reward(
        self,
        action: int,  # 0: hold, 1: buy, 2: sell
        price_change: float,
        position: int,  # -1: short, 0: flat, 1: long
        trades_count: int,
        max_drawdown: float
    ) -> float:
        """
        Calculate trading reward.
        
        Args:
            action: Action taken
            price_change: Price change (as fraction)
            position: Current position
            trades_count: Number of trades in episode
            max_drawdown: Maximum drawdown
        
        Returns:
            Reward value
        """
        # Profit component
        if position == 1:  # Long
            profit = price_change
        elif position == -1:  # Short
            profit = -price_change
        else:
            profit = 0
        
        profit_reward = profit * self.profit_coef
        
        # Turnover penalty
        turnover_penalty = trades_count * self.turnover_penalty
        
        # Drawdown penalty
        drawdown_penalty = max_drawdown * self.drawdown_penalty
        
        # Total reward
        reward = profit_reward - turnover_penalty - drawdown_penalty
        
        return reward
