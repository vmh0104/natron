"""
Loss Functions for Natron Transformer Multi-Task Training
Includes: Multi-task loss, Contrastive loss, Masked modeling loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss combining:
    - Buy/Sell binary cross-entropy
    - Direction cross-entropy
    - Regime cross-entropy
    """
    
    def __init__(self, loss_weights: dict):
        super().__init__()
        self.loss_weights = loss_weights
        
        # Binary cross-entropy for buy/sell
        self.bce_loss = nn.BCELoss()
        
        # Cross-entropy for direction and regime
        self.ce_loss = nn.NLLLoss()
        
    def forward(self, predictions: dict, targets: dict) -> dict:
        """
        Args:
            predictions: Dict with 'buy', 'sell', 'direction', 'regime'
            targets: Dict with 'buy', 'sell', 'direction', 'regime'
            
        Returns:
            Dict with individual losses and total loss
        """
        losses = {}
        
        # Buy loss
        buy_loss = self.bce_loss(
            predictions['buy'].squeeze(),
            targets['buy'].squeeze()
        )
        losses['buy'] = buy_loss * self.loss_weights.get('buy', 1.0)
        
        # Sell loss
        sell_loss = self.bce_loss(
            predictions['sell'].squeeze(),
            targets['sell'].squeeze()
        )
        losses['sell'] = sell_loss * self.loss_weights.get('sell', 1.0)
        
        # Direction loss
        direction_loss = self.ce_loss(
            predictions['direction'],
            targets['direction'].squeeze()
        )
        losses['direction'] = direction_loss * self.loss_weights.get('direction', 1.0)
        
        # Regime loss
        regime_loss = self.ce_loss(
            predictions['regime'],
            targets['regime'].squeeze()
        )
        losses['regime'] = regime_loss * self.loss_weights.get('regime', 1.5)
        
        # Total loss
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return losses


class MaskedModelingLoss(nn.Module):
    """
    Loss for masked feature reconstruction (pretraining Phase 1).
    Uses MSE loss for continuous features.
    """
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def forward(self, 
                reconstructed: torch.Tensor,
                original: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            reconstructed: (batch_size, seq_len, feature_dim)
            original: (batch_size, seq_len, feature_dim)
            mask: (batch_size, seq_len) - True for masked positions
            
        Returns:
            loss: Scalar loss value
        """
        # Compute MSE only for masked positions
        mse = self.mse_loss(reconstructed, original)  # (batch_size, seq_len, feature_dim)
        mse = mse.mean(dim=-1)  # Average over features: (batch_size, seq_len)
        
        # Mask: only compute loss for masked positions
        mask_expanded = mask.unsqueeze(-1).expand_as(mse)  # (batch_size, seq_len)
        masked_loss = mse * mask_expanded.float()
        
        # Average over masked positions
        loss = masked_loss.sum() / (mask.sum().float() + 1e-8)
        
        return loss


class ContrastiveLoss(nn.Module):
    """
    InfoNCE / SimCLR style contrastive loss for pretraining.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, 
                anchor: torch.Tensor,
                positive: torch.Tensor,
                negatives: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            anchor: (batch_size, projection_dim) - anchor embeddings
            positive: (batch_size, projection_dim) - positive (augmented) embeddings
            negatives: Optional (batch_size, n_negatives, projection_dim)
            
        Returns:
            loss: Contrastive loss
        """
        batch_size = anchor.size(0)
        
        # Normalize embeddings (should already be normalized, but ensure)
        anchor = F.normalize(anchor, p=2, dim=-1)
        positive = F.normalize(positive, p=2, dim=-1)
        
        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature  # (batch_size,)
        
        if negatives is not None:
            # Use provided negatives
            negatives = F.normalize(negatives, p=2, dim=-1)
            # Compute similarity with negatives
            neg_sim = torch.bmm(
                anchor.unsqueeze(1),  # (batch_size, 1, projection_dim)
                negatives.transpose(1, 2)  # (batch_size, projection_dim, n_negatives)
            ).squeeze(1) / self.temperature  # (batch_size, n_negatives)
            
            # Concatenate positive and negatives
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch_size, 1 + n_negatives)
        else:
            # Use other samples in batch as negatives
            # Compute all pairwise similarities
            all_embeddings = torch.cat([positive.unsqueeze(1), anchor.unsqueeze(1)], dim=1)  # (batch_size, 2, projection_dim)
            all_embeddings = all_embeddings.view(batch_size * 2, -1)  # (batch_size * 2, projection_dim)
            
            # Similarity matrix
            sim_matrix = torch.mm(all_embeddings, all_embeddings.t()) / self.temperature  # (batch_size * 2, batch_size * 2)
            
            # Mask out self-similarity
            mask = torch.eye(batch_size * 2, device=anchor.device).bool()
            sim_matrix.masked_fill_(mask, float('-inf'))
            
            # For each anchor, positive is at index batch_size + i
            labels = torch.arange(batch_size, device=anchor.device) + batch_size
            
            # Use cross-entropy loss
            loss = F.cross_entropy(sim_matrix[:batch_size], labels)
            return loss
        
        # Labels: 0 is the positive
        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class AugmentedContrastiveLoss(nn.Module):
    """
    Contrastive loss with data augmentation.
    Creates positive pairs through augmentation.
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.contrastive_loss = ContrastiveLoss(temperature)
        
    def augment_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentation to sequence.
        Augmentations:
        - Random noise
        - Time masking
        - Feature masking
        """
        augmented = x.clone()
        batch_size, seq_len, feature_dim = x.shape
        
        # Random noise (small)
        noise = torch.randn_like(x) * 0.01
        augmented = augmented + noise
        
        # Random time masking (mask 10% of timesteps)
        mask_prob = 0.1
        time_mask = torch.rand(batch_size, seq_len, device=x.device) > mask_prob
        time_mask = time_mask.unsqueeze(-1).expand_as(augmented)
        augmented = augmented * time_mask.float()
        
        return augmented
    
    def forward(self, 
                encoder: nn.Module,
                contrastive_head: nn.Module,
                x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder: NatronTransformerEncoder
            contrastive_head: ContrastiveHead
            x: (batch_size, seq_len, feature_dim)
            
        Returns:
            loss: Contrastive loss
        """
        # Original encoding
        encoded_orig = encoder(x)
        projected_orig = contrastive_head(encoded_orig)
        
        # Augmented encoding
        x_aug = self.augment_sequence(x)
        encoded_aug = encoder(x_aug)
        projected_aug = contrastive_head(encoded_aug)
        
        # Contrastive loss
        loss = self.contrastive_loss(projected_orig, projected_aug)
        
        return loss
