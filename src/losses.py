"""
Natron Transformer - Loss Functions
Multi-task losses for supervised training and pretraining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning
    Combines buy, sell, direction, and regime losses
    """
    
    def __init__(
        self,
        loss_weights: Dict[str, float],
        direction_weights: Optional[torch.Tensor] = None,
        regime_weights: Optional[torch.Tensor] = None
    ):
        """
        Args:
            loss_weights: Weights for each task loss
            direction_weights: Class weights for direction (2 classes)
            regime_weights: Class weights for regime (6 classes)
        """
        super().__init__()
        
        self.loss_weights = loss_weights
        
        # Binary cross-entropy for buy/sell
        self.bce_loss = nn.BCELoss()
        
        # Cross-entropy for direction and regime
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
            predictions: Dict with 'buy', 'sell', 'direction', 'regime'
            targets: Dict with 'buy', 'sell', 'direction', 'regime'
            
        Returns:
            Dict with total loss and individual losses
        """
        # Buy loss
        buy_loss = self.bce_loss(predictions['buy'], targets['buy'])
        
        # Sell loss
        sell_loss = self.bce_loss(predictions['sell'], targets['sell'])
        
        # Direction loss
        direction_loss = self.direction_loss(predictions['direction'], targets['direction'])
        
        # Regime loss
        regime_loss = self.regime_loss(predictions['regime'], targets['regime'])
        
        # Combined loss
        total_loss = (
            self.loss_weights['buy'] * buy_loss +
            self.loss_weights['sell'] * sell_loss +
            self.loss_weights['direction'] * direction_loss +
            self.loss_weights['regime'] * regime_loss
        )
        
        return {
            'total': total_loss,
            'buy': buy_loss,
            'sell': sell_loss,
            'direction': direction_loss,
            'regime': regime_loss
        }


class MaskedReconstructionLoss(nn.Module):
    """
    Loss for masked token reconstruction (pretraining)
    """
    
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
    
    def forward(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute reconstruction loss only on masked positions
        
        Args:
            reconstructed: Reconstructed sequence (batch, seq_len, input_dim)
            original: Original sequence (batch, seq_len, input_dim)
            mask: Boolean mask (batch, seq_len, 1) - True for masked positions
            
        Returns:
            Scalar loss
        """
        # Only compute loss on masked positions
        masked_reconstructed = reconstructed * mask
        masked_original = original * mask
        
        loss = self.mse_loss(masked_reconstructed, masked_original)
        return loss


class InfoNCELoss(nn.Module):
    """
    InfoNCE Contrastive Loss for pretraining
    Used in self-supervised learning
    """
    
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss
        
        Args:
            anchor: Anchor embeddings (batch, dim)
            positive: Positive embeddings (batch, dim)
            negatives: Negative embeddings (batch * (N-1), dim) or None
            
        Returns:
            Scalar loss
        """
        batch_size = anchor.size(0)
        
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        
        # Positive similarity
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature  # (batch,)
        
        # Negative similarity (use all other samples in batch)
        if negatives is None:
            # Use in-batch negatives
            all_embeddings = torch.cat([anchor, positive], dim=0)  # (2*batch, dim)
            
            # Compute all pairwise similarities
            sim_matrix = torch.matmul(anchor, all_embeddings.T) / self.temperature  # (batch, 2*batch)
            
            # Create mask to exclude self-similarity
            mask = torch.eye(batch_size, dtype=torch.bool, device=anchor.device)
            mask = torch.cat([mask, torch.zeros_like(mask)], dim=1)
            
            # Mask out self-similarity
            sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
            
            # LogSumExp over all samples
            log_sum_exp_all = torch.logsumexp(sim_matrix, dim=1)
            
            # Loss
            loss = -pos_sim + log_sum_exp_all
        else:
            # Use provided negatives
            negatives = F.normalize(negatives, dim=-1)
            neg_sim = torch.matmul(anchor, negatives.T) / self.temperature  # (batch, num_negatives)
            
            # Combine positive and negative similarities
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # (batch, 1 + num_negatives)
            
            # Loss (cross-entropy with positive at index 0)
            labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
            loss = F.cross_entropy(logits, labels)
            return loss
        
        return loss.mean()


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance
    Can be used as alternative to CrossEntropyLoss
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits (batch, num_classes)
            targets: Target labels (batch,)
            
        Returns:
            Scalar loss
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def compute_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute classification accuracy
    
    Args:
        predictions: Logits or probabilities
        targets: Ground truth labels
        
    Returns:
        Accuracy as float
    """
    if predictions.dim() > 1 and predictions.size(1) > 1:
        # Multi-class: take argmax
        pred_labels = predictions.argmax(dim=1)
    else:
        # Binary: threshold at 0.5
        pred_labels = (predictions > 0.5).long().squeeze()
    
    correct = (pred_labels == targets).sum().item()
    total = targets.size(0)
    
    return correct / total


def compute_f1_score(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Compute F1 score for binary classification
    
    Args:
        predictions: Probabilities (batch,)
        targets: Binary labels (batch,)
        
    Returns:
        F1 score
    """
    pred_labels = (predictions > 0.5).long()
    
    tp = ((pred_labels == 1) & (targets == 1)).sum().item()
    fp = ((pred_labels == 1) & (targets == 0)).sum().item()
    fn = ((pred_labels == 0) & (targets == 1)).sum().item()
    
    if tp + fp == 0:
        precision = 0.0
    else:
        precision = tp / (tp + fp)
    
    if tp + fn == 0:
        recall = 0.0
    else:
        recall = tp / (tp + fn)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def create_data_augmentation(x: torch.Tensor, augmentation_type: str = 'noise') -> torch.Tensor:
    """
    Data augmentation for time series
    
    Args:
        x: Input sequence (batch, seq_len, dim)
        augmentation_type: Type of augmentation
        
    Returns:
        Augmented sequence
    """
    if augmentation_type == 'noise':
        # Add Gaussian noise
        noise = torch.randn_like(x) * 0.01
        return x + noise
    
    elif augmentation_type == 'scale':
        # Random scaling
        scale = torch.rand(x.size(0), 1, 1, device=x.device) * 0.2 + 0.9  # [0.9, 1.1]
        return x * scale
    
    elif augmentation_type == 'mask':
        # Random masking (for contrastive learning)
        mask = torch.rand_like(x) > 0.1  # Keep 90%
        return x * mask
    
    elif augmentation_type == 'cutout':
        # Temporal cutout
        batch_size, seq_len, dim = x.shape
        cutout_len = int(seq_len * 0.1)
        
        augmented = x.clone()
        for i in range(batch_size):
            start = torch.randint(0, seq_len - cutout_len, (1,)).item()
            augmented[i, start:start+cutout_len, :] = 0
        
        return augmented
    
    else:
        return x


if __name__ == "__main__":
    # Test losses
    print("🧪 Testing Loss Functions...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32
    
    # Test multi-task loss
    print("\n1️⃣ Testing MultiTaskLoss...")
    loss_weights = {'buy': 1.0, 'sell': 1.0, 'direction': 1.5, 'regime': 2.0}
    mt_loss = MultiTaskLoss(loss_weights)
    
    predictions = {
        'buy': torch.rand(batch_size).to(device),
        'sell': torch.rand(batch_size).to(device),
        'direction': torch.randn(batch_size, 2).to(device),
        'regime': torch.randn(batch_size, 6).to(device)
    }
    
    targets = {
        'buy': torch.randint(0, 2, (batch_size,)).float().to(device),
        'sell': torch.randint(0, 2, (batch_size,)).float().to(device),
        'direction': torch.randint(0, 2, (batch_size,)).to(device),
        'regime': torch.randint(0, 6, (batch_size,)).to(device)
    }
    
    losses = mt_loss(predictions, targets)
    print(f"  Total loss: {losses['total'].item():.4f}")
    print(f"  Buy loss: {losses['buy'].item():.4f}")
    print(f"  Sell loss: {losses['sell'].item():.4f}")
    print(f"  Direction loss: {losses['direction'].item():.4f}")
    print(f"  Regime loss: {losses['regime'].item():.4f}")
    
    # Test InfoNCE loss
    print("\n2️⃣ Testing InfoNCELoss...")
    infonce_loss = InfoNCELoss(temperature=0.07)
    
    anchor = torch.randn(batch_size, 128).to(device)
    positive = torch.randn(batch_size, 128).to(device)
    
    loss = infonce_loss(anchor, positive)
    print(f"  InfoNCE loss: {loss.item():.4f}")
    
    # Test masked reconstruction loss
    print("\n3️⃣ Testing MaskedReconstructionLoss...")
    recon_loss = MaskedReconstructionLoss()
    
    original = torch.randn(batch_size, 96, 100).to(device)
    reconstructed = torch.randn(batch_size, 96, 100).to(device)
    mask = (torch.rand(batch_size, 96, 1) < 0.15).to(device)
    
    loss = recon_loss(reconstructed, original, mask)
    print(f"  Reconstruction loss: {loss.item():.4f}")
    
    # Test accuracy
    print("\n4️⃣ Testing Accuracy...")
    preds = torch.randn(batch_size, 2).to(device)
    targets = torch.randint(0, 2, (batch_size,)).to(device)
    
    acc = compute_accuracy(preds, targets)
    print(f"  Accuracy: {acc:.4f}")
    
    # Test F1 score
    print("\n5️⃣ Testing F1 Score...")
    preds = torch.rand(batch_size).to(device)
    targets = torch.randint(0, 2, (batch_size,)).float().to(device)
    
    f1 = compute_f1_score(preds, targets)
    print(f"  F1 Score: {f1:.4f}")
    
    print("\n✅ Loss tests successful!")
