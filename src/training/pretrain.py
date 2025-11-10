"""
Natron V2 Phase 1: Pretraining Module
Unsupervised pretraining with masked modeling and contrastive learning
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from typing import Dict

from src.models.natron_transformer import PretrainTransformer
from src.models.losses import PretrainLoss


class PretrainManager:
    """
    Manages Phase 1 pretraining: masked modeling + contrastive learning
    """
    
    def __init__(self, config: dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        
        # Model
        self.model = PretrainTransformer(config).to(device)
        
        # Loss
        self.criterion = PretrainLoss(config)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir='logs/pretrain')
        
        self.best_loss = float('inf')
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train one epoch with masked modeling.
        """
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Pretrain Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            original = batch['original'].to(self.device)
            masked = batch['masked'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(masked, mode='reconstruction')
            
            # Calculate loss
            batch_dict = {
                'original': original,
                'mask': mask
            }
            losses = self.criterion(outputs, batch_dict, mode='reconstruction')
            loss = losses['total']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip']
            )
            
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            # Tensorboard logging
            global_step = epoch * len(dataloader) + batch_idx
            self.writer.add_scalar('Pretrain/Loss', loss.item(), global_step)
        
        avg_loss = total_loss / len(dataloader)
        return {'loss': avg_loss}
    
    def train_contrastive_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train one epoch with contrastive learning.
        Uses data augmentation (noise injection) to create positive pairs.
        """
        self.model.train()
        total_loss = 0.0
        
        pbar = tqdm(dataloader, desc=f"Contrastive Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            original = batch['original'].to(self.device)
            
            # Create two augmented views
            aug1 = self._augment(original)
            aug2 = self._augment(original)
            
            # Forward pass for both views
            self.optimizer.zero_grad()
            outputs1 = self.model(aug1, mode='contrastive')
            outputs2 = self.model(aug2, mode='contrastive')
            
            # Contrastive loss
            loss = self.criterion.contrastive_loss(
                outputs1['projected'],
                outputs2['projected']
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip']
            )
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
            global_step = epoch * len(dataloader) + batch_idx
            self.writer.add_scalar('Contrastive/Loss', loss.item(), global_step)
        
        avg_loss = total_loss / len(dataloader)
        return {'loss': avg_loss}
    
    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simple data augmentation for time series:
        - Add Gaussian noise
        - Random masking
        - Scaling
        """
        x = x.clone()
        
        # Gaussian noise
        if torch.rand(1).item() > 0.5:
            noise = torch.randn_like(x) * 0.01
            x = x + noise
        
        # Random masking
        if torch.rand(1).item() > 0.5:
            mask = torch.rand_like(x) > 0.1
            x = x * mask
        
        # Scaling
        if torch.rand(1).item() > 0.5:
            scale = 0.9 + 0.2 * torch.rand(1).item()
            x = x * scale
        
        return x
    
    def train(self, dataloader: DataLoader, num_epochs: int, 
              use_contrastive: bool = True):
        """
        Full pretraining loop.
        
        Args:
            dataloader: Pretrain dataloader
            num_epochs: Number of epochs
            use_contrastive: Whether to use contrastive learning
        """
        print("=" * 80)
        print("PHASE 1: PRETRAINING")
        print("=" * 80)
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Masked modeling
            metrics = self.train_epoch(dataloader, epoch)
            
            # Contrastive learning (every other epoch)
            if use_contrastive and epoch % 2 == 0:
                cont_metrics = self.train_contrastive_epoch(dataloader, epoch)
                metrics['contrastive_loss'] = cont_metrics['loss']
            
            # Learning rate scheduling
            self.scheduler.step(metrics['loss'])
            
            # Save best model
            if metrics['loss'] < self.best_loss:
                self.best_loss = metrics['loss']
                self.save_checkpoint('model/pretrain_best.pt')
                print(f"✓ Best model saved (loss: {self.best_loss:.6f})")
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'model/pretrain_epoch_{epoch}.pt')
        
        print("\n" + "=" * 80)
        print("PRETRAINING COMPLETE")
        print("=" * 80)
        
        self.writer.close()
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_loss': self.best_loss
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        print(f"Checkpoint loaded from {path}")
