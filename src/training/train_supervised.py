"""
Natron V2 Phase 2: Supervised Training Module
Multi-task supervised fine-tuning
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
from typing import Dict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.models.natron_transformer import NatronTransformer, PretrainTransformer
from src.models.losses import MultiTaskLoss


class SupervisedTrainer:
    """
    Manages Phase 2 supervised training with multi-task learning
    """
    
    def __init__(self, config: dict, device: str = 'cuda', 
                 pretrain_checkpoint: str = None):
        self.config = config
        self.device = device
        
        # Model
        self.model = NatronTransformer(config).to(device)
        
        # Load pretrained weights if available
        if pretrain_checkpoint and os.path.exists(pretrain_checkpoint):
            self._load_pretrained_weights(pretrain_checkpoint)
            print(f"✓ Loaded pretrained weights from {pretrain_checkpoint}")
        
        # Loss
        self.criterion = MultiTaskLoss(config)
        
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
        self.writer = SummaryWriter(log_dir='logs/supervised')
        
        self.best_val_loss = float('inf')
        self.best_metrics = {}
    
    def _load_pretrained_weights(self, checkpoint_path: str):
        """Load pretrained encoder weights"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Create pretrain model and load weights
        pretrain_model = PretrainTransformer(self.config).to(self.device)
        pretrain_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Transfer to supervised model
        pretrain_model.transfer_weights_to(self.model)
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0.0
        task_losses = {'buy': 0.0, 'sell': 0.0, 'direction': 0.0, 'regime': 0.0}
        
        pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}")
        
        for batch_idx, (sequences, labels) in enumerate(pbar):
            # Move to device
            sequences = sequences.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            predictions = self.model(sequences)
            
            # Calculate loss
            losses = self.criterion(predictions, labels)
            loss = losses['total']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['training']['gradient_clip']
            )
            self.optimizer.step()
            
            # Logging
            total_loss += loss.item()
            for task in ['buy', 'sell', 'direction', 'regime']:
                task_losses[task] += losses[task].item()
            
            pbar.set_postfix({'loss': loss.item()})
            
            # Tensorboard logging
            global_step = epoch * len(dataloader) + batch_idx
            self.writer.add_scalar('Train/Total_Loss', loss.item(), global_step)
            for task, task_loss in losses.items():
                if task != 'total':
                    self.writer.add_scalar(f'Train/{task}_loss', task_loss.item(), global_step)
        
        # Average losses
        avg_loss = total_loss / len(dataloader)
        avg_task_losses = {k: v / len(dataloader) for k, v in task_losses.items()}
        
        return {'total_loss': avg_loss, **avg_task_losses}
    
    def validate(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        task_losses = {'buy': 0.0, 'sell': 0.0, 'direction': 0.0, 'regime': 0.0}
        
        # Collect predictions for metrics
        all_preds = {
            'buy': [], 'sell': [], 'direction': [], 'regime': []
        }
        all_labels = {
            'buy': [], 'sell': [], 'direction': [], 'regime': []
        }
        
        with torch.no_grad():
            for sequences, labels in tqdm(dataloader, desc="Validation"):
                sequences = sequences.to(self.device)
                labels = {k: v.to(self.device) for k, v in labels.items()}
                
                # Forward pass
                predictions = self.model(sequences)
                
                # Calculate loss
                losses = self.criterion(predictions, labels)
                total_loss += losses['total'].item()
                
                for task in ['buy', 'sell', 'direction', 'regime']:
                    task_losses[task] += losses[task].item()
                
                # Collect predictions
                buy_pred = (torch.sigmoid(predictions['buy_logits']) > 0.5).float()
                sell_pred = (torch.sigmoid(predictions['sell_logits']) > 0.5).float()
                direction_pred = predictions['direction_logits'].argmax(dim=-1)
                regime_pred = predictions['regime_logits'].argmax(dim=-1)
                
                all_preds['buy'].append(buy_pred.cpu().numpy())
                all_preds['sell'].append(sell_pred.cpu().numpy())
                all_preds['direction'].append(direction_pred.cpu().numpy())
                all_preds['regime'].append(regime_pred.cpu().numpy())
                
                all_labels['buy'].append(labels['buy'].cpu().numpy())
                all_labels['sell'].append(labels['sell'].cpu().numpy())
                all_labels['direction'].append(labels['direction'].cpu().numpy())
                all_labels['regime'].append(labels['regime'].cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(dataloader)
        avg_task_losses = {k: v / len(dataloader) for k, v in task_losses.items()}
        
        # Concatenate all predictions
        for task in all_preds:
            all_preds[task] = np.concatenate(all_preds[task])
            all_labels[task] = np.concatenate(all_labels[task])
        
        # Calculate accuracy for each task
        metrics = {
            'val_loss': avg_loss,
            'buy_accuracy': accuracy_score(all_labels['buy'], all_preds['buy']),
            'sell_accuracy': accuracy_score(all_labels['sell'], all_preds['sell']),
            'direction_accuracy': accuracy_score(all_labels['direction'], all_preds['direction']),
            'regime_accuracy': accuracy_score(all_labels['regime'], all_preds['regime']),
            'direction_f1': f1_score(all_labels['direction'], all_preds['direction'], average='binary'),
            'regime_f1': f1_score(all_labels['regime'], all_preds['regime'], average='weighted')
        }
        
        # Tensorboard logging
        for metric_name, value in metrics.items():
            self.writer.add_scalar(f'Val/{metric_name}', value, epoch)
        
        return metrics
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int, freeze_encoder_epochs: int = 0):
        """
        Full supervised training loop.
        
        Args:
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of epochs
            freeze_encoder_epochs: Number of epochs to freeze encoder (fine-tuning)
        """
        print("=" * 80)
        print("PHASE 2: SUPERVISED TRAINING")
        print("=" * 80)
        
        # Optionally freeze encoder for initial epochs
        if freeze_encoder_epochs > 0:
            print(f"Freezing encoder for first {freeze_encoder_epochs} epochs")
            self.model.freeze_encoder()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            
            # Unfreeze encoder after initial epochs
            if epoch == freeze_encoder_epochs + 1:
                print("Unfreezing encoder")
                self.model.unfreeze_encoder()
            
            # Train
            train_metrics = self.train_epoch(train_loader, epoch)
            print(f"Train Loss: {train_metrics['total_loss']:.6f}")
            
            # Validate
            val_metrics = self.validate(val_loader, epoch)
            print(f"Val Loss: {val_metrics['val_loss']:.6f}")
            print(f"Direction Acc: {val_metrics['direction_accuracy']:.4f}, "
                  f"Regime Acc: {val_metrics['regime_accuracy']:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['val_loss'])
            
            # Save best model
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.best_metrics = val_metrics
                self.save_checkpoint('model/supervised_best.pt')
                print(f"✓ Best model saved (val_loss: {self.best_val_loss:.6f})")
            
            # Save periodic checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'model/supervised_epoch_{epoch}.pt')
        
        print("\n" + "=" * 80)
        print("SUPERVISED TRAINING COMPLETE")
        print("=" * 80)
        print(f"Best Validation Loss: {self.best_val_loss:.6f}")
        print(f"Best Metrics: {self.best_metrics}")
        
        self.writer.close()
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
            'best_metrics': self.best_metrics
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.best_metrics = checkpoint.get('best_metrics', {})
        print(f"Checkpoint loaded from {path}")
