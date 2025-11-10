"""
Natron Phase 2: Supervised Multi-Task Training
Author: Natron AI System
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import yaml
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.natron_transformer import NatronTransformer, NatronPretrainModel
from src.models.losses import MultiTaskLoss
from src.data.dataset_loader import NatronDataModule


class NatronSupervisedTrainer:
    """
    Supervised trainer for Natron Transformer.
    Multi-task learning: buy, sell, direction, regime.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.supervised_config = self.config['supervised']
        self.model_config = self.config['model']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {self.device}")
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=self.config['logging']['tensorboard_dir'] + '/supervised')
        
        # Metrics
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epoch = 0
        self.patience_counter = 0
    
    def create_model(self, num_features: int, load_pretrained: bool = False):
        """
        Create supervised model.
        
        Args:
            num_features: Number of input features
            load_pretrained: If True, load from pretrained checkpoint
        """
        if load_pretrained:
            print("📥 Loading pretrained encoder...")
            pretrain_path = self.config['pretrain']['checkpoint_path']
            
            # Load pretrained model
            pretrain_model = NatronPretrainModel(
                num_features=num_features,
                d_model=self.model_config['d_model'],
                nhead=self.model_config['nhead'],
                num_encoder_layers=self.model_config['num_encoder_layers'],
                dim_feedforward=self.model_config['dim_feedforward'],
                dropout=self.model_config['dropout'],
                activation=self.model_config['activation']
            )
            
            try:
                checkpoint = torch.load(pretrain_path.replace('.pt', '_best.pt'), map_location=self.device)
                pretrain_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded pretrained weights from {pretrain_path}")
                
                # Transfer to supervised model
                self.model = pretrain_model.transfer_to_supervised()
            except FileNotFoundError:
                print("⚠️ Pretrained checkpoint not found, training from scratch")
                self.model = NatronTransformer(
                    num_features=num_features,
                    d_model=self.model_config['d_model'],
                    nhead=self.model_config['nhead'],
                    num_encoder_layers=self.model_config['num_encoder_layers'],
                    dim_feedforward=self.model_config['dim_feedforward'],
                    dropout=self.model_config['dropout'],
                    activation=self.model_config['activation']
                )
        else:
            print("🔨 Training from scratch...")
            self.model = NatronTransformer(
                num_features=num_features,
                d_model=self.model_config['d_model'],
                nhead=self.model_config['nhead'],
                num_encoder_layers=self.model_config['num_encoder_layers'],
                dim_feedforward=self.model_config['dim_feedforward'],
                dropout=self.model_config['dropout'],
                activation=self.model_config['activation']
            )
        
        self.model = self.model.to(self.device)
        
        # Freeze encoder if specified
        if self.supervised_config.get('freeze_encoder', False):
            self.model.freeze_encoder()
        
        print(f"✅ Supervised model created")
        print(f"   Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def create_optimizer(self):
        """Create optimizer, scheduler, and loss function"""
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.supervised_config['learning_rate'],
            weight_decay=self.supervised_config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        self.loss_fn = MultiTaskLoss(
            task_weights=self.supervised_config['task_weights']
        )
        
        print(f"✅ Optimizer created: AdamW (lr={self.supervised_config['learning_rate']})")
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0
        task_losses = {'buy': 0, 'sell': 0, 'direction': 0, 'regime': 0}
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Train Epoch {self.epoch}")
        
        for batch in pbar:
            # Get data
            sequences = batch['sequence'].to(self.device)
            targets = {
                'buy': batch['buy'].squeeze().to(self.device),
                'sell': batch['sell'].squeeze().to(self.device),
                'direction': batch['direction'].squeeze().to(self.device),
                'regime': batch['regime'].squeeze().to(self.device)
            }
            
            # Forward pass
            predictions = self.model(sequences)
            
            # Calculate loss
            loss, losses = self.loss_fn(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += losses['total']
            for key in task_losses:
                task_losses[key] += losses[key]
            num_batches += 1
            
            pbar.set_postfix({
                'loss': losses['total'],
                'buy': losses['buy'],
                'sell': losses['sell']
            })
        
        # Average losses
        metrics = {
            'loss': total_loss / num_batches,
            'buy_loss': task_losses['buy'] / num_batches,
            'sell_loss': task_losses['sell'] / num_batches,
            'direction_loss': task_losses['direction'] / num_batches,
            'regime_loss': task_losses['regime'] / num_batches
        }
        
        return metrics
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        task_losses = {'buy': 0, 'sell': 0, 'direction': 0, 'regime': 0}
        num_batches = 0
        
        # Collect predictions and targets for metrics
        all_preds = {'buy': [], 'sell': [], 'direction': [], 'regime': []}
        all_targets = {'buy': [], 'sell': [], 'direction': [], 'regime': []}
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                sequences = batch['sequence'].to(self.device)
                targets = {
                    'buy': batch['buy'].squeeze().to(self.device),
                    'sell': batch['sell'].squeeze().to(self.device),
                    'direction': batch['direction'].squeeze().to(self.device),
                    'regime': batch['regime'].squeeze().to(self.device)
                }
                
                # Forward pass
                predictions = self.model(sequences)
                
                # Calculate loss
                loss, losses = self.loss_fn(predictions, targets)
                
                # Update metrics
                total_loss += losses['total']
                for key in task_losses:
                    task_losses[key] += losses[key]
                num_batches += 1
                
                # Collect predictions
                all_preds['buy'].extend((predictions['buy_prob'] > 0.5).cpu().numpy())
                all_preds['sell'].extend((predictions['sell_prob'] > 0.5).cpu().numpy())
                all_preds['direction'].extend(predictions['direction_logits'].argmax(dim=1).cpu().numpy())
                all_preds['regime'].extend(predictions['regime_logits'].argmax(dim=1).cpu().numpy())
                
                # Collect targets
                all_targets['buy'].extend(targets['buy'].cpu().numpy())
                all_targets['sell'].extend(targets['sell'].cpu().numpy())
                all_targets['direction'].extend(targets['direction'].cpu().numpy())
                all_targets['regime'].extend(targets['regime'].cpu().numpy())
        
        # Calculate accuracies
        buy_acc = accuracy_score(all_targets['buy'], all_preds['buy'])
        sell_acc = accuracy_score(all_targets['sell'], all_preds['sell'])
        direction_acc = accuracy_score(all_targets['direction'], all_preds['direction'])
        regime_acc = accuracy_score(all_targets['regime'], all_preds['regime'])
        
        metrics = {
            'loss': total_loss / num_batches,
            'buy_loss': task_losses['buy'] / num_batches,
            'sell_loss': task_losses['sell'] / num_batches,
            'direction_loss': task_losses['direction'] / num_batches,
            'regime_loss': task_losses['regime'] / num_batches,
            'buy_acc': buy_acc,
            'sell_acc': sell_acc,
            'direction_acc': direction_acc,
            'regime_acc': regime_acc,
            'avg_acc': (buy_acc + sell_acc + direction_acc + regime_acc) / 4
        }
        
        return metrics
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model to {best_path}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Full supervised training loop"""
        print(f"\n🚀 Starting Supervised Training...")
        print(f"   Epochs: {self.supervised_config['epochs']}")
        print(f"   Batch size: {self.supervised_config['batch_size']}")
        print(f"   Early stopping patience: {self.supervised_config['early_stopping_patience']}")
        
        for epoch in range(self.supervised_config['epochs']):
            self.epoch = epoch + 1
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log metrics
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train/{key}', value, self.epoch)
            
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, self.epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Save best model
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.best_val_acc = val_metrics['avg_acc']
                self.save_checkpoint(self.supervised_config['checkpoint_path'], is_best=True)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Regular checkpoint every 10 epochs
            if self.epoch % 10 == 0:
                self.save_checkpoint(
                    self.supervised_config['checkpoint_path'].replace('.pt', f'_epoch{self.epoch}.pt')
                )
            
            print(f"\n📊 Epoch {self.epoch} Summary:")
            print(f"   Train Loss: {train_metrics['loss']:.4f}")
            print(f"   Val Loss: {val_metrics['loss']:.4f}")
            print(f"   Val Accuracies:")
            print(f"     - Buy: {val_metrics['buy_acc']:.4f}")
            print(f"     - Sell: {val_metrics['sell_acc']:.4f}")
            print(f"     - Direction: {val_metrics['direction_acc']:.4f}")
            print(f"     - Regime: {val_metrics['regime_acc']:.4f}")
            print(f"     - Average: {val_metrics['avg_acc']:.4f}")
            print(f"   Best Val Loss: {self.best_val_loss:.4f}")
            print(f"   Best Val Acc: {self.best_val_acc:.4f}")
            
            # Early stopping
            if self.patience_counter >= self.supervised_config['early_stopping_patience']:
                print(f"\n⚠️ Early stopping triggered after {self.epoch} epochs")
                break
        
        print(f"\n✅ Supervised training completed!")
        self.writer.close()


def main():
    """Main supervised training function"""
    print("=" * 80)
    print("🧠 NATRON TRANSFORMER - PHASE 2: SUPERVISED MULTI-TASK TRAINING")
    print("=" * 80)
    
    # Load data
    data_module = NatronDataModule(config_path="config/config.yaml")
    pipeline_data = data_module.prepare_full_pipeline(mode='supervised')
    
    # Create trainer
    trainer = NatronSupervisedTrainer(config_path="config/config.yaml")
    trainer.create_model(
        num_features=pipeline_data['num_features'],
        load_pretrained=True  # Set to False to train from scratch
    )
    trainer.create_optimizer()
    
    # Train
    trainer.train(
        train_loader=pipeline_data['train_loader'],
        val_loader=pipeline_data['val_loader']
    )
    
    print("\n✅ Supervised model saved!")
    print(f"   Path: {trainer.supervised_config['checkpoint_path']}")


if __name__ == "__main__":
    main()
