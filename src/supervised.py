"""
Natron Transformer - Phase 2: Supervised Fine-Tuning
Multi-task learning for buy/sell/direction/regime prediction
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional
import os

from model_natron import NatronTransformer
from losses import MultiTaskLoss, compute_accuracy, compute_f1_score


class SupervisedTrainer:
    """
    Trainer for supervised multi-task learning
    """
    
    def __init__(
        self,
        model: NatronTransformer,
        config: Dict,
        device: str = 'cuda'
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Supervised config
        self.supervised_config = config['training']['supervised']
        self.epochs = self.supervised_config['epochs']
        self.lr = self.supervised_config['learning_rate']
        self.weight_decay = self.supervised_config['weight_decay']
        self.freeze_encoder = self.supervised_config['freeze_encoder']
        
        # Loss weights
        loss_weights = self.supervised_config['loss_weights']
        
        # Class weights for imbalanced data
        direction_weights = torch.FloatTensor(
            self.supervised_config['class_weights']['direction']
        ).to(device)
        regime_weights = torch.FloatTensor(
            self.supervised_config['class_weights']['regime']
        ).to(device)
        
        # Multi-task loss
        self.criterion = MultiTaskLoss(
            loss_weights=loss_weights,
            direction_weights=direction_weights,
            regime_weights=regime_weights
        )
        
        # Freeze encoder if specified
        if self.freeze_encoder:
            self.model.freeze_encoder()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # TensorBoard
        self.writer = SummaryWriter(
            log_dir=os.path.join(config['logging']['tensorboard_dir'], 'supervised')
        )
        
        print(f"🔧 Supervised Trainer initialized")
        print(f"  Epochs: {self.epochs}")
        print(f"  Learning rate: {self.lr}")
        print(f"  Freeze encoder: {self.freeze_encoder}")
        print(f"  Loss weights: {loss_weights}")
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with average losses and metrics
        """
        self.model.train()
        
        total_losses = {
            'total': 0.0,
            'buy': 0.0,
            'sell': 0.0,
            'direction': 0.0,
            'regime': 0.0
        }
        
        metrics = {
            'buy_acc': 0.0,
            'sell_acc': 0.0,
            'direction_acc': 0.0,
            'regime_acc': 0.0,
            'buy_f1': 0.0,
            'sell_f1': 0.0
        }
        
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Train Epoch {epoch}/{self.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Unpack batch
            sequences, buy_labels, sell_labels, direction_labels, regime_labels = batch
            
            sequences = sequences.to(self.device)
            buy_labels = buy_labels.to(self.device)
            sell_labels = sell_labels.to(self.device)
            direction_labels = direction_labels.to(self.device)
            regime_labels = regime_labels.to(self.device)
            
            # Forward pass
            predictions = self.model(sequences)
            
            # Compute loss
            targets = {
                'buy': buy_labels,
                'sell': sell_labels,
                'direction': direction_labels,
                'regime': regime_labels
            }
            
            losses = self.criterion(predictions, targets)
            loss = losses['total']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track losses
            for key in total_losses:
                total_losses[key] += losses[key].item()
            
            # Compute metrics
            with torch.no_grad():
                metrics['buy_acc'] += compute_accuracy(predictions['buy'], buy_labels)
                metrics['sell_acc'] += compute_accuracy(predictions['sell'], sell_labels)
                metrics['direction_acc'] += compute_accuracy(predictions['direction'], direction_labels)
                metrics['regime_acc'] += compute_accuracy(predictions['regime'], regime_labels)
                metrics['buy_f1'] += compute_f1_score(predictions['buy'], buy_labels)
                metrics['sell_f1'] += compute_f1_score(predictions['sell'], sell_labels)
            
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'buy_acc': metrics['buy_acc'] / num_batches,
                'dir_acc': metrics['direction_acc'] / num_batches
            })
            
            # Log to TensorBoard
            global_step = epoch * len(dataloader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/BuyAcc', metrics['buy_acc'] / num_batches, global_step)
        
        # Average metrics
        for key in total_losses:
            total_losses[key] /= num_batches
        
        for key in metrics:
            metrics[key] /= num_batches
        
        return {**total_losses, **metrics}
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Validate model
        
        Args:
            dataloader: Validation data loader
            
        Returns:
            Dictionary with losses and metrics
        """
        self.model.eval()
        
        total_losses = {
            'total': 0.0,
            'buy': 0.0,
            'sell': 0.0,
            'direction': 0.0,
            'regime': 0.0
        }
        
        metrics = {
            'buy_acc': 0.0,
            'sell_acc': 0.0,
            'direction_acc': 0.0,
            'regime_acc': 0.0,
            'buy_f1': 0.0,
            'sell_f1': 0.0
        }
        
        num_batches = 0
        
        pbar = tqdm(dataloader, desc="Validation")
        
        for batch in pbar:
            # Unpack batch
            sequences, buy_labels, sell_labels, direction_labels, regime_labels = batch
            
            sequences = sequences.to(self.device)
            buy_labels = buy_labels.to(self.device)
            sell_labels = sell_labels.to(self.device)
            direction_labels = direction_labels.to(self.device)
            regime_labels = regime_labels.to(self.device)
            
            # Forward pass
            predictions = self.model(sequences)
            
            # Compute loss
            targets = {
                'buy': buy_labels,
                'sell': sell_labels,
                'direction': direction_labels,
                'regime': regime_labels
            }
            
            losses = self.criterion(predictions, targets)
            
            # Track losses
            for key in total_losses:
                total_losses[key] += losses[key].item()
            
            # Compute metrics
            metrics['buy_acc'] += compute_accuracy(predictions['buy'], buy_labels)
            metrics['sell_acc'] += compute_accuracy(predictions['sell'], sell_labels)
            metrics['direction_acc'] += compute_accuracy(predictions['direction'], direction_labels)
            metrics['regime_acc'] += compute_accuracy(predictions['regime'], regime_labels)
            metrics['buy_f1'] += compute_f1_score(predictions['buy'], buy_labels)
            metrics['sell_f1'] += compute_f1_score(predictions['sell'], sell_labels)
            
            num_batches += 1
        
        # Average metrics
        for key in total_losses:
            total_losses[key] /= num_batches
        
        for key in metrics:
            metrics[key] /= num_batches
        
        return {**total_losses, **metrics}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader
    ) -> NatronTransformer:
        """
        Full supervised training loop
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Trained model
        """
        print(f"\n🚀 Starting Supervised Training...")
        print(f"  Total epochs: {self.epochs}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")
        
        best_val_loss = float('inf')
        checkpoint_dir = self.config['paths']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(1, self.epochs + 1):
            # Train epoch
            train_metrics = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log metrics
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics['total']:.4f} | Val Loss: {val_metrics['total']:.4f}")
            print(f"  Buy Acc: {train_metrics['buy_acc']:.4f} | {val_metrics['buy_acc']:.4f}")
            print(f"  Sell Acc: {train_metrics['sell_acc']:.4f} | {val_metrics['sell_acc']:.4f}")
            print(f"  Direction Acc: {train_metrics['direction_acc']:.4f} | {val_metrics['direction_acc']:.4f}")
            print(f"  Regime Acc: {train_metrics['regime_acc']:.4f} | {val_metrics['regime_acc']:.4f}")
            print(f"  Buy F1: {train_metrics['buy_f1']:.4f} | {val_metrics['buy_f1']:.4f}")
            
            # TensorBoard logging
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'Val/{key}', value, epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['total'])
            
            # Save best model
            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                checkpoint_path = os.path.join(checkpoint_dir, 'supervised_best.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': best_val_loss,
                    'val_metrics': val_metrics
                }, checkpoint_path)
                print(f"  💾 Best model saved (val_loss: {best_val_loss:.4f})")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'supervised_epoch_{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['total']
                }, checkpoint_path)
                print(f"  💾 Checkpoint saved: epoch {epoch}")
        
        print(f"\n✅ Supervised training complete!")
        print(f"  Best val loss: {best_val_loss:.4f}")
        
        # Load best model
        best_checkpoint = os.path.join(checkpoint_dir, 'supervised_best.pt')
        checkpoint = torch.load(best_checkpoint)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.writer.close()
        
        return self.model


def run_supervised_training(
    model: NatronTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict,
    device: str = 'cuda'
) -> NatronTransformer:
    """
    Convenience function to run supervised training
    
    Args:
        model: Natron Transformer model
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration dictionary
        device: Device to train on
        
    Returns:
        Trained model
    """
    trainer = SupervisedTrainer(model, config, device)
    trained_model = trainer.train(train_loader, val_loader)
    return trained_model


if __name__ == "__main__":
    # Test supervised training
    print("🧪 Testing Supervised Training...")
    
    import yaml
    from model_natron import create_model
    from dataset_loader import SequenceDataset
    from torch.utils.data import DataLoader
    
    # Load config
    with open('/workspace/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dummy data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_samples = 500
    
    sequences = np.random.randn(n_samples, 96, 100)
    buy_labels = np.random.randint(0, 2, n_samples)
    sell_labels = np.random.randint(0, 2, n_samples)
    direction_labels = np.random.randint(0, 2, n_samples)
    regime_labels = np.random.randint(0, 6, n_samples)
    
    dataset = SequenceDataset(sequences, buy_labels, sell_labels, direction_labels, regime_labels)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=32, shuffle=False)
    
    # Create model
    model = create_model(config, device)
    
    # Run 2 epochs for testing
    config['training']['supervised']['epochs'] = 2
    trainer = SupervisedTrainer(model, config, device)
    
    print("\n🔄 Running test supervised training (2 epochs)...")
    trained_model = trainer.train(train_loader, val_loader)
    
    print("\n✅ Supervised training test successful!")
