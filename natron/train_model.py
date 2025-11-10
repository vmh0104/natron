"""
Natron Model Training Script
Trains multi-head transformer with mixed precision and GPU support.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml
import pandas as pd
from pathlib import Path
import logging
from tqdm import tqdm
import json
from datetime import datetime

from model_natron import NatronTransformer, NatronLoss
from dataset_loader import create_dataloaders

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class Trainer:
    """Trainer class for Natron model."""
    
    def __init__(self, config_path: str):
        """
        Initialize trainer from config.
        
        Args:
            config_path: Path to training config YAML
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Mixed precision
        self.use_amp = self.config.get('use_mixed_precision', True) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Load data
        logger.info("Loading processed data...")
        df = pd.read_csv(self.config['data_path'])
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        self.train_loader, self.val_loader, self.test_loader = create_dataloaders(
            df=df,
            train_ratio=self.config.get('train_ratio', 0.7),
            val_ratio=self.config.get('val_ratio', 0.15),
            sequence_length=self.config.get('sequence_length', 96),
            forecast_horizon=self.config.get('forecast_horizon', 5),
            batch_size=self.config.get('batch_size', 32),
            num_workers=self.config.get('num_workers', 4)
        )
        
        # Get feature dimension from first batch
        sample_batch = next(iter(self.train_loader))
        input_dim = sample_batch['features'].shape[2]
        
        # Initialize model
        logger.info("Initializing model...")
        self.model = NatronTransformer(
            input_dim=input_dim,
            d_model=self.config.get('d_model', 256),
            nhead=self.config.get('nhead', 8),
            num_layers=self.config.get('num_layers', 6),
            dim_feedforward=self.config.get('dim_feedforward', 1024),
            dropout=self.config.get('dropout', 0.1),
            max_seq_len=self.config.get('sequence_length', 96),
            num_regime_classes=self.config.get('num_regime_classes', 6),
            num_forecast_classes=self.config.get('num_forecast_classes', 2)
        ).to(self.device)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Loss function
        self.criterion = NatronLoss(
            regime_weight=self.config.get('regime_weight', 1.0),
            context_weight=self.config.get('context_weight', 0.5),
            forecast_weight=self.config.get('forecast_weight', 1.0)
        )
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.get('num_epochs', 100),
            eta_min=self.config.get('min_lr', 1e-6)
        )
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = []
        
        # Output paths
        self.output_dir = Path(self.config.get('output_dir', 'checkpoints'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.output_dir / self.config.get('model_name', 'natron_v6.pt')
    
    def train_epoch(self) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components = {'regime': 0.0, 'context': 0.0, 'forecast': 0.0}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch in pbar:
            # Move to device
            features = batch['features'].to(self.device)
            targets = {
                'regime': batch['regime'].to(self.device),
                'context': batch['context'].to(self.device),
                'forecast': batch['forecast'].to(self.device)
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    predictions = self.model(features)
                    loss, loss_dict = self.criterion(predictions, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                predictions = self.model(features)
                loss, loss_dict = self.criterion(predictions, targets)
                loss.backward()
                self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += loss_dict[key]
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'regime': f"{loss_dict['regime']:.4f}",
                'forecast': f"{loss_dict['forecast']:.4f}"
            })
        
        # Average losses
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return {
            'total_loss': avg_loss,
            **loss_components
        }
    
    def validate(self) -> dict:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        loss_components = {'regime': 0.0, 'context': 0.0, 'forecast': 0.0}
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                features = batch['features'].to(self.device)
                targets = {
                    'regime': batch['regime'].to(self.device),
                    'context': batch['context'].to(self.device),
                    'forecast': batch['forecast'].to(self.device)
                }
                
                if self.use_amp:
                    with autocast():
                        predictions = self.model(features)
                        loss, loss_dict = self.criterion(predictions, targets)
                else:
                    predictions = self.model(features)
                    loss, loss_dict = self.criterion(predictions, targets)
                
                total_loss += loss.item()
                for key in loss_components:
                    loss_components[key] += loss_dict[key]
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return {
            'total_loss': avg_loss,
            **loss_components
        }
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save latest
        latest_path = self.output_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)
        
        # Save best
        if is_best:
            torch.save(checkpoint, self.model_path)
            logger.info(f"Saved best model to {self.model_path}")
    
    def train(self):
        """Main training loop."""
        num_epochs = self.config.get('num_epochs', 100)
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Learning rate step
            self.scheduler.step()
            
            # Log metrics
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_metrics['total_loss']:.4f} | "
                f"Val Loss: {val_metrics['total_loss']:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )
            
            # Save history
            epoch_history = {
                'epoch': epoch + 1,
                'train': train_metrics,
                'val': val_metrics,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.train_history.append(epoch_history)
            
            # Save checkpoint
            is_best = val_metrics['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total_loss']
            
            self.save_checkpoint(is_best=is_best)
        
        # Save training history
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        logger.info("Training completed!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")


if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config_train.yaml"
    
    trainer = Trainer(config_path)
    trainer.train()
