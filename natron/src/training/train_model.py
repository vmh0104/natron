"""
Training Script for Natron Model

This script trains the multi-head transformer model with mixed precision
and GPU acceleration.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import json
import logging
from datetime import datetime
from tqdm import tqdm
import os

from .model_natron import NatronTransformer, NatronLoss
from .dataset_loader import create_dataloaders
from ..data_pipeline.data_pipeline import DataPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Trainer class for Natron model.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to training configuration YAML
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create output directories
        self.model_dir = Path(self.config['paths']['model_dir'])
        self.log_dir = Path(self.config['paths']['log_dir'])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.criterion = None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_history = []
    
    def load_data(self):
        """Load and prepare data."""
        logger.info("Loading data...")
        
        # Load processed data
        processed_path = self.config['data']['processed_path']
        df = pd.read_csv(processed_path, index_col=0, parse_dates=True)
        
        # Get feature columns (exclude targets and metadata)
        exclude_cols = ['regime', 'regime_name', 'target', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        logger.info(f"Found {len(feature_cols)} features")
        
        return df, feature_cols
    
    def build_model(self, input_dim: int):
        """Build the model."""
        logger.info("Building model...")
        
        model_config = self.config['model']
        self.model = NatronTransformer(
            input_dim=input_dim,
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            num_layers=model_config['num_layers'],
            dim_feedforward=model_config['dim_feedforward'],
            dropout=model_config['dropout'],
            seq_len=model_config['seq_len'],
            num_regime_classes=model_config['num_regime_classes']
        ).to(self.device)
        
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Loss function
        loss_config = self.config['training']['loss_weights']
        self.criterion = NatronLoss(
            regime_weight=loss_config['regime'],
            context_weight=loss_config['context'],
            forecast_weight=loss_config['forecast']
        )
        
        # Optimizer
        optimizer_config = self.config['training']['optimizer']
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config['weight_decay']
        )
        
        # Learning rate scheduler
        scheduler_config = self.config['training']['scheduler']
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=scheduler_config['T_max'],
            eta_min=scheduler_config['eta_min']
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components = {'regime': 0.0, 'context': 0.0, 'forecast': 0.0}
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch+1}")
        for batch in pbar:
            # Move to device
            features = batch['features'].to(self.device)
            regime_target = batch['regime_target'].to(self.device)
            forecast_target = batch['forecast_target'].to(self.device)
            context_target = batch['context_target'].to(self.device)
            
            # Forward pass with mixed precision
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                with autocast():
                    regime_pred, context_pred, forecast_pred = self.model(features)
                    loss, loss_dict = self.criterion(
                        regime_pred, context_pred, forecast_pred,
                        regime_target, context_target, forecast_target
                    )
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                regime_pred, context_pred, forecast_pred = self.model(features)
                loss, loss_dict = self.criterion(
                    regime_pred, context_pred, forecast_pred,
                    regime_target, context_target, forecast_target
                )
                loss.backward()
                self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            loss_components['regime'] += loss_dict['regime']
            loss_components['context'] += loss_dict['context']
            loss_components['forecast'] += loss_dict['forecast']
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'regime': f"{loss_dict['regime']:.4f}",
                'context': f"{loss_dict['context']:.4f}",
                'forecast': f"{loss_dict['forecast']:.4f}"
            })
        
        # Average losses
        n_batches = len(train_loader)
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in loss_components.items()}
        
        return avg_loss, avg_components
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        loss_components = {'regime': 0.0, 'context': 0.0, 'forecast': 0.0}
        
        # Accuracy tracking
        regime_correct = 0
        forecast_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                features = batch['features'].to(self.device)
                regime_target = batch['regime_target'].to(self.device)
                forecast_target = batch['forecast_target'].to(self.device)
                context_target = batch['context_target'].to(self.device)
                
                # Forward pass
                if self.scaler is not None:
                    with autocast():
                        regime_pred, context_pred, forecast_pred = self.model(features)
                        loss, loss_dict = self.criterion(
                            regime_pred, context_pred, forecast_pred,
                            regime_target, context_target, forecast_target
                        )
                else:
                    regime_pred, context_pred, forecast_pred = self.model(features)
                    loss, loss_dict = self.criterion(
                        regime_pred, context_pred, forecast_pred,
                        regime_target, context_target, forecast_target
                    )
                
                # Update statistics
                total_loss += loss.item()
                loss_components['regime'] += loss_dict['regime']
                loss_components['context'] += loss_dict['context']
                loss_components['forecast'] += loss_dict['forecast']
                
                # Compute accuracies
                regime_pred_class = regime_pred.argmax(dim=1)
                forecast_pred_class = forecast_pred.argmax(dim=1)
                
                regime_correct += (regime_pred_class == regime_target).sum().item()
                forecast_correct += (forecast_pred_class == forecast_target).sum().item()
                total_samples += len(regime_target)
        
        # Average metrics
        n_batches = len(val_loader)
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in loss_components.items()}
        regime_acc = regime_correct / total_samples
        forecast_acc = forecast_correct / total_samples
        
        return avg_loss, avg_components, regime_acc, forecast_acc
    
    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_history': self.train_history,
            'config': self.config
        }
        
        # Save latest checkpoint
        checkpoint_path = self.model_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.model_dir / self.config['paths']['model_name']
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def train(self):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Load data
        df, feature_cols = self.load_data()
        
        # Create dataloaders
        train_config = self.config['training']
        train_loader, val_loader, test_loader = create_dataloaders(
            df=df,
            feature_cols=feature_cols,
            train_ratio=train_config['train_ratio'],
            val_ratio=train_config['val_ratio'],
            seq_len=self.config['model']['seq_len'],
            batch_size=train_config['batch_size'],
            num_workers=train_config.get('num_workers', 4)
        )
        
        # Build model
        self.build_model(input_dim=len(feature_cols))
        
        # Training loop
        num_epochs = train_config['num_epochs']
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_loss, train_components = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_components, regime_acc, forecast_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Log metrics
            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_components': train_components,
                'val_loss': val_loss,
                'val_components': val_components,
                'regime_accuracy': regime_acc,
                'forecast_accuracy': forecast_acc,
                'learning_rate': current_lr
            }
            self.train_history.append(epoch_log)
            
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Regime Acc: {regime_acc:.4f} | "
                f"Forecast Acc: {forecast_acc:.4f} | "
                f"LR: {current_lr:.6f}"
            )
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(is_best=is_best)
        
        # Save training history
        history_path = self.log_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        
        logger.info("Training complete!")
        logger.info(f"Best validation loss: {self.best_val_loss:.4f}")


if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/config_train.yaml"
    
    trainer = Trainer(config_path)
    trainer.train()
