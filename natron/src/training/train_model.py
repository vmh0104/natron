"""
Model Training Script for Natron
Trains multi-head transformer with regime, context, and forecast heads.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
import logging
from tqdm import tqdm
import json
from datetime import datetime
from typing import Optional, Dict

from .model_natron import NatronTransformer
from .dataset_loader import create_data_loaders, NatronDataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NatronTrainer:
    """Trainer for Natron transformer model."""
    
    def __init__(self, config_path: str):
        """
        Initialize trainer.
        
        Args:
            config_path: Path to training configuration YAML
        """
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None  # For mixed precision
        self.best_val_loss = float('inf')
        self.train_history = []
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML."""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _create_model(self, input_dim: int) -> NatronTransformer:
        """Create Natron transformer model."""
        model_config = self.config['model']
        
        model = NatronTransformer(
            input_dim=input_dim,
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            num_layers=model_config['num_layers'],
            dim_feedforward=model_config['dim_feedforward'],
            dropout=model_config['dropout'],
            activation=model_config['activation'],
            regime_classes=model_config['regime_classes'],
            context_output_dim=model_config['context_output_dim'],
            forecast_classes=model_config['forecast_classes'],
            max_seq_len=self.config['data']['sequence_length']
        )
        
        return model.to(self.device)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer."""
        training_config = self.config['training']
        
        if training_config['optimizer'] == 'AdamW':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay']
            )
        else:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=training_config['learning_rate']
            )
        
        return optimizer
    
    def _create_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        training_config = self.config['training']
        
        if training_config['scheduler'] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=training_config['min_lr']
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )
        
        return scheduler
    
    def _compute_loss(self, outputs: tuple, targets: dict, loss_weights: dict) -> torch.Tensor:
        """
        Compute weighted loss.
        
        Args:
            outputs: Tuple of (regime_logits, context_score, forecast_logits)
            targets: Dictionary with regime, context, forecast targets
            loss_weights: Dictionary with loss weights
            
        Returns:
            Total loss
        """
        regime_logits, context_score, forecast_logits = outputs
        
        # Regime classification loss
        regime_loss = nn.CrossEntropyLoss()(regime_logits, targets['regime'])
        
        # Context strength loss (MSE)
        context_loss = nn.MSELoss()(context_score.squeeze(), targets['context'])
        
        # Forecast direction loss
        forecast_loss = nn.CrossEntropyLoss()(forecast_logits, targets['forecast'])
        
        # Weighted total loss
        total_loss = (
            loss_weights['regime'] * regime_loss +
            loss_weights['context'] * context_loss +
            loss_weights['forecast'] * forecast_loss
        )
        
        return total_loss, {
            'regime_loss': regime_loss.item(),
            'context_loss': context_loss.item(),
            'forecast_loss': forecast_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        loss_components = {'regime_loss': 0.0, 'context_loss': 0.0, 'forecast_loss': 0.0}
        num_batches = 0
        
        loss_weights = self.config['training']['loss_weights']
        use_mixed_precision = self.config['hardware']['mixed_precision']
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in pbar:
            features = batch['features'].to(self.device)
            targets = {
                'regime': batch['regime'].to(self.device),
                'context': batch['context'].to(self.device),
                'forecast': batch['forecast'].to(self.device)
            }
            
            self.optimizer.zero_grad()
            
            if use_mixed_precision:
                with autocast():
                    outputs = self.model(features)
                    loss, loss_dict = self._compute_loss(outputs, targets, loss_weights)
                
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training']['gradient_clip'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(features)
                loss, loss_dict = self._compute_loss(outputs, targets, loss_weights)
                loss.backward()
                
                # Gradient clipping
                if self.config['training']['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer.step()
            
            total_loss += loss_dict['total_loss']
            for key in loss_components:
                loss_components[key] += loss_dict[key]
            num_batches += 1
            
            pbar.set_postfix({'loss': loss_dict['total_loss']})
        
        avg_losses = {k: v / num_batches for k, v in loss_components.items()}
        avg_losses['total_loss'] = total_loss / num_batches
        
        return avg_losses
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0
        loss_components = {'regime_loss': 0.0, 'context_loss': 0.0, 'forecast_loss': 0.0}
        num_batches = 0
        
        loss_weights = self.config['training']['loss_weights']
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                features = batch['features'].to(self.device)
                targets = {
                    'regime': batch['regime'].to(self.device),
                    'context': batch['context'].to(self.device),
                    'forecast': batch['forecast'].to(self.device)
                }
                
                outputs = self.model(features)
                loss, loss_dict = self._compute_loss(outputs, targets, loss_weights)
                
                total_loss += loss_dict['total_loss']
                for key in loss_components:
                    loss_components[key] += loss_dict[key]
                num_batches += 1
        
        avg_losses = {k: v / num_batches for k, v in loss_components.items()}
        avg_losses['total_loss'] = total_loss / num_batches
        
        return avg_losses
    
    def train(self):
        """Main training loop."""
        logger.info("=" * 60)
        logger.info("NATRON MODEL TRAINING - Starting")
        logger.info("=" * 60)
        
        # Load data
        data_config = self.config['data']
        logger.info(f"Loading processed data from {data_config['processed_file']}")
        df = pd.read_csv(data_config['processed_file'])
        
        # Split data
        train_size = int(len(df) * data_config['train_split'])
        val_size = int(len(df) * data_config['val_split'])
        
        train_data = df[:train_size]
        val_data = df[train_size:train_size + val_size]
        test_data = df[train_size + val_size:] if data_config['test_split'] > 0 else None
        
        logger.info(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data) if test_data is not None else 0}")
        
        # Create data loaders
        train_loader, val_loader, test_loader, scaler = create_data_loaders(
            train_data,
            val_data,
            test_data,
            sequence_length=data_config['sequence_length'],
            batch_size=data_config['batch_size'],
            num_workers=data_config['num_workers'],
            pin_memory=data_config['pin_memory']
        )
        
        # Save scaler
        scaler_path = Path(self.config['output']['model_dir']) / 'scaler.pkl'
        NatronDataset.save_scaler(scaler, str(scaler_path))
        logger.info(f"Saved scaler to {scaler_path}")
        
        # Create model
        input_dim = len(train_loader.dataset.get_feature_names())
        logger.info(f"Input dimension: {input_dim}")
        
        self.model = self._create_model(input_dim)
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision scaler
        if self.config['hardware']['mixed_precision']:
            self.scaler = GradScaler()
        
        # Training loop
        epochs = self.config['training']['epochs']
        patience = self.config['training']['early_stopping_patience']
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"\nEpoch {epoch+1}/{epochs}")
            
            # Train
            train_losses = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_losses = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Log metrics
            logger.info(f"Train Loss: {train_losses['total_loss']:.4f} | "
                       f"Val Loss: {val_losses['total_loss']:.4f}")
            logger.info(f"  Regime: {train_losses['regime_loss']:.4f} / {val_losses['regime_loss']:.4f}")
            logger.info(f"  Context: {train_losses['context_loss']:.4f} / {val_losses['context_loss']:.4f}")
            logger.info(f"  Forecast: {train_losses['forecast_loss']:.4f} / {val_losses['forecast_loss']:.4f}")
            
            # Save history
            history_entry = {
                'epoch': epoch + 1,
                'train': train_losses,
                'val': val_losses,
                'lr': self.optimizer.param_groups[0]['lr']
            }
            self.train_history.append(history_entry)
            
            # Save best model
            if val_losses['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['total_loss']
                patience_counter = 0
                
                model_path = Path(self.config['output']['model_dir']) / self.config['output']['model_name']
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'config': self.config,
                    'epoch': epoch + 1,
                    'val_loss': val_losses['total_loss'],
                    'input_dim': input_dim
                }, str(model_path))
                logger.info(f"Saved best model to {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
        
        # Save training history
        history_path = Path(self.config['output']['log_dir']) / 'training_history.json'
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        logger.info(f"Saved training history to {history_path}")
        
        logger.info("=" * 60)
        logger.info("NATRON MODEL TRAINING - Complete")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Natron Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    
    args = parser.parse_args()
    
    trainer = NatronTrainer(args.config)
    trainer.train()


if __name__ == '__main__':
    main()
