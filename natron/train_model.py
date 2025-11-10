"""
Model Training Script for Natron AI Trading System
Trains multi-head transformer with mixed precision and GPU support.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import yaml
from pathlib import Path
import json
from datetime import datetime
from tqdm import tqdm
import os

from model_natron import NatronTransformer, NatronLoss
from dataset_loader import create_dataloaders


class Trainer:
    """
    Trainer class for Natron model.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize trainer from config file.
        
        Args:
            config_path: Path to training configuration YAML
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"CUDA Version: {torch.version.cuda}")
        
        # Create output directory
        self.output_dir = Path(self.config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Mixed precision
        self.use_amp = self.config.get('use_mixed_precision', True) and self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        
        # Model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training state
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        self.epoch = 0
    
    def build_model(self, n_features: int):
        """
        Build model from config.
        
        Args:
            n_features: Number of input features
        """
        model_config = self.config['model']
        self.model = NatronTransformer(
            n_features=n_features,
            d_model=model_config.get('d_model', 128),
            nhead=model_config.get('nhead', 8),
            num_layers=model_config.get('num_layers', 6),
            dim_feedforward=model_config.get('dim_feedforward', 512),
            dropout=model_config.get('dropout', 0.1),
            max_seq_len=model_config.get('max_seq_len', 96),
            n_regime_classes=model_config.get('n_regime_classes', 6)
        ).to(self.device)
        
        print(f"\nModel architecture:")
        print(f"  Input features: {n_features}")
        print(f"  Model dimension: {model_config.get('d_model', 128)}")
        print(f"  Attention heads: {model_config.get('nhead', 8)}")
        print(f"  Encoder layers: {model_config.get('num_layers', 6)}")
        print(f"  Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def build_optimizer_and_scheduler(self):
        """Build optimizer and learning rate scheduler."""
        opt_config = self.config['optimizer']
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_config.get('lr', 1e-4),
            weight_decay=opt_config.get('weight_decay', 1e-5),
            betas=(opt_config.get('beta1', 0.9), opt_config.get('beta2', 0.999))
        )
        
        scheduler_config = self.config.get('scheduler', {})
        if scheduler_config.get('type') == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=scheduler_config.get('T_max', self.config['training']['epochs']),
                eta_min=scheduler_config.get('eta_min', 1e-6)
            )
        else:
            self.scheduler = None
    
    def build_criterion(self, class_weights: Optional[torch.Tensor] = None):
        """
        Build loss criterion.
        
        Args:
            class_weights: Optional class weights for regime classification
        """
        loss_config = self.config['loss']
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        
        self.criterion = NatronLoss(
            regime_weight=loss_config.get('regime_weight', 1.0),
            context_weight=loss_config.get('context_weight', 0.5),
            forecast_weight=loss_config.get('forecast_weight', 1.0),
            class_weights=class_weights
        )
    
    def train_epoch(self, train_loader) -> dict:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        loss_components = {'regime': 0.0, 'context': 0.0, 'forecast': 0.0}
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch+1}/{self.config['training']['epochs']} [Train]")
        
        for batch in pbar:
            X, y_regime, y_context, y_forecast = [b.to(self.device) for b in batch]
            
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    regime_logits, context_score, forecast_logits = self.model(X)
                    loss, loss_dict = self.criterion(
                        regime_logits, context_score, forecast_logits,
                        y_regime, y_context, y_forecast
                    )
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                regime_logits, context_score, forecast_logits = self.model(X)
                loss, loss_dict = self.criterion(
                    regime_logits, context_score, forecast_logits,
                    y_regime, y_context, y_forecast
                )
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            for key in loss_components:
                loss_components[key] += loss_dict[key]
            n_batches += 1
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'r': f"{loss_dict['regime']:.4f}",
                'c': f"{loss_dict['context']:.4f}",
                'f': f"{loss_dict['forecast']:.4f}"
            })
        
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in loss_components.items()}
        
        return {'total': avg_loss, **avg_components}
    
    def validate(self, val_loader) -> dict:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        loss_components = {'regime': 0.0, 'context': 0.0, 'forecast': 0.0}
        n_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                X, y_regime, y_context, y_forecast = [b.to(self.device) for b in batch]
                
                regime_logits, context_score, forecast_logits = self.model(X)
                loss, loss_dict = self.criterion(
                    regime_logits, context_score, forecast_logits,
                    y_regime, y_context, y_forecast
                )
                
                total_loss += loss.item()
                for key in loss_components:
                    loss_components[key] += loss_dict[key]
                n_batches += 1
        
        avg_loss = total_loss / n_batches
        avg_components = {k: v / n_batches for k, v in loss_components.items()}
        
        return {'total': avg_loss, **avg_components}
    
    def save_checkpoint(self, is_best: bool = False):
        """
        Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save latest checkpoint
        checkpoint_path = self.output_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.output_dir / 'natron_v6.pt'
            torch.save(checkpoint, best_path)
            print(f"\n✓ Saved best model to {best_path}")
    
    def train(self, train_loader, val_loader):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        num_epochs = self.config['training']['epochs']
        
        print(f"\n{'='*60}")
        print("NATRON MODEL TRAINING - Starting")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics)
            
            # Validate
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics)
            
            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {train_metrics['total']:.6f} "
                  f"(R:{train_metrics['regime']:.4f} C:{train_metrics['context']:.4f} F:{train_metrics['forecast']:.4f})")
            print(f"  Val Loss:   {val_metrics['total']:.6f} "
                  f"(R:{val_metrics['regime']:.4f} C:{val_metrics['context']:.4f} F:{val_metrics['forecast']:.4f})")
            if self.scheduler is not None:
                print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save checkpoint
            is_best = val_metrics['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['total']
            
            self.save_checkpoint(is_best=is_best)
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE!")
        print(f"Best validation loss: {self.best_val_loss:.6f}")
        print(f"{'='*60}\n")
        
        # Save training history
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }
        history_path = self.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Natron Model')
    parser.add_argument('--config', type=str, default='config_train.yaml',
                       help='Path to training config YAML')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to processed data CSV')
    args = parser.parse_args()
    
    # Load processed data
    print("Loading processed data...")
    processed_df = pd.read_csv(args.data, index_col=0)
    
    # Extract features and labels
    feature_cols = [col for col in processed_df.columns 
                   if col not in ['open', 'high', 'low', 'close', 'volume', 'regime']]
    feature_df = processed_df[feature_cols]
    labels = processed_df['regime'].astype(int)
    
    # Create sequences
    from data_pipeline import DataPipeline
    pipeline = DataPipeline()
    X, y_regime, y_context, y_forecast = pipeline.create_sequences(
        feature_df, labels,
        sequence_length=96,
        forecast_horizon=5
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader, norm_stats = create_dataloaders(
        X, y_regime, y_context, y_forecast,
        train_ratio=0.7,
        val_ratio=0.15,
        batch_size=32
    )
    
    # Initialize trainer
    trainer = Trainer(args.config)
    trainer.build_model(n_features=X.shape[-1])
    trainer.build_optimizer_and_scheduler()
    
    # Compute class weights for regime classification
    regime_counts = np.bincount(y_regime)
    class_weights = torch.tensor(1.0 / (regime_counts + 1e-6), dtype=torch.float32)
    trainer.build_criterion(class_weights=class_weights)
    
    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    import pandas as pd
    main()
