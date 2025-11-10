"""
Phase 2: Supervised Fine-Tuning Pipeline
Multi-task learning for buy/sell/direction/regime prediction

Can optionally load pretrained weights from Phase 1
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import yaml
import os
import time
from tqdm import tqdm
from typing import Dict, Optional
import sys

from model_natron import NatronTransformer, NatronPretrainModel
from losses import MultiTaskLoss, MultiTaskFocalLoss, compute_metrics
from dataset_loader import DatasetManager
from feature_engine import FeatureEngine
from label_generator import LabelGenerator


class SupervisedTrainer:
    """Trainer for Phase 2: Supervised Fine-Tuning"""
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.patience_counter = 0
        
        # Create checkpoint directory
        os.makedirs(config['supervised']['checkpoint_dir'], exist_ok=True)
        
        # Create logs directory
        os.makedirs(config['logging']['save_dir'], exist_ok=True)
    
    def train_epoch(
        self,
        model: NatronTransformer,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: MultiTaskLoss,
        scaler: GradScaler,
        use_amp: bool = True,
        gradient_clip: float = 1.0
    ) -> Dict[str, float]:
        """Train one epoch"""
        model.train()
        
        total_loss = 0
        buy_loss_sum = 0
        sell_loss_sum = 0
        direction_loss_sum = 0
        regime_loss_sum = 0
        
        # Metrics
        buy_acc_sum = 0
        sell_acc_sum = 0
        direction_acc_sum = 0
        regime_acc_sum = 0
        
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        
        for sequences, labels in pbar:
            sequences = sequences.to(self.device)
            labels = {k: v.to(self.device) for k, v in labels.items()}
            
            # Forward pass with mixed precision
            with autocast(enabled=use_amp):
                outputs = model(sequences)
                losses = loss_fn(outputs, labels)
                loss = losses['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            scaler.step(optimizer)
            scaler.update()
            
            # Compute metrics
            with torch.no_grad():
                metrics = compute_metrics(outputs, labels)
            
            # Accumulate losses and metrics
            total_loss += loss.item()
            buy_loss_sum += losses['buy_loss'].item()
            sell_loss_sum += losses['sell_loss'].item()
            direction_loss_sum += losses['direction_loss'].item()
            regime_loss_sum += losses['regime_loss'].item()
            
            buy_acc_sum += metrics['buy_acc']
            sell_acc_sum += metrics['sell_acc']
            direction_acc_sum += metrics['direction_acc']
            regime_acc_sum += metrics['regime_acc']
            
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'dir_acc': f"{metrics['direction_acc']:.3f}",
                'reg_acc': f"{metrics['regime_acc']:.3f}"
            })
        
        return {
            'total_loss': total_loss / num_batches,
            'buy_loss': buy_loss_sum / num_batches,
            'sell_loss': sell_loss_sum / num_batches,
            'direction_loss': direction_loss_sum / num_batches,
            'regime_loss': regime_loss_sum / num_batches,
            'buy_acc': buy_acc_sum / num_batches,
            'sell_acc': sell_acc_sum / num_batches,
            'direction_acc': direction_acc_sum / num_batches,
            'regime_acc': regime_acc_sum / num_batches,
            'avg_acc': (buy_acc_sum + sell_acc_sum + direction_acc_sum + regime_acc_sum) / (4 * num_batches)
        }
    
    def validate(
        self,
        model: NatronTransformer,
        val_loader: DataLoader,
        loss_fn: MultiTaskLoss
    ) -> Dict[str, float]:
        """Validate model"""
        model.eval()
        
        total_loss = 0
        buy_loss_sum = 0
        sell_loss_sum = 0
        direction_loss_sum = 0
        regime_loss_sum = 0
        
        buy_acc_sum = 0
        sell_acc_sum = 0
        direction_acc_sum = 0
        regime_acc_sum = 0
        
        num_batches = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(self.device)
                labels = {k: v.to(self.device) for k, v in labels.items()}
                
                # Forward pass
                outputs = model(sequences)
                losses = loss_fn(outputs, labels)
                metrics = compute_metrics(outputs, labels)
                
                total_loss += losses['total_loss'].item()
                buy_loss_sum += losses['buy_loss'].item()
                sell_loss_sum += losses['sell_loss'].item()
                direction_loss_sum += losses['direction_loss'].item()
                regime_loss_sum += losses['regime_loss'].item()
                
                buy_acc_sum += metrics['buy_acc']
                sell_acc_sum += metrics['sell_acc']
                direction_acc_sum += metrics['direction_acc']
                regime_acc_sum += metrics['regime_acc']
                
                num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'buy_loss': buy_loss_sum / num_batches,
            'sell_loss': sell_loss_sum / num_batches,
            'direction_loss': direction_loss_sum / num_batches,
            'regime_loss': regime_loss_sum / num_batches,
            'buy_acc': buy_acc_sum / num_batches,
            'sell_acc': sell_acc_sum / num_batches,
            'direction_acc': direction_acc_sum / num_batches,
            'regime_acc': regime_acc_sum / num_batches,
            'avg_acc': (buy_acc_sum + sell_acc_sum + direction_acc_sum + regime_acc_sum) / (4 * num_batches)
        }
    
    def train(
        self,
        model: NatronTransformer,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        """Complete supervised training loop"""
        print("\n" + "="*60)
        print("🎯 Phase 2: Supervised Fine-Tuning Started")
        print("="*60)
        
        # Loss function
        loss_fn = MultiTaskLoss(
            buy_weight=self.config['supervised']['loss_weights']['buy'],
            sell_weight=self.config['supervised']['loss_weights']['sell'],
            direction_weight=self.config['supervised']['loss_weights']['direction'],
            regime_weight=self.config['supervised']['loss_weights']['regime']
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['supervised']['learning_rate'],
            weight_decay=self.config['supervised']['weight_decay']
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Mixed precision scaler
        scaler = GradScaler(enabled=self.config['hardware']['mixed_precision'])
        
        # Training loop
        num_epochs = self.config['supervised']['epochs']
        early_stopping_patience = self.config['supervised']['early_stopping_patience']
        
        for epoch in range(num_epochs):
            print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(
                model, train_loader, optimizer, loss_fn, scaler,
                use_amp=self.config['hardware']['mixed_precision'],
                gradient_clip=self.config['supervised']['gradient_clip']
            )
            
            # Validate
            val_metrics = self.validate(model, val_loader, loss_fn)
            
            # Print metrics
            print(f"\n📊 Train - Loss: {train_metrics['total_loss']:.4f} | "
                  f"Avg Acc: {train_metrics['avg_acc']:.4f}")
            print(f"   Buy: {train_metrics['buy_acc']:.3f} | "
                  f"Sell: {train_metrics['sell_acc']:.3f} | "
                  f"Dir: {train_metrics['direction_acc']:.3f} | "
                  f"Regime: {train_metrics['regime_acc']:.3f}")
            
            print(f"\n📊 Val - Loss: {val_metrics['total_loss']:.4f} | "
                  f"Avg Acc: {val_metrics['avg_acc']:.4f}")
            print(f"   Buy: {val_metrics['buy_acc']:.3f} | "
                  f"Sell: {val_metrics['sell_acc']:.3f} | "
                  f"Dir: {val_metrics['direction_acc']:.3f} | "
                  f"Regime: {val_metrics['regime_acc']:.3f}")
            
            # Learning rate scheduling
            scheduler.step(val_metrics['total_loss'])
            
            # Save best model (based on accuracy)
            if val_metrics['avg_acc'] > self.best_accuracy:
                self.best_accuracy = val_metrics['avg_acc']
                self.best_loss = val_metrics['total_loss']
                self.patience_counter = 0
                
                checkpoint_path = os.path.join(
                    self.config['supervised']['checkpoint_dir'],
                    'best_supervised.pt'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': self.best_loss,
                    'accuracy': self.best_accuracy,
                    'config': self.config
                }, checkpoint_path)
                print(f"💾 Saved best model: {checkpoint_path}")
            else:
                self.patience_counter += 1
            
            # Early stopping
            if self.patience_counter >= early_stopping_patience:
                print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
                break
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(
                    self.config['supervised']['checkpoint_dir'],
                    f'supervised_epoch_{epoch+1}.pt'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_metrics['total_loss'],
                    'accuracy': val_metrics['avg_acc'],
                    'config': self.config
                }, checkpoint_path)
        
        # Save final model as natron_v2.pt
        final_model_path = 'models/natron_v2.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': self.config,
            'best_accuracy': self.best_accuracy,
            'best_loss': self.best_loss
        }, final_model_path)
        
        print("\n" + "="*60)
        print("✅ Supervised Training Complete!")
        print(f"🏆 Best validation accuracy: {self.best_accuracy:.4f}")
        print(f"💾 Final model saved: {final_model_path}")
        print("="*60)


def load_pretrained_weights(
    natron_model: NatronTransformer,
    pretrain_checkpoint_path: str,
    device: str
):
    """Load pretrained weights into Natron model"""
    print(f"\n🔄 Loading pretrained weights from {pretrain_checkpoint_path}...")
    
    checkpoint = torch.load(pretrain_checkpoint_path, map_location=device)
    
    # Create pretrain model and load weights
    pretrain_model = NatronPretrainModel(
        num_features=natron_model.num_features,
        d_model=natron_model.d_model,
        nhead=checkpoint['config']['model']['nhead'],
        num_encoder_layers=checkpoint['config']['model']['num_encoder_layers'],
        dim_feedforward=checkpoint['config']['model']['dim_feedforward'],
        dropout=checkpoint['config']['model']['dropout'],
        activation=checkpoint['config']['model']['activation'],
        sequence_length=natron_model.sequence_length
    )
    
    pretrain_model.load_state_dict(checkpoint['model_state_dict'])
    
    # Transfer weights
    pretrain_model.transfer_to_natron(natron_model)
    
    print("✅ Pretrained weights loaded successfully!")


def main():
    """Main supervised training pipeline"""
    
    # Load config
    with open('/workspace/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Data preparation
    print("\n📂 Loading and preparing data...")
    feature_engine = FeatureEngine(verbose=True)
    label_generator = LabelGenerator(config.get('labels', {}))
    dataset_manager = DatasetManager(config)
    
    # Load data
    df, scaler = dataset_manager.load_and_prepare_data(
        config['data']['csv_path'],
        feature_engine,
        label_generator
    )
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = dataset_manager.create_datasets(
        df,
        train_split=config['data']['train_split'],
        val_split=config['data']['val_split'],
        test_split=config['data']['test_split']
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = dataset_manager.create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=config['supervised']['batch_size'],
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    # Create model
    print("\n🏗️  Building Natron Transformer...")
    model = NatronTransformer(
        num_features=config['data']['feature_count'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        activation=config['model']['activation'],
        buy_head_dims=config['model']['buy_head_dims'],
        sell_head_dims=config['model']['sell_head_dims'],
        direction_head_dims=config['model']['direction_head_dims'],
        regime_head_dims=config['model']['regime_head_dims'],
        sequence_length=config['data']['sequence_length']
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {num_params:,} parameters")
    
    # Load pretrained weights if available
    pretrain_checkpoint = os.path.join(
        config['pretrain']['checkpoint_dir'],
        'best_pretrain.pt'
    )
    
    if os.path.exists(pretrain_checkpoint):
        load_pretrained_weights(model, pretrain_checkpoint, device)
        
        # Optionally freeze encoder
        if config['supervised'].get('freeze_encoder', False):
            model.freeze_encoder()
    else:
        print("⚠️  No pretrained checkpoint found, training from scratch")
    
    # Train
    trainer = SupervisedTrainer(config, device=device)
    trainer.train(model, train_loader, val_loader)
    
    print("\n🎉 Phase 2 (Supervised Training) pipeline complete!")


if __name__ == "__main__":
    main()
