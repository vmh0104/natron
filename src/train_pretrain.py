"""
Phase 1: Pretraining Pipeline
Unsupervised learning to build market structure representations

Methods:
- Masked token reconstruction
- Contrastive learning (InfoNCE)
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
from typing import Dict
import sys

from model_natron import NatronPretrainModel
from losses import PretrainingLoss
from dataset_loader import DatasetManager, TradingSequenceDataset
from feature_engine import FeatureEngine
from label_generator import LabelGenerator


class PretrainingTrainer:
    """Trainer for Phase 1: Pretraining"""
    
    def __init__(self, config: Dict, device: str = 'cuda'):
        self.config = config
        self.device = device
        self.best_loss = float('inf')
        
        # Create checkpoint directory
        os.makedirs(config['pretrain']['checkpoint_dir'], exist_ok=True)
        
    def create_mask(self, x: torch.Tensor, mask_ratio: float = 0.15) -> torch.Tensor:
        """
        Create random mask for tokens
        
        Args:
            x: Input tensor (batch, seq, features)
            mask_ratio: Ratio of tokens to mask
            
        Returns:
            Boolean mask (batch, seq)
        """
        batch_size, seq_len, _ = x.shape
        
        # Number of tokens to mask per sequence
        num_masked = int(seq_len * mask_ratio)
        
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=x.device)
        
        for i in range(batch_size):
            # Random indices to mask
            masked_indices = torch.randperm(seq_len)[:num_masked]
            mask[i, masked_indices] = True
        
        return mask
    
    def apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply mask to input by setting masked tokens to zero
        
        Args:
            x: Input tensor (batch, seq, features)
            mask: Boolean mask (batch, seq)
            
        Returns:
            Masked input
        """
        mask_expanded = mask.unsqueeze(-1).expand_as(x)
        return x.masked_fill(mask_expanded, 0.0)
    
    def train_epoch(
        self,
        model: NatronPretrainModel,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_fn: PretrainingLoss,
        scaler: GradScaler,
        use_amp: bool = True
    ) -> Dict[str, float]:
        """Train one epoch"""
        model.train()
        
        total_loss = 0
        reconstruction_loss_sum = 0
        contrastive_loss_sum = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc="Training")
        
        for sequences, _ in pbar:
            sequences = sequences.to(self.device)
            
            # Create mask
            mask = self.create_mask(sequences, self.config['pretrain']['mask_ratio'])
            
            # Apply mask to input
            masked_sequences = self.apply_mask(sequences, mask)
            
            # Forward pass with mixed precision
            with autocast(enabled=use_amp):
                outputs = model(masked_sequences, mask=mask)
                losses = loss_fn(outputs, sequences, mask=mask)
                loss = losses['total_loss']
            
            # Backward pass
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Accumulate losses
            total_loss += loss.item()
            reconstruction_loss_sum += losses['reconstruction_loss'].item()
            contrastive_loss_sum += losses['contrastive_loss'].item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{losses['reconstruction_loss'].item():.4f}",
                'contr': f"{losses['contrastive_loss'].item():.4f}"
            })
        
        return {
            'total_loss': total_loss / num_batches,
            'reconstruction_loss': reconstruction_loss_sum / num_batches,
            'contrastive_loss': contrastive_loss_sum / num_batches
        }
    
    def validate(
        self,
        model: NatronPretrainModel,
        val_loader: DataLoader,
        loss_fn: PretrainingLoss
    ) -> Dict[str, float]:
        """Validate model"""
        model.eval()
        
        total_loss = 0
        reconstruction_loss_sum = 0
        contrastive_loss_sum = 0
        num_batches = 0
        
        with torch.no_grad():
            for sequences, _ in val_loader:
                sequences = sequences.to(self.device)
                
                # Create mask
                mask = self.create_mask(sequences, self.config['pretrain']['mask_ratio'])
                masked_sequences = self.apply_mask(sequences, mask)
                
                # Forward pass
                outputs = model(masked_sequences, mask=mask)
                losses = loss_fn(outputs, sequences, mask=mask)
                
                total_loss += losses['total_loss'].item()
                reconstruction_loss_sum += losses['reconstruction_loss'].item()
                contrastive_loss_sum += losses['contrastive_loss'].item()
                num_batches += 1
        
        return {
            'total_loss': total_loss / num_batches,
            'reconstruction_loss': reconstruction_loss_sum / num_batches,
            'contrastive_loss': contrastive_loss_sum / num_batches
        }
    
    def train(
        self,
        model: NatronPretrainModel,
        train_loader: DataLoader,
        val_loader: DataLoader
    ):
        """Complete pretraining loop"""
        print("\n" + "="*60)
        print("🚀 Phase 1: Pretraining Started")
        print("="*60)
        
        # Loss function
        loss_fn = PretrainingLoss(
            reconstruction_weight=self.config['pretrain']['loss_weights']['reconstruction'],
            contrastive_weight=self.config['pretrain']['loss_weights']['contrastive'],
            temperature=self.config['pretrain']['contrastive_temp']
        )
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config['pretrain']['learning_rate'],
            weight_decay=self.config['pretrain']['weight_decay']
        )
        
        # Scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Mixed precision scaler
        scaler = GradScaler(enabled=self.config['hardware']['mixed_precision'])
        
        # Training loop
        num_epochs = self.config['pretrain']['epochs']
        
        for epoch in range(num_epochs):
            print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
            
            # Train
            train_metrics = self.train_epoch(
                model, train_loader, optimizer, loss_fn, scaler,
                use_amp=self.config['hardware']['mixed_precision']
            )
            
            # Validate
            val_metrics = self.validate(model, val_loader, loss_fn)
            
            # Print metrics
            print(f"\n📊 Train Loss: {train_metrics['total_loss']:.4f} | "
                  f"Recon: {train_metrics['reconstruction_loss']:.4f} | "
                  f"Contr: {train_metrics['contrastive_loss']:.4f}")
            print(f"📊 Val Loss: {val_metrics['total_loss']:.4f} | "
                  f"Recon: {val_metrics['reconstruction_loss']:.4f} | "
                  f"Contr: {val_metrics['contrastive_loss']:.4f}")
            
            # Learning rate scheduling
            scheduler.step(val_metrics['total_loss'])
            
            # Save best model
            if val_metrics['total_loss'] < self.best_loss:
                self.best_loss = val_metrics['total_loss']
                checkpoint_path = os.path.join(
                    self.config['pretrain']['checkpoint_dir'],
                    'best_pretrain.pt'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': self.best_loss,
                    'config': self.config
                }, checkpoint_path)
                print(f"💾 Saved best model: {checkpoint_path}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(
                    self.config['pretrain']['checkpoint_dir'],
                    f'pretrain_epoch_{epoch+1}.pt'
                )
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_metrics['total_loss'],
                    'config': self.config
                }, checkpoint_path)
        
        print("\n" + "="*60)
        print("✅ Pretraining Complete!")
        print(f"🏆 Best validation loss: {self.best_loss:.4f}")
        print("="*60)


def main():
    """Main pretraining pipeline"""
    
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
        batch_size=config['pretrain']['batch_size'],
        num_workers=config['hardware']['num_workers'],
        pin_memory=config['hardware']['pin_memory']
    )
    
    # Create model
    print("\n🏗️  Building Pretrain Model...")
    model = NatronPretrainModel(
        num_features=config['data']['feature_count'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['num_encoder_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        activation=config['model']['activation'],
        sequence_length=config['data']['sequence_length']
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created with {num_params:,} parameters")
    
    # Train
    trainer = PretrainingTrainer(config, device=device)
    trainer.train(model, train_loader, val_loader)
    
    print("\n🎉 Phase 1 (Pretraining) pipeline complete!")


if __name__ == "__main__":
    main()
