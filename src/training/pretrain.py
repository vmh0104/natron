"""
Natron Phase 1: Unsupervised Pretraining
- Masked Modeling
- Contrastive Learning
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
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.natron_transformer import NatronPretrainModel
from src.models.losses import MaskedModelingLoss, ContrastiveLoss, HybridPretrainLoss
from src.data.dataset_loader import NatronDataModule


class NatronPretrainer:
    """
    Pretrainer for Natron Transformer.
    Implements unsupervised pretraining with masked modeling and contrastive learning.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.pretrain_config = self.config['pretrain']
        self.model_config = self.config['model']
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {self.device}")
        
        # Initialize model
        self.model = None
        self.optimizer = None
        self.scheduler = None
        
        # Loss functions
        self.masked_loss = MaskedModelingLoss()
        self.contrastive_loss = ContrastiveLoss(
            temperature=self.pretrain_config.get('contrastive_temperature', 0.07)
        )
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=self.config['logging']['tensorboard_dir'] + '/pretrain')
        
        # Metrics
        self.best_loss = float('inf')
        self.epoch = 0
    
    def create_model(self, num_features: int):
        """Create pretraining model"""
        self.model = NatronPretrainModel(
            num_features=num_features,
            d_model=self.model_config['d_model'],
            nhead=self.model_config['nhead'],
            num_encoder_layers=self.model_config['num_encoder_layers'],
            dim_feedforward=self.model_config['dim_feedforward'],
            dropout=self.model_config['dropout'],
            activation=self.model_config['activation']
        ).to(self.device)
        
        print(f"✅ Pretrain model created")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def create_optimizer(self):
        """Create optimizer and scheduler"""
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.pretrain_config['learning_rate'],
            weight_decay=self.pretrain_config['weight_decay']
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        print(f"✅ Optimizer created: AdamW (lr={self.pretrain_config['learning_rate']})")
    
    def mask_sequence(self, x: torch.Tensor, mask_ratio: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Randomly mask features in sequence.
        
        Args:
            x: Input tensor (batch, seq, features)
            mask_ratio: Ratio of features to mask
            
        Returns:
            Masked input and mask tensor
        """
        batch_size, seq_len, num_features = x.shape
        
        # Create random mask
        mask = torch.rand(batch_size, seq_len, num_features, device=x.device) < mask_ratio
        
        # Apply mask (set masked positions to 0)
        masked_x = x.clone()
        masked_x[mask] = 0
        
        return masked_x, mask
    
    def train_epoch_masked(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train one epoch with masked modeling.
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Pretrain Epoch {self.epoch} (Masked)")
        
        for batch in pbar:
            # Get sequences
            sequences = batch['view1'].to(self.device)  # Use view1 as base
            
            # Create masked version
            masked_sequences, mask = self.mask_sequence(
                sequences,
                mask_ratio=self.pretrain_config['mask_ratio']
            )
            
            # Forward pass
            reconstructed = self.model.forward_masked(masked_sequences, None)
            
            # Calculate loss (only on masked positions)
            loss = self.masked_loss(reconstructed, sequences, mask)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return {'masked_loss': avg_loss}
    
    def train_epoch_contrastive(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train one epoch with contrastive learning.
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Pretrain Epoch {self.epoch} (Contrastive)")
        
        for batch in pbar:
            # Get two augmented views
            view1 = batch['view1'].to(self.device)
            view2 = batch['view2'].to(self.device)
            
            # Forward pass for both views
            z1 = self.model.forward_contrastive(view1)
            z2 = self.model.forward_contrastive(view2)
            
            # Calculate contrastive loss
            loss = self.contrastive_loss(z1, z2)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return {'contrastive_loss': avg_loss}
    
    def train_epoch_hybrid(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        Train one epoch with both masked modeling and contrastive learning.
        """
        self.model.train()
        total_loss = 0
        total_masked = 0
        total_contrastive = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Pretrain Epoch {self.epoch} (Hybrid)")
        
        for batch in pbar:
            # Get two augmented views
            view1 = batch['view1'].to(self.device)
            view2 = batch['view2'].to(self.device)
            
            # Masked modeling on view1
            masked_view1, mask = self.mask_sequence(view1, self.pretrain_config['mask_ratio'])
            reconstructed = self.model.forward_masked(masked_view1, None)
            masked_loss = self.masked_loss(reconstructed, view1, mask)
            
            # Contrastive learning between views
            z1 = self.model.forward_contrastive(view1)
            z2 = self.model.forward_contrastive(view2)
            contrastive_loss = self.contrastive_loss(z1, z2)
            
            # Combined loss
            loss = 0.5 * masked_loss + 0.5 * contrastive_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_masked += masked_loss.item()
            total_contrastive += contrastive_loss.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': loss.item(),
                'masked': masked_loss.item(),
                'contra': contrastive_loss.item()
            })
        
        return {
            'total_loss': total_loss / num_batches,
            'masked_loss': total_masked / num_batches,
            'contrastive_loss': total_contrastive / num_batches
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation"):
                view1 = batch['view1'].to(self.device)
                view2 = batch['view2'].to(self.device)
                
                # Masked modeling
                masked_view1, mask = self.mask_sequence(view1, self.pretrain_config['mask_ratio'])
                reconstructed = self.model.forward_masked(masked_view1, None)
                masked_loss = self.masked_loss(reconstructed, view1, mask)
                
                # Contrastive
                z1 = self.model.forward_contrastive(view1)
                z2 = self.model.forward_contrastive(view2)
                contrastive_loss = self.contrastive_loss(z1, z2)
                
                loss = 0.5 * masked_loss + 0.5 * contrastive_loss
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}
    
    def save_checkpoint(self, path: str, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
            print(f"💾 Saved best model to {best_path}")
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """
        Full pretraining loop.
        """
        print(f"\n🚀 Starting Pretraining...")
        print(f"   Epochs: {self.pretrain_config['epochs']}")
        print(f"   Batch size: {self.pretrain_config['batch_size']}")
        print(f"   Use masked: {self.pretrain_config['use_masked_modeling']}")
        print(f"   Use contrastive: {self.pretrain_config['use_contrastive']}")
        
        for epoch in range(self.pretrain_config['epochs']):
            self.epoch = epoch + 1
            
            # Train
            if self.pretrain_config['use_masked_modeling'] and self.pretrain_config['use_contrastive']:
                train_metrics = self.train_epoch_hybrid(train_loader)
            elif self.pretrain_config['use_masked_modeling']:
                train_metrics = self.train_epoch_masked(train_loader)
            else:
                train_metrics = self.train_epoch_contrastive(train_loader)
            
            # Validate
            val_metrics = self.validate(val_loader)
            
            # Log metrics
            for key, value in train_metrics.items():
                self.writer.add_scalar(f'train/{key}', value, self.epoch)
            
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, self.epoch)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['val_loss'])
            
            # Save checkpoint
            if val_metrics['val_loss'] < self.best_loss:
                self.best_loss = val_metrics['val_loss']
                self.save_checkpoint(self.pretrain_config['checkpoint_path'], is_best=True)
            
            # Regular checkpoint every 10 epochs
            if self.epoch % 10 == 0:
                self.save_checkpoint(self.pretrain_config['checkpoint_path'].replace('.pt', f'_epoch{self.epoch}.pt'))
            
            print(f"\n📊 Epoch {self.epoch} Summary:")
            print(f"   Train: {train_metrics}")
            print(f"   Val: {val_metrics}")
            print(f"   Best loss: {self.best_loss:.4f}")
        
        print(f"\n✅ Pretraining completed!")
        self.writer.close()


def main():
    """Main pretraining function"""
    print("=" * 80)
    print("🧠 NATRON TRANSFORMER - PHASE 1: UNSUPERVISED PRETRAINING")
    print("=" * 80)
    
    # Load data
    data_module = NatronDataModule(config_path="config/config.yaml")
    pipeline_data = data_module.prepare_full_pipeline(mode='contrastive')
    
    # Create pretrainer
    pretrainer = NatronPretrainer(config_path="config/config.yaml")
    pretrainer.create_model(num_features=pipeline_data['num_features'])
    pretrainer.create_optimizer()
    
    # Train
    pretrainer.train(
        train_loader=pipeline_data['train_loader'],
        val_loader=pipeline_data['val_loader']
    )
    
    print("\n✅ Pretrained model saved!")
    print(f"   Path: {pretrainer.pretrain_config['checkpoint_path']}")


if __name__ == "__main__":
    main()
