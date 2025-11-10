"""
Natron Transformer - Phase 1: Pretraining
Unsupervised learning via masked modeling and contrastive learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional
import os

from model_natron import NatronTransformer, MaskedLanguageModel, ContrastiveProjection
from losses import MaskedReconstructionLoss, InfoNCELoss, create_data_augmentation


class PretrainTrainer:
    """
    Trainer for unsupervised pretraining phase
    Combines masked reconstruction and contrastive learning
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
        
        # Pretraining config
        self.pretrain_config = config['training']['pretrain']
        self.epochs = self.pretrain_config['epochs']
        self.lr = self.pretrain_config['learning_rate']
        self.weight_decay = self.pretrain_config['weight_decay']
        self.mask_ratio = self.pretrain_config['mask_ratio']
        self.contrastive_weight = self.pretrain_config['contrastive_weight']
        self.temperature = self.pretrain_config['contrastive_temperature']
        
        # Create pretraining heads
        self.mlm_model = MaskedLanguageModel(model, config['model']['input_dim']).to(device)
        self.contrastive_proj = ContrastiveProjection(model.d_model, 128).to(device)
        
        # Losses
        self.recon_loss_fn = MaskedReconstructionLoss()
        self.contrastive_loss_fn = InfoNCELoss(temperature=self.temperature)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            list(self.mlm_model.parameters()) + list(self.contrastive_proj.parameters()),
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
        self.writer = SummaryWriter(log_dir=os.path.join(config['logging']['tensorboard_dir'], 'pretrain'))
        
        print(f"🔧 Pretraining Trainer initialized")
        print(f"  Epochs: {self.epochs}")
        print(f"  Learning rate: {self.lr}")
        print(f"  Mask ratio: {self.mask_ratio}")
        print(f"  Contrastive weight: {self.contrastive_weight}")
    
    def create_random_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create random mask for sequence
        
        Args:
            x: Input sequence (batch, seq_len, dim)
            
        Returns:
            Boolean mask (batch, seq_len, 1)
        """
        batch_size, seq_len, _ = x.shape
        mask = torch.rand(batch_size, seq_len, 1, device=self.device) < self.mask_ratio
        return mask
    
    def mask_sequence(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply mask to sequence (replace with zeros)
        
        Args:
            x: Input sequence (batch, seq_len, dim)
            mask: Boolean mask (batch, seq_len, 1)
            
        Returns:
            Masked sequence
        """
        return x * (~mask).float()
    
    def train_epoch(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            dataloader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary with average losses
        """
        self.mlm_model.train()
        self.contrastive_proj.train()
        
        total_recon_loss = 0.0
        total_contrastive_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{self.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Get sequences (ignore labels for pretraining)
            if isinstance(batch, (list, tuple)):
                sequences = batch[0].to(self.device)
            else:
                sequences = batch.to(self.device)
            
            # 1. Masked Reconstruction Task
            # Create mask
            mask = self.create_random_mask(sequences)
            masked_sequences = self.mask_sequence(sequences, mask)
            
            # Reconstruct
            reconstructed = self.mlm_model(masked_sequences)
            recon_loss = self.recon_loss_fn(reconstructed, sequences, mask)
            
            # 2. Contrastive Learning Task
            # Create two augmented views
            aug1 = create_data_augmentation(sequences, 'noise')
            aug2 = create_data_augmentation(sequences, 'scale')
            
            # Get representations
            encoded1 = self.model.get_encoder_output(aug1)
            encoded2 = self.model.get_encoder_output(aug2)
            
            # Pool and project
            pooled1 = encoded1.mean(dim=1)  # (batch, d_model)
            pooled2 = encoded2.mean(dim=1)
            
            proj1 = self.contrastive_proj(pooled1)
            proj2 = self.contrastive_proj(pooled2)
            
            # Contrastive loss
            contrastive_loss = self.contrastive_loss_fn(proj1, proj2)
            
            # Combined loss
            loss = recon_loss + self.contrastive_weight * contrastive_loss
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mlm_model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track losses
            total_recon_loss += recon_loss.item()
            total_contrastive_loss += contrastive_loss.item()
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'recon': recon_loss.item(),
                'contr': contrastive_loss.item(),
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Log to TensorBoard
            global_step = epoch * len(dataloader) + batch_idx
            self.writer.add_scalar('Train/Loss', loss.item(), global_step)
            self.writer.add_scalar('Train/ReconLoss', recon_loss.item(), global_step)
            self.writer.add_scalar('Train/ContrastiveLoss', contrastive_loss.item(), global_step)
        
        # Average losses
        avg_losses = {
            'total': total_loss / num_batches,
            'reconstruction': total_recon_loss / num_batches,
            'contrastive': total_contrastive_loss / num_batches
        }
        
        return avg_losses
    
    def train(self, dataloader: DataLoader) -> NatronTransformer:
        """
        Full pretraining loop
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Pretrained model
        """
        print(f"\n🚀 Starting Pretraining...")
        print(f"  Total epochs: {self.epochs}")
        print(f"  Batches per epoch: {len(dataloader)}")
        
        best_loss = float('inf')
        checkpoint_dir = self.config['paths']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(1, self.epochs + 1):
            # Train epoch
            losses = self.train_epoch(dataloader, epoch)
            
            # Log epoch losses
            print(f"\n📊 Epoch {epoch} Summary:")
            print(f"  Total Loss: {losses['total']:.4f}")
            print(f"  Reconstruction Loss: {losses['reconstruction']:.4f}")
            print(f"  Contrastive Loss: {losses['contrastive']:.4f}")
            
            # TensorBoard
            self.writer.add_scalar('Epoch/TotalLoss', losses['total'], epoch)
            self.writer.add_scalar('Epoch/ReconLoss', losses['reconstruction'], epoch)
            self.writer.add_scalar('Epoch/ContrastiveLoss', losses['contrastive'], epoch)
            
            # Learning rate scheduling
            self.scheduler.step(losses['total'])
            
            # Save best model
            if losses['total'] < best_loss:
                best_loss = losses['total']
                checkpoint_path = os.path.join(checkpoint_dir, 'pretrain_best.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': best_loss
                }, checkpoint_path)
                print(f"  💾 Best model saved (loss: {best_loss:.4f})")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                checkpoint_path = os.path.join(checkpoint_dir, f'pretrain_epoch_{epoch}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': losses['total']
                }, checkpoint_path)
                print(f"  💾 Checkpoint saved: epoch {epoch}")
        
        print(f"\n✅ Pretraining complete!")
        print(f"  Best loss: {best_loss:.4f}")
        
        # Load best model
        best_checkpoint = os.path.join(checkpoint_dir, 'pretrain_best.pt')
        checkpoint = torch.load(best_checkpoint)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.writer.close()
        
        return self.model


def run_pretraining(
    model: NatronTransformer,
    dataloader: DataLoader,
    config: Dict,
    device: str = 'cuda'
) -> NatronTransformer:
    """
    Convenience function to run pretraining
    
    Args:
        model: Natron Transformer model
        dataloader: Training data loader
        config: Configuration dictionary
        device: Device to train on
        
    Returns:
        Pretrained model
    """
    trainer = PretrainTrainer(model, config, device)
    pretrained_model = trainer.train(dataloader)
    return pretrained_model


if __name__ == "__main__":
    # Test pretraining
    print("🧪 Testing Pretraining...")
    
    import yaml
    from model_natron import create_model
    from dataset_loader import PretrainDataset
    from torch.utils.data import DataLoader
    
    # Load config
    with open('/workspace/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dummy data
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_samples = 500
    sequences = np.random.randn(n_samples, 96, 100)
    
    dataset = PretrainDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create model
    model = create_model(config, device)
    
    # Run 2 epochs for testing
    config['training']['pretrain']['epochs'] = 2
    trainer = PretrainTrainer(model, config, device)
    
    print("\n🔄 Running test pretraining (2 epochs)...")
    pretrained_model = trainer.train(dataloader)
    
    print("\n✅ Pretraining test successful!")
