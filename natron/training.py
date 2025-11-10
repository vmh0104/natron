"""
Natron Training Scripts - Phase 1 (Pretraining) and Phase 2 (Supervised)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional
import os
from tqdm import tqdm
from model import NatronModel, MaskedModelingHead


class SequenceDataset(Dataset):
    """Dataset for sequence data."""
    
    def __init__(self, X: np.ndarray, y: Optional[Dict[str, np.ndarray]] = None):
        self.X = torch.FloatTensor(X)
        self.y = y
        if y is not None:
            self.y = {k: torch.LongTensor(v) for k, v in y.items()}
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], {k: v[idx] for k, v in self.y.items()}
        return self.X[idx]


class PretrainingTrainer:
    """Phase 1: Pretraining with masked modeling."""
    
    def __init__(self, 
                 model: NatronModel,
                 device: torch.device,
                 mask_prob: float = 0.15):
        self.model = model.to(device)
        self.device = device
        self.mask_prob = mask_prob
        
        # Add reconstruction head
        self.reconstruction_head = MaskedModelingHead(
            d_model=model.encoder.d_model,
            num_features=100,
            dropout=0.1
        ).to(device)
        
        self.criterion = nn.MSELoss()
    
    def create_mask(self, batch_size: int, seq_len: int) -> torch.Tensor:
        """Create random mask for masked modeling."""
        mask = torch.rand(batch_size, seq_len) > self.mask_prob
        return mask.to(self.device)
    
    def mask_input(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mask input features."""
        masked_x = x.clone()
        # Mask by setting to zero
        masked_x[~mask] = 0
        return masked_x
    
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer):
        """Train one epoch."""
        self.model.train()
        self.reconstruction_head.train()
        total_loss = 0
        
        for batch in tqdm(dataloader, desc="Pretraining"):
            x = batch.to(self.device)
            batch_size, seq_len, num_features = x.shape
            
            # Create mask
            mask = self.create_mask(batch_size, seq_len)
            
            # Mask input
            masked_x = self.mask_input(x, mask)
            
            # Forward pass
            encoded = self.model.encoder(masked_x, mask)
            
            # Reconstruct only masked positions
            reconstructed = self.reconstruction_head(encoded)
            
            # Loss: MSE on masked positions only
            loss = self.criterion(reconstructed[~mask], x[~mask])
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def train(self, 
              train_loader: DataLoader,
              num_epochs: int = 10,
              lr: float = 1e-4,
              weight_decay: float = 1e-5):
        """Train pretraining phase."""
        # Combine model and reconstruction head parameters
        params = list(self.model.parameters()) + list(self.reconstruction_head.parameters())
        optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        
        best_loss = float('inf')
        
        for epoch in range(num_epochs):
            avg_loss = self.train_epoch(train_loader, optimizer)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
        
        return self.model


class SupervisedTrainer:
    """Phase 2: Supervised fine-tuning."""
    
    def __init__(self, model: NatronModel, device: torch.device):
        self.model = model.to(device)
        self.device = device
        
        # Loss functions
        self.buy_criterion = nn.BCELoss()
        self.sell_criterion = nn.BCELoss()
        self.direction_criterion = nn.CrossEntropyLoss()
        self.regime_criterion = nn.CrossEntropyLoss()
        
        # Task weights
        self.task_weights = {
            'buy': 1.0,
            'sell': 1.0,
            'direction': 1.0,
            'regime': 1.0
        }
    
    def compute_loss(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss."""
        losses = {}
        
        # Buy loss
        buy_loss = self.buy_criterion(outputs['buy'], targets['buy'].float())
        losses['buy'] = buy_loss
        
        # Sell loss
        sell_loss = self.sell_criterion(outputs['sell'], targets['sell'].float())
        losses['sell'] = sell_loss
        
        # Direction loss
        direction_loss = self.direction_criterion(outputs['direction'], targets['direction'])
        losses['direction'] = direction_loss
        
        # Regime loss
        regime_loss = self.regime_criterion(outputs['regime'], targets['regime'])
        losses['regime'] = regime_loss
        
        # Weighted total loss
        total_loss = sum(self.task_weights[k] * losses[k] for k in losses)
        losses['total'] = total_loss
        
        return losses
    
    def train_epoch(self, dataloader: DataLoader, optimizer: optim.Optimizer, scheduler: Optional[object] = None):
        """Train one epoch."""
        self.model.train()
        total_losses = {'buy': 0, 'sell': 0, 'direction': 0, 'regime': 0, 'total': 0}
        
        for batch in tqdm(dataloader, desc="Training"):
            x, y = batch
            x = x.to(self.device)
            y = {k: v.to(self.device) for k, v in y.items()}
            
            # Forward
            outputs = self.model(x)
            
            # Loss
            losses = self.compute_loss(outputs, y)
            
            # Backward
            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Accumulate losses
            for k in total_losses:
                total_losses[k] += losses[k].item()
        
        # Average losses
        for k in total_losses:
            total_losses[k] /= len(dataloader)
        
        return total_losses
    
    @torch.no_grad()
    def validate(self, dataloader: DataLoader):
        """Validate model."""
        self.model.eval()
        total_losses = {'buy': 0, 'sell': 0, 'direction': 0, 'regime': 0, 'total': 0}
        correct = {'buy': 0, 'sell': 0, 'direction': 0, 'regime': 0}
        total_samples = 0
        
        for batch in tqdm(dataloader, desc="Validating"):
            x, y = batch
            x = x.to(self.device)
            y = {k: v.to(self.device) for k, v in y.items()}
            
            # Forward
            outputs = self.model(x)
            
            # Loss
            losses = self.compute_loss(outputs, y)
            
            # Accumulate losses
            for k in total_losses:
                total_losses[k] += losses[k].item()
            
            # Accuracy
            batch_size = x.size(0)
            total_samples += batch_size
            
            # Buy/Sell (threshold 0.5)
            correct['buy'] += ((outputs['buy'] > 0.5).long() == y['buy']).sum().item()
            correct['sell'] += ((outputs['sell'] > 0.5).long() == y['sell']).sum().item()
            
            # Direction/Regime (argmax)
            correct['direction'] += (outputs['direction'].argmax(dim=-1) == y['direction']).sum().item()
            correct['regime'] += (outputs['regime'].argmax(dim=-1) == y['regime']).sum().item()
        
        # Average losses and compute accuracies
        for k in total_losses:
            total_losses[k] /= len(dataloader)
        
        accuracies = {k: correct[k] / total_samples for k in correct}
        
        return total_losses, accuracies
    
    def train(self,
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: int = 50,
              lr: float = 1e-4,
              weight_decay: float = 1e-5,
              patience: int = 5,
              save_path: str = 'model/natron_v2.pt'):
        """Train supervised phase."""
        optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience, verbose=True
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        for epoch in range(num_epochs):
            # Train
            train_losses = self.train_epoch(train_loader, optimizer, scheduler)
            
            # Validate
            val_losses, accuracies = self.validate(val_loader)
            
            # Learning rate scheduling
            scheduler.step(val_losses['total'])
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"Train Loss: {train_losses['total']:.6f}")
            print(f"Val Loss: {val_losses['total']:.6f}")
            print(f"Accuracies - Buy: {accuracies['buy']:.4f}, Sell: {accuracies['sell']:.4f}, "
                  f"Direction: {accuracies['direction']:.4f}, Regime: {accuracies['regime']:.4f}")
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'val_loss': best_val_loss,
                }, save_path)
                print(f"Saved best model to {save_path}")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience * 2:  # Early stopping
                    print("Early stopping triggered")
                    break
        
        return self.model
