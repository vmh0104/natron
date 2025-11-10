"""
Natron Trainer - Handles Phase 1 (Pretrain), Phase 2 (Supervised), Phase 3 (RL)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, Optional, Tuple
import os
from tqdm import tqdm


class FinancialDataset(Dataset):
    """Dataset for financial sequences"""
    
    def __init__(self, X: np.ndarray, y: Optional[Dict[str, np.ndarray]] = None):
        self.X = torch.FloatTensor(X)
        self.y = y
        if y is not None:
            self.y = {k: torch.LongTensor(v) if k != 'buy' and k != 'sell' 
                     else torch.FloatTensor(v) for k, v in y.items()}
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return {
                'x': self.X[idx],
                'buy': self.y['buy'][idx],
                'sell': self.y['sell'][idx],
                'direction': self.y['direction'][idx],
                'regime': self.y['regime'][idx]
            }
        else:
            return {'x': self.X[idx]}


class NatronTrainer:
    """Main trainer for Natron model"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss functions
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()
    
    def train_phase1_pretrain(
        self,
        train_loader: DataLoader,
        epochs: int = 10,
        mask_prob: float = 0.15,
        contrastive_weight: float = 0.5
    ) -> list:
        """
        Phase 1: Pretraining with masked modeling and contrastive learning
        """
        self.model.train()
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            pbar = tqdm(train_loader, desc=f'Pretrain Epoch {epoch+1}/{epochs}')
            
            for batch in pbar:
                x = batch['x'].to(self.device)
                batch_size, seq_len, num_features = x.shape
                
                # Create random mask
                mask = torch.rand(batch_size, seq_len) < mask_prob
                mask = mask.to(self.device)
                
                # Create augmented version (add noise)
                noise = torch.randn_like(x) * 0.01
                x_aug = x + noise
                x_aug = torch.clamp(x_aug, 0, 1)  # Normalize if needed
                
                self.optimizer.zero_grad()
                
                # Forward pass
                if hasattr(self.model, 'base_model'):
                    # Pretrain wrapper model
                    outputs = self.model(x, x_aug=x_aug, mask=mask)
                else:
                    # Direct model
                    outputs = self.model(x, mask=mask, mode='pretrain')
                
                # Reconstruction loss (masked modeling)
                if 'reconstruction' in outputs:
                    recon = outputs['reconstruction']
                    if mask is not None:
                        masked_x = x[mask]
                        # Handle different reconstruction shapes
                        if recon.dim() == 3:  # [B, L, F]
                            recon = recon[mask]
                        elif recon.dim() == 2:  # Already flattened
                            pass
                        recon_loss = self.mse_loss(recon, masked_x)
                    else:
                        recon_loss = self.mse_loss(recon, x)
                else:
                    recon_loss = torch.tensor(0.0, device=self.device)
                
                # Contrastive loss (InfoNCE)
                if 'projection' in outputs and 'projection_aug' in outputs:
                    proj = outputs['projection']
                    proj_aug = outputs['projection_aug']
                    
                    # Normalize
                    proj = F.normalize(proj, p=2, dim=1)
                    proj_aug = F.normalize(proj_aug, p=2, dim=1)
                    
                    # Positive pairs: (proj[i], proj_aug[i])
                    # Negative pairs: (proj[i], proj_aug[j]) for i != j
                    batch_size_proj = proj.size(0)
                    temperature = 0.1
                    
                    # Compute similarity matrix
                    sim_matrix = torch.matmul(proj, proj_aug.T) / temperature
                    
                    # Positive pairs are on the diagonal
                    labels = torch.arange(batch_size_proj, device=self.device)
                    contrastive_loss = self.ce_loss(sim_matrix, labels)
                else:
                    contrastive_loss = torch.tensor(0.0, device=self.device)
                
                # Total loss
                total_loss = (1 - contrastive_weight) * recon_loss + contrastive_weight * contrastive_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                epoch_loss += total_loss.item()
                pbar.set_postfix({'loss': total_loss.item()})
            
            avg_loss = epoch_loss / len(train_loader)
            losses.append(avg_loss)
            print(f'Epoch {epoch+1} Average Loss: {avg_loss:.4f}')
        
        return losses
    
    def train_phase2_supervised(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        task_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, list]:
        """
        Phase 2: Supervised fine-tuning with multi-task learning
        """
        if task_weights is None:
            task_weights = {
                'buy': 1.0,
                'sell': 1.0,
                'direction': 1.0,
                'regime': 1.0
            }
        
        self.model.train()
        history = {
            'train_loss': [],
            'val_loss': [],
            'train_buy_acc': [],
            'train_sell_acc': [],
            'train_direction_acc': [],
            'train_regime_acc': []
        }
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_metrics = {
                'buy_correct': 0,
                'sell_correct': 0,
                'direction_correct': 0,
                'regime_correct': 0,
                'total': 0
            }
            
            pbar = tqdm(train_loader, desc=f'Supervised Epoch {epoch+1}/{epochs}')
            
            for batch in pbar:
                x = batch['x'].to(self.device)
                buy = batch['buy'].to(self.device)
                sell = batch['sell'].to(self.device)
                direction = batch['direction'].to(self.device)
                regime = batch['regime'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(x, mode='supervised')
                
                # Compute losses
                buy_loss = self.bce_loss(outputs['buy'], buy) * task_weights['buy']
                sell_loss = self.bce_loss(outputs['sell'], sell) * task_weights['sell']
                direction_loss = self.ce_loss(outputs['direction'], direction) * task_weights['direction']
                regime_loss = self.ce_loss(outputs['regime'], regime) * task_weights['regime']
                
                total_loss = buy_loss + sell_loss + direction_loss + regime_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += total_loss.item()
                
                # Metrics
                buy_pred = (outputs['buy'] > 0.5).long()
                sell_pred = (outputs['sell'] > 0.5).long()
                direction_pred = outputs['direction'].argmax(dim=1)
                regime_pred = outputs['regime'].argmax(dim=1)
                
                train_metrics['buy_correct'] += (buy_pred == buy.long()).sum().item()
                train_metrics['sell_correct'] += (sell_pred == sell.long()).sum().item()
                train_metrics['direction_correct'] += (direction_pred == direction).sum().item()
                train_metrics['regime_correct'] += (regime_pred == regime).sum().item()
                train_metrics['total'] += len(buy)
                
                pbar.set_postfix({
                    'loss': total_loss.item(),
                    'buy_acc': train_metrics['buy_correct'] / train_metrics['total'],
                    'dir_acc': train_metrics['direction_correct'] / train_metrics['total']
                })
            
            avg_train_loss = train_loss / len(train_loader)
            history['train_loss'].append(avg_train_loss)
            history['train_buy_acc'].append(train_metrics['buy_correct'] / train_metrics['total'])
            history['train_sell_acc'].append(train_metrics['sell_correct'] / train_metrics['total'])
            history['train_direction_acc'].append(train_metrics['direction_correct'] / train_metrics['total'])
            history['train_regime_acc'].append(train_metrics['regime_correct'] / train_metrics['total'])
            
            # Validation
            if val_loader is not None:
                val_loss, val_metrics = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                
                self.scheduler.step(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    print(f'New best validation loss: {best_val_loss:.4f}')
            else:
                self.scheduler.step(avg_train_loss)
            
            print(f'Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}')
        
        return history
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, Dict]:
        """Validate model"""
        self.model.eval()
        val_loss = 0.0
        metrics = {
            'buy_correct': 0,
            'sell_correct': 0,
            'direction_correct': 0,
            'regime_correct': 0,
            'total': 0
        }
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch['x'].to(self.device)
                buy = batch['buy'].to(self.device)
                sell = batch['sell'].to(self.device)
                direction = batch['direction'].to(self.device)
                regime = batch['regime'].to(self.device)
                
                outputs = self.model(x, mode='supervised')
                
                buy_loss = self.bce_loss(outputs['buy'], buy)
                sell_loss = self.bce_loss(outputs['sell'], sell)
                direction_loss = self.ce_loss(outputs['direction'], direction)
                regime_loss = self.ce_loss(outputs['regime'], regime)
                
                total_loss = buy_loss + sell_loss + direction_loss + regime_loss
                val_loss += total_loss.item()
                
                # Metrics
                buy_pred = (outputs['buy'] > 0.5).long()
                sell_pred = (outputs['sell'] > 0.5).long()
                direction_pred = outputs['direction'].argmax(dim=1)
                regime_pred = outputs['regime'].argmax(dim=1)
                
                metrics['buy_correct'] += (buy_pred == buy.long()).sum().item()
                metrics['sell_correct'] += (sell_pred == sell.long()).sum().item()
                metrics['direction_correct'] += (direction_pred == direction).sum().item()
                metrics['regime_correct'] += (regime_pred == regime).sum().item()
                metrics['total'] += len(buy)
        
        avg_val_loss = val_loss / len(val_loader)
        return avg_val_loss, metrics
    
    def save_model(self, path: str):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        print(f'Model saved to {path}')
    
    def load_model(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'Model loaded from {path}')
