"""
Phase 2: Supervised Fine-Tuning for Multi-Task Learning
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
from model import NatronModel
from sklearn.metrics import accuracy_score, classification_report


class TradingDataset(Dataset):
    """Dataset for supervised trading tasks"""
    
    def __init__(self, X: np.ndarray, y: dict):
        self.X = torch.FloatTensor(X)
        self.y = {k: torch.tensor(v, dtype=torch.long if 'regime' in k or 'direction' in k else torch.float) 
                  for k, v in y.items()}
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'input': self.X[idx],
            'buy': self.y.get('buy', torch.tensor(0.0))[idx],
            'sell': self.y.get('sell', torch.tensor(0.0))[idx],
            'direction': self.y.get('direction', torch.tensor(0))[idx],
            'regime': self.y.get('regime', torch.tensor(0))[idx]
        }


class MultiTaskLoss(nn.Module):
    """Weighted multi-task loss"""
    
    def __init__(
        self,
        buy_weight: float = 1.0,
        sell_weight: float = 1.0,
        direction_weight: float = 1.0,
        regime_weight: float = 1.0
    ):
        super().__init__()
        self.buy_weight = buy_weight
        self.sell_weight = sell_weight
        self.direction_weight = direction_weight
        self.regime_weight = regime_weight
        
        self.bce_loss = nn.BCELoss()
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, predictions: dict, targets: dict) -> dict:
        losses = {}
        total_loss = 0
        
        # Buy/Sell losses (binary classification)
        if 'buy' in predictions and 'buy' in targets:
            buy_loss = self.bce_loss(predictions['buy'], targets['buy'].float())
            losses['buy'] = buy_loss
            total_loss += self.buy_weight * buy_loss
        
        if 'sell' in predictions and 'sell' in targets:
            sell_loss = self.bce_loss(predictions['sell'], targets['sell'].float())
            losses['sell'] = sell_loss
            total_loss += self.sell_weight * sell_loss
        
        # Direction loss (binary classification)
        if 'direction' in predictions and 'direction' in targets:
            direction_loss = self.ce_loss(predictions['direction'], targets['direction'])
            losses['direction'] = direction_loss
            total_loss += self.direction_weight * direction_loss
        
        # Regime loss (6-class classification)
        if 'regime' in predictions and 'regime' in targets:
            regime_loss = self.ce_loss(predictions['regime'], targets['regime'])
            losses['regime'] = regime_loss
            total_loss += self.regime_weight * regime_loss
        
        losses['total'] = total_loss
        return losses


class SupervisedTrainer:
    """Supervised fine-tuning trainer"""
    
    def __init__(
        self,
        input_dim: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        pretrained_encoder_path: str = None
    ):
        self.device = device
        
        # Initialize model
        self.model = NatronModel(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        ).to(device)
        
        # Load pretrained encoder if provided
        if pretrained_encoder_path and os.path.exists(pretrained_encoder_path):
            print(f"Loading pretrained encoder from {pretrained_encoder_path}")
            checkpoint = torch.load(pretrained_encoder_path, map_location=device)
            self.model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            print("Pretrained encoder loaded successfully")
        
        # Loss function
        self.criterion = MultiTaskLoss()
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        save_path: str = 'models/natron_v2.pt',
        freeze_encoder: bool = False
    ):
        """Train the multi-task model"""
        
        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            print("Encoder frozen, only training heads")
        
        # Optimizer
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_losses = {'total': 0, 'buy': 0, 'sell': 0, 'direction': 0, 'regime': 0}
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                inputs = batch['input'].to(self.device)
                targets = {
                    'buy': batch['buy'].to(self.device),
                    'sell': batch['sell'].to(self.device),
                    'direction': batch['direction'].to(self.device),
                    'regime': batch['regime'].to(self.device)
                }
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = self.model(inputs)
                
                # Calculate loss
                losses = self.criterion(predictions, targets)
                
                # Backward pass
                losses['total'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # Accumulate losses
                for key in train_losses:
                    if key in losses:
                        train_losses[key] += losses[key].item()
            
            # Average training losses
            for key in train_losses:
                train_losses[key] /= len(train_loader)
            
            # Validation
            val_losses, val_metrics = self.validate(val_loader)
            
            # Update learning rate
            scheduler.step(val_losses['total'])
            
            # Print metrics
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_losses['total']:.6f} "
                  f"(Buy: {train_losses['buy']:.4f}, Sell: {train_losses['sell']:.4f}, "
                  f"Dir: {train_losses['direction']:.4f}, Regime: {train_losses['regime']:.4f})")
            print(f"Val Loss: {val_losses['total']:.6f} "
                  f"(Buy: {val_losses['buy']:.4f}, Sell: {val_losses['sell']:.4f}, "
                  f"Dir: {val_losses['direction']:.4f}, Regime: {val_losses['regime']:.4f})")
            print(f"Val Metrics: {val_metrics}")
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_losses['total'],
                    'val_metrics': val_metrics
                }, save_path)
                print(f"✓ Saved best model to {save_path}")
    
    def validate(self, val_loader: DataLoader) -> tuple:
        """Validate the model"""
        self.model.eval()
        losses = {'total': 0, 'buy': 0, 'sell': 0, 'direction': 0, 'regime': 0}
        
        all_predictions = {'buy': [], 'sell': [], 'direction': [], 'regime': []}
        all_targets = {'buy': [], 'sell': [], 'direction': [], 'regime': []}
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['input'].to(self.device)
                targets = {
                    'buy': batch['buy'].to(self.device),
                    'sell': batch['sell'].to(self.device),
                    'direction': batch['direction'].to(self.device),
                    'regime': batch['regime'].to(self.device)
                }
                
                predictions = self.model(inputs)
                loss_dict = self.criterion(predictions, targets)
                
                for key in losses:
                    if key in loss_dict:
                        losses[key] += loss_dict[key].item()
                
                # Collect predictions and targets for metrics
                all_predictions['buy'].extend(
                    (predictions['buy'].cpu().numpy() > 0.5).astype(int)
                )
                all_predictions['sell'].extend(
                    (predictions['sell'].cpu().numpy() > 0.5).astype(int)
                )
                all_predictions['direction'].extend(
                    predictions['direction'].cpu().numpy().argmax(axis=1)
                )
                all_predictions['regime'].extend(
                    predictions['regime'].cpu().numpy().argmax(axis=1)
                )
                
                all_targets['buy'].extend(targets['buy'].cpu().numpy())
                all_targets['sell'].extend(targets['sell'].cpu().numpy())
                all_targets['direction'].extend(targets['direction'].cpu().numpy())
                all_targets['regime'].extend(targets['regime'].cpu().numpy())
        
        # Average losses
        for key in losses:
            losses[key] /= len(val_loader)
        
        # Calculate metrics
        metrics = {
            'buy_acc': accuracy_score(all_targets['buy'], all_predictions['buy']),
            'sell_acc': accuracy_score(all_targets['sell'], all_predictions['sell']),
            'direction_acc': accuracy_score(all_targets['direction'], all_predictions['direction']),
            'regime_acc': accuracy_score(all_targets['regime'], all_predictions['regime'])
        }
        
        return losses, metrics
