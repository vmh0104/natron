"""
Phase 1: Pretraining with Masked Modeling and Contrastive Learning
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import os
from model import NatronEncoder, MaskedModelingHead


class MaskedDataset(Dataset):
    """Dataset for masked token modeling"""
    
    def __init__(self, X: np.ndarray, mask_prob: float = 0.15):
        self.X = torch.FloatTensor(X)
        self.mask_prob = mask_prob
        self.seq_len = X.shape[1]
        self.feature_dim = X.shape[2]
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sequence = self.X[idx].clone()
        
        # Create random mask
        mask = torch.rand(self.seq_len) < self.mask_prob
        masked_sequence = sequence.clone()
        
        # Mask tokens: replace with zeros (or random noise)
        masked_sequence[mask] = 0.0
        
        return {
            'input': masked_sequence,
            'target': sequence,
            'mask': mask
        }


class ContrastiveDataset(Dataset):
    """Dataset for contrastive learning"""
    
    def __init__(self, X: np.ndarray):
        self.X = torch.FloatTensor(X)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sequence = self.X[idx]
        
        # Create two augmented views (add noise)
        noise1 = torch.randn_like(sequence) * 0.01
        noise2 = torch.randn_like(sequence) * 0.01
        
        view1 = sequence + noise1
        view2 = sequence + noise2
        
        return {
            'view1': view1,
            'view2': view2
        }


class Pretrainer:
    """Pretraining manager"""
    
    def __init__(
        self,
        input_dim: int = 100,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        self.device = device
        self.input_dim = input_dim
        self.d_model = d_model
        
        # Encoder
        self.encoder = NatronEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers
        ).to(device)
        
        # Masked modeling head
        self.masked_head = MaskedModelingHead(d_model, input_dim).to(device)
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model // 2)
        ).to(device)
    
    def train_masked_modeling(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        lr: float = 1e-4,
        save_path: str = 'models/pretrain_masked.pt'
    ):
        """Train with masked token reconstruction"""
        optimizer = optim.AdamW(
            list(self.encoder.parameters()) + list(self.masked_head.parameters()),
            lr=lr,
            weight_decay=1e-5
        )
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.encoder.train()
            self.masked_head.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                input_seq = batch['input'].to(self.device)
                target_seq = batch['target'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                optimizer.zero_grad()
                
                # Encode
                encoded = self.encoder(input_seq)
                
                # Predict masked tokens
                predictions = self.masked_head(encoded)
                
                # Loss only on masked positions
                loss = criterion(predictions[mask], target_seq[mask])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.masked_head.parameters()),
                    max_norm=1.0
                )
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.encoder.eval()
            self.masked_head.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_seq = batch['input'].to(self.device)
                    target_seq = batch['target'].to(self.device)
                    mask = batch['mask'].to(self.device)
                    
                    encoded = self.encoder(input_seq)
                    predictions = self.masked_head(encoded)
                    loss = criterion(predictions[mask], target_seq[mask])
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            print(f'Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'encoder_state_dict': self.encoder.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, save_path)
                print(f'Saved best model to {save_path}')
    
    def train_contrastive(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 30,
        lr: float = 1e-4,
        temperature: float = 0.07,
        save_path: str = 'models/pretrain_contrastive.pt'
    ):
        """Train with contrastive learning (InfoNCE)"""
        optimizer = optim.AdamW(
            list(self.encoder.parameters()) + list(self.projection_head.parameters()),
            lr=lr,
            weight_decay=1e-5
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # Training
            self.encoder.train()
            self.projection_head.train()
            train_loss = 0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                view1 = batch['view1'].to(self.device)
                view2 = batch['view2'].to(self.device)
                
                optimizer.zero_grad()
                
                # Encode both views
                encoded1 = self.encoder(view1)
                encoded2 = self.encoder(view2)
                
                # Pool and project
                pooled1 = encoded1.mean(dim=1)  # [batch_size, d_model]
                pooled2 = encoded2.mean(dim=1)
                
                proj1 = self.projection_head(pooled1)
                proj2 = self.projection_head(pooled2)
                
                # Normalize
                proj1 = F.normalize(proj1, dim=-1)
                proj2 = F.normalize(proj2, dim=-1)
                
                # InfoNCE loss
                batch_size = proj1.size(0)
                labels = torch.arange(batch_size).to(self.device)
                
                # Similarity matrix
                logits = torch.matmul(proj1, proj2.t()) / temperature
                
                # Symmetric loss
                loss = (F.cross_entropy(logits, labels) + 
                       F.cross_entropy(logits.t(), labels)) / 2
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.projection_head.parameters()),
                    max_norm=1.0
                )
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            self.encoder.eval()
            self.projection_head.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    view1 = batch['view1'].to(self.device)
                    view2 = batch['view2'].to(self.device)
                    
                    encoded1 = self.encoder(view1)
                    encoded2 = self.encoder(view2)
                    
                    pooled1 = encoded1.mean(dim=1)
                    pooled2 = encoded2.mean(dim=1)
                    
                    proj1 = self.projection_head(pooled1)
                    proj2 = self.projection_head(pooled2)
                    
                    proj1 = F.normalize(proj1, dim=-1)
                    proj2 = F.normalize(proj2, dim=-1)
                    
                    batch_size = proj1.size(0)
                    labels = torch.arange(batch_size).to(self.device)
                    logits = torch.matmul(proj1, proj2.t()) / temperature
                    loss = (F.cross_entropy(logits, labels) + 
                           F.cross_entropy(logits.t(), labels)) / 2
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            print(f'Epoch {epoch+1}: Train Loss={train_loss:.6f}, Val Loss={val_loss:.6f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save({
                    'encoder_state_dict': self.encoder.state_dict(),
                    'epoch': epoch,
                    'val_loss': val_loss
                }, save_path)
                print(f'Saved best model to {save_path}')
    
    def get_encoder(self):
        """Return the trained encoder"""
        return self.encoder
