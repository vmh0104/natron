"""
Phase 1: Pretraining Script - Masked Modeling and Contrastive Learning

Train the Natron encoder to learn latent market representations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import argparse
from tqdm import tqdm
import json

from feature_engine import FeatureEngine
from dataset_loader import SequenceCreator, NatronDataset
from model_natron import create_natron_model
from losses import MaskedModelingLoss, ContrastiveLoss


def train_masked_modeling(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-4,
    save_dir: str = './checkpoints'
):
    """Train using masked token reconstruction."""
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = MaskedModelingLoss()
    
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch in pbar:
            features = batch['features'].to(device)
            masked_features = batch['masked_features'].to(device)
            mask = batch['mask'].to(device)
            original_features = batch['original_features'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(masked_features, mode='pretrain')
            reconstructed = outputs['reconstructed']
            
            # Loss
            loss = criterion(reconstructed, original_features, mask)
            
            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                features = batch['features'].to(device)
                masked_features = batch['masked_features'].to(device)
                mask = batch['mask'].to(device)
                original_features = batch['original_features'].to(device)
                
                outputs = model(masked_features, mode='pretrain')
                reconstructed = outputs['reconstructed']
                
                loss = criterion(reconstructed, original_features, mask)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': model.encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, os.path.join(save_dir, 'pretrain_best.pt'))
            print(f'  -> Saved best model (val_loss={avg_val_loss:.6f})')
    
    return model


def train_contrastive(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 50,
    lr: float = 1e-4,
    save_dir: str = './checkpoints'
):
    """Train using contrastive learning."""
    from model_natron import ContrastiveEncoder
    
    contrastive_model = ContrastiveEncoder(model.encoder, projection_dim=128).to(device)
    contrastive_model.train()
    
    optimizer = optim.AdamW(contrastive_model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = ContrastiveLoss(temperature=0.07)
    
    os.makedirs(save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    # Data augmentation: time masking and feature noise
    def augment_sequence(seq):
        """Simple augmentation: add noise."""
        noise = torch.randn_like(seq) * 0.01
        return seq + noise
    
    for epoch in range(epochs):
        # Training
        contrastive_model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for batch in pbar:
            features = batch['features'].to(device)
            
            # Create two augmented views
            z1 = augment_sequence(features)
            z2 = augment_sequence(features)
            
            # Get projections
            proj1 = contrastive_model(z1)
            proj2 = contrastive_model(z2)
            
            # Loss
            loss = criterion(proj1, proj2)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(contrastive_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        contrastive_model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]'):
                features = batch['features'].to(device)
                
                z1 = augment_sequence(features)
                z2 = augment_sequence(features)
                
                proj1 = contrastive_model(z1)
                proj2 = contrastive_model(z2)
                
                loss = criterion(proj1, proj2)
                val_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_losses)
        scheduler.step(avg_val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': contrastive_model.encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, os.path.join(save_dir, 'contrastive_best.pt'))
            print(f'  -> Saved best model (val_loss={avg_val_loss:.6f})')
    
    return contrastive_model.encoder


def main():
    parser = argparse.ArgumentParser(description='Natron Phase 1: Pretraining')
    parser.add_argument('--data_path', type=str, default='data_export.csv', help='Path to OHLCV data')
    parser.add_argument('--method', type=str, default='masked', choices=['masked', 'contrastive'], help='Pretraining method')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--sequence_length', type=int, default=96, help='Sequence length')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    print(f'Using device: {device}')
    
    # Load data
    print('Loading data...')
    df = pd.read_csv(args.data_path)
    print(f'Loaded {len(df)} rows')
    
    # Generate features
    print('Generating features...')
    feature_engine = FeatureEngine()
    features_df = feature_engine.fit_transform(df)
    print(f'Generated {len(features_df.columns)} features')
    
    # Create sequences
    print('Creating sequences...')
    sequence_creator = SequenceCreator(sequence_length=args.sequence_length)
    features_array, _ = sequence_creator.create_sequences(features_df, labels=None)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = sequence_creator.create_dataloaders(
        features_array,
        labels_dict=None,
        batch_size=args.batch_size,
        mode='pretrain',
        shuffle=True
    )
    
    print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')
    
    # Create model
    print('Creating model...')
    model = create_natron_model(
        input_dim=len(features_df.columns),
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=args.sequence_length
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    
    # Train
    print(f'\nStarting {args.method} pretraining...')
    if args.method == 'masked':
        trained_encoder = train_masked_modeling(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr, save_dir=args.save_dir
        )
    else:
        trained_encoder = train_contrastive(
            model, train_loader, val_loader, device,
            epochs=args.epochs, lr=args.lr, save_dir=args.save_dir
        )
    
    print('\nPretraining complete!')


if __name__ == '__main__':
    main()
