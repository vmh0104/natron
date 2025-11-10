"""
Phase 1: Pretraining Script for Natron Transformer
Unsupervised learning: Masked Modeling + Contrastive Learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import argparse
from pathlib import Path

from dataset_loader import SequenceCreator
from model_natron import NatronPretrainModel
from losses import PretrainLoss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    recon_losses = []
    contrastive_losses = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (X, _) in enumerate(pbar):
        X = X.to(device)
        
        # Forward pass
        outputs = model(X)
        
        # Compute loss
        loss_dict = criterion(outputs, X, outputs['mask'])
        loss = loss_dict['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        recon_losses.append(loss_dict['reconstruction_loss'].item())
        contrastive_losses.append(loss_dict['contrastive_loss'].item())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'recon': f"{loss_dict['reconstruction_loss'].item():.4f}",
            'contrast': f"{loss_dict['contrastive_loss'].item():.4f}"
        })
    
    avg_loss = total_loss / len(dataloader)
    avg_recon = np.mean(recon_losses)
    avg_contrast = np.mean(contrastive_losses)
    
    return avg_loss, avg_recon, avg_contrast


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for X, _ in dataloader:
            X = X.to(device)
            outputs = model(X)
            loss_dict = criterion(outputs, X, outputs['mask'])
            total_loss += loss_dict['total_loss'].item()
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description='Natron Phase 1: Pretraining')
    parser.add_argument('--data', type=str, default='data_export.csv', help='Path to CSV data')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--sequence_length', type=int, default=96, help='Sequence length')
    parser.add_argument('--save_dir', type=str, default='./models', help='Model save directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    print(f"Data shape: {df.shape}")
    
    # Create dataset
    print("Creating dataset...")
    sequence_creator = SequenceCreator(sequence_length=args.sequence_length)
    train_dataset, val_dataset, test_dataset = sequence_creator.create_dataset(df)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Get number of features
    num_features = len(sequence_creator.feature_columns)
    print(f"Number of features: {num_features}")
    
    # Initialize model
    print("Initializing model...")
    model = NatronPretrainModel(
        num_features=num_features,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        sequence_length=args.sequence_length
    ).to(args.device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = PretrainLoss(
        reconstruction_weight=1.0,
        contrastive_weight=0.5,
        temperature=0.07
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    print("\nStarting pretraining...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_recon, train_contrast = train_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch
        )
        
        # Validate
        val_loss = validate(model, val_loader, criterion, args.device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f} (Recon: {train_recon:.4f}, Contrast: {train_contrast:.4f})")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, 'natron_pretrain_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'num_features': num_features,
                'args': vars(args)
            }, save_path)
            print(f"  Saved best model to {save_path}")
        
        # Save checkpoint
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'natron_pretrain_epoch{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
    
    print("\nPretraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
