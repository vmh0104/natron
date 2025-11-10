"""
Phase 2: Supervised Fine-Tuning Script for Natron Transformer
Multi-task learning: Buy/Sell, Direction, Regime prediction
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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from dataset_loader import SequenceCreator
from model_natron import NatronTransformer
from losses import MultiTaskLoss


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    loss_components = {'buy': [], 'sell': [], 'direction': [], 'regime': []}
    
    all_buy_preds = []
    all_buy_targets = []
    all_sell_preds = []
    all_sell_targets = []
    all_direction_preds = []
    all_direction_targets = []
    all_regime_preds = []
    all_regime_targets = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (X, labels) in enumerate(pbar):
        X = X.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        
        # Forward pass
        predictions = model(X)
        
        # Compute loss
        loss_dict = criterion(predictions, labels)
        loss = loss_dict['total_loss']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track losses
        total_loss += loss.item()
        for key in loss_components:
            if f'{key}_loss' in loss_dict:
                loss_components[key].append(loss_dict[f'{key}_loss'].item())
        
        # Collect predictions for metrics
        all_buy_preds.extend((predictions['buy'].squeeze() > 0.5).cpu().numpy())
        all_buy_targets.extend(labels['buy'].squeeze().cpu().numpy())
        all_sell_preds.extend((predictions['sell'].squeeze() > 0.5).cpu().numpy())
        all_sell_targets.extend(labels['sell'].squeeze().cpu().numpy())
        all_direction_preds.extend(predictions['direction'].argmax(dim=-1).cpu().numpy())
        all_direction_targets.extend(labels['direction'].squeeze().cpu().numpy())
        all_regime_preds.extend(predictions['regime'].argmax(dim=-1).cpu().numpy())
        all_regime_targets.extend(labels['regime'].squeeze().cpu().numpy())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'buy': f"{loss_dict.get('buy_loss', 0):.4f}",
            'regime': f"{loss_dict.get('regime_loss', 0):.4f}"
        })
    
    # Compute metrics
    buy_acc = accuracy_score(all_buy_targets, all_buy_preds)
    sell_acc = accuracy_score(all_sell_targets, all_sell_preds)
    direction_acc = accuracy_score(all_direction_targets, all_direction_preds)
    regime_acc = accuracy_score(all_regime_targets, all_regime_preds)
    
    avg_loss = total_loss / len(dataloader)
    avg_losses = {k: np.mean(v) if v else 0.0 for k, v in loss_components.items()}
    
    metrics = {
        'buy_acc': buy_acc,
        'sell_acc': sell_acc,
        'direction_acc': direction_acc,
        'regime_acc': regime_acc
    }
    
    return avg_loss, avg_losses, metrics


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    
    all_buy_preds = []
    all_buy_targets = []
    all_sell_preds = []
    all_sell_targets = []
    all_direction_preds = []
    all_direction_targets = []
    all_regime_preds = []
    all_regime_targets = []
    
    with torch.no_grad():
        for X, labels in dataloader:
            X = X.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            predictions = model(X)
            loss_dict = criterion(predictions, labels)
            total_loss += loss_dict['total_loss'].item()
            
            # Collect predictions
            all_buy_preds.extend((predictions['buy'].squeeze() > 0.5).cpu().numpy())
            all_buy_targets.extend(labels['buy'].squeeze().cpu().numpy())
            all_sell_preds.extend((predictions['sell'].squeeze() > 0.5).cpu().numpy())
            all_sell_targets.extend(labels['sell'].squeeze().cpu().numpy())
            all_direction_preds.extend(predictions['direction'].argmax(dim=-1).cpu().numpy())
            all_direction_targets.extend(labels['direction'].squeeze().cpu().numpy())
            all_regime_preds.extend(predictions['regime'].argmax(dim=-1).cpu().numpy())
            all_regime_targets.extend(labels['regime'].squeeze().cpu().numpy())
    
    # Compute metrics
    buy_acc = accuracy_score(all_buy_targets, all_buy_preds)
    sell_acc = accuracy_score(all_sell_targets, all_sell_preds)
    direction_acc = accuracy_score(all_direction_targets, all_direction_preds)
    regime_acc = accuracy_score(all_regime_targets, all_regime_preds)
    
    avg_loss = total_loss / len(dataloader)
    metrics = {
        'buy_acc': buy_acc,
        'sell_acc': sell_acc,
        'direction_acc': direction_acc,
        'regime_acc': regime_acc
    }
    
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description='Natron Phase 2: Supervised Fine-Tuning')
    parser.add_argument('--data', type=str, default='data_export.csv', help='Path to CSV data')
    parser.add_argument('--pretrain_model', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--d_model', type=int, default=256, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=1024, help='Feedforward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--sequence_length', type=int, default=96, help='Sequence length')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder during fine-tuning')
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
    train_loader, val_loader, test_loader = sequence_creator.create_dataloaders(
        train_dataset, val_dataset, test_dataset,
        batch_size=args.batch_size
    )
    
    # Get number of features
    num_features = len(sequence_creator.feature_columns)
    print(f"Number of features: {num_features}")
    
    # Initialize model
    print("Initializing model...")
    model = NatronTransformer(
        num_features=num_features,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        sequence_length=args.sequence_length,
        freeze_encoder=args.freeze_encoder
    ).to(args.device)
    
    # Load pretrained weights if provided
    if args.pretrain_model and os.path.exists(args.pretrain_model):
        print(f"Loading pretrained model from {args.pretrain_model}...")
        checkpoint = torch.load(args.pretrain_model, map_location=args.device)
        
        # Transfer encoder weights
        pretrain_model = NatronPretrainModel(
            num_features=num_features,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            sequence_length=args.sequence_length
        )
        pretrain_model.load_state_dict(checkpoint['model_state_dict'])
        
        # Copy encoder weights
        model.input_projection.load_state_dict(pretrain_model.input_projection.state_dict())
        model.pos_encoder.load_state_dict(pretrain_model.pos_encoder.state_dict())
        for i, layer in enumerate(model.transformer_encoder):
            layer.load_state_dict(pretrain_model.transformer_encoder[i].state_dict())
        
        print("Pretrained encoder weights loaded!")
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = MultiTaskLoss(
        buy_weight=1.0,
        sell_weight=1.0,
        direction_weight=1.0,
        regime_weight=1.0
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    print("\nStarting supervised fine-tuning...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_losses, train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, args.device, epoch
        )
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, args.device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"    Buy Acc: {train_metrics['buy_acc']:.3f}, Sell Acc: {train_metrics['sell_acc']:.3f}")
        print(f"    Direction Acc: {train_metrics['direction_acc']:.3f}, Regime Acc: {train_metrics['regime_acc']:.3f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"    Buy Acc: {val_metrics['buy_acc']:.3f}, Sell Acc: {val_metrics['sell_acc']:.3f}")
        print(f"    Direction Acc: {val_metrics['direction_acc']:.3f}, Regime Acc: {val_metrics['regime_acc']:.3f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.save_dir, 'natron_v2.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
                'num_features': num_features,
                'feature_columns': sequence_creator.feature_columns,
                'args': vars(args)
            }, save_path)
            print(f"  Saved best model to {save_path}")
        
        # Save checkpoint
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'natron_epoch{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
    
    # Test evaluation
    print("\nEvaluating on test set...")
    test_loss, test_metrics = validate(model, test_loader, criterion, args.device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Metrics: {test_metrics}")
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    main()
