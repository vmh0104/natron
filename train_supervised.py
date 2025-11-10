"""
Phase 2: Supervised Fine-Tuning Script

Train multi-task heads on buy/sell, direction, and regime classification.
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
from label_generator import LabelGenerator
from dataset_loader import SequenceCreator
from model_natron import create_natron_model
from losses import MultiTaskLoss


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    task_losses = {'buy': [], 'sell': [], 'direction': [], 'regime': []}
    
    for batch in tqdm(train_loader, desc='Training'):
        features = batch['features'].to(device)
        
        # Prepare targets
        targets = {
            'buy': batch['buy'].to(device),
            'sell': batch['sell'].to(device),
            'direction': batch['direction'].to(device),
            'regime': batch['regime'].to(device)
        }
        
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(features, mode='supervised')
        
        # Calculate loss
        losses = criterion(predictions, targets)
        
        # Backward
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += losses['total'].item()
        for key in task_losses:
            if key in losses:
                task_losses[key].append(losses[key].item())
    
    avg_losses = {key: np.mean(vals) if vals else 0 for key, vals in task_losses.items()}
    avg_losses['total'] = total_loss / len(train_loader)
    
    return avg_losses


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
):
    """Validate model."""
    model.eval()
    total_loss = 0
    task_losses = {'buy': [], 'sell': [], 'direction': [], 'regime': []}
    
    # Metrics
    buy_correct = 0
    sell_correct = 0
    direction_correct = 0
    regime_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validating'):
            features = batch['features'].to(device)
            
            targets = {
                'buy': batch['buy'].to(device),
                'sell': batch['sell'].to(device),
                'direction': batch['direction'].to(device),
                'regime': batch['regime'].to(device)
            }
            
            # Forward pass
            predictions = model(features, mode='supervised')
            
            # Calculate loss
            losses = criterion(predictions, targets)
            
            total_loss += losses['total'].item()
            for key in task_losses:
                if key in losses:
                    task_losses[key].append(losses[key].item())
            
            # Calculate accuracy
            batch_size = features.size(0)
            total_samples += batch_size
            
            # Buy accuracy
            buy_pred = (predictions['buy'].squeeze() > 0.5).long()
            buy_correct += (buy_pred == targets['buy'].squeeze().long()).sum().item()
            
            # Sell accuracy
            sell_pred = (predictions['sell'].squeeze() > 0.5).long()
            sell_correct += (sell_pred == targets['sell'].squeeze().long()).sum().item()
            
            # Direction accuracy
            direction_pred = predictions['direction'].argmax(dim=-1)
            direction_correct += (direction_pred == targets['direction'].squeeze()).sum().item()
            
            # Regime accuracy
            regime_pred = predictions['regime'].argmax(dim=-1)
            regime_correct += (regime_pred == targets['regime'].squeeze()).sum().item()
    
    avg_losses = {key: np.mean(vals) if vals else 0 for key, vals in task_losses.items()}
    avg_losses['total'] = total_loss / len(val_loader)
    
    accuracies = {
        'buy': buy_correct / total_samples,
        'sell': sell_correct / total_samples,
        'direction': direction_correct / total_samples,
        'regime': regime_correct / total_samples
    }
    
    return avg_losses, accuracies


def main():
    parser = argparse.ArgumentParser(description='Natron Phase 2: Supervised Fine-Tuning')
    parser.add_argument('--data_path', type=str, default='data_export.csv', help='Path to OHLCV data')
    parser.add_argument('--pretrained_path', type=str, default=None, help='Path to pretrained encoder')
    parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder weights')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--sequence_length', type=int, default=96, help='Sequence length')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Save directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    
    # Loss weights
    parser.add_argument('--buy_weight', type=float, default=1.0, help='Buy loss weight')
    parser.add_argument('--sell_weight', type=float, default=1.0, help='Sell loss weight')
    parser.add_argument('--direction_weight', type=float, default=1.0, help='Direction loss weight')
    parser.add_argument('--regime_weight', type=float, default=1.0, help='Regime loss weight')
    
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
    
    # Generate labels
    print('Generating labels...')
    label_generator = LabelGenerator(lookahead_periods=5)
    y_buy, y_sell, y_direction, y_regime = label_generator.generate_labels(df, features_df)
    
    labels_dict = {
        'buy': y_buy,
        'sell': y_sell,
        'direction': y_direction,
        'regime': y_regime
    }
    
    print(f'Label distribution:')
    print(f'  Buy: {y_buy.sum()} ({y_buy.mean()*100:.2f}%)')
    print(f'  Sell: {y_sell.sum()} ({y_sell.mean()*100:.2f}%)')
    print(f'  Direction Up: {(y_direction==1).sum()} ({(y_direction==1).mean()*100:.2f}%)')
    print(f'  Regime distribution: {y_regime.value_counts().to_dict()}')
    
    # Create sequences
    print('Creating sequences...')
    sequence_creator = SequenceCreator(sequence_length=args.sequence_length)
    features_array, labels_dict = sequence_creator.create_sequences(features_df, labels_dict)
    
    # Create dataloaders
    train_loader, val_loader, test_loader = sequence_creator.create_dataloaders(
        features_array,
        labels_dict=labels_dict,
        batch_size=args.batch_size,
        mode='supervised',
        shuffle=True
    )
    
    print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}')
    
    # Create model
    print('Creating model...')
    model = create_natron_model(
        input_dim=len(features_df.columns),
        d_model=256,
        nhead=8,
        num_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        max_seq_len=args.sequence_length,
        freeze_encoder=args.freeze_encoder,
        pretrained_path=args.pretrained_path
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    
    # Loss and optimizer
    criterion = MultiTaskLoss(
        buy_weight=args.buy_weight,
        sell_weight=args.sell_weight,
        direction_weight=args.direction_weight,
        regime_weight=args.regime_weight
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')
    
    # Training loop
    print('\nStarting supervised fine-tuning...')
    for epoch in range(args.epochs):
        print(f'\nEpoch {epoch+1}/{args.epochs}')
        
        # Train
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Train Losses: {train_losses}')
        
        # Validate
        val_losses, val_accuracies = validate(model, val_loader, criterion, device)
        print(f'Val Losses: {val_losses}')
        print(f'Val Accuracies: {val_accuracies}')
        
        scheduler.step(val_losses['total'])
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['total'],
                'val_accuracies': val_accuracies,
                'feature_names': feature_engine.feature_names,
            }, os.path.join(args.save_dir, 'natron_v2.pt'))
            print(f'  -> Saved best model (val_loss={val_losses["total"]:.6f})')
    
    print('\nTraining complete!')
    print(f'Best model saved to: {os.path.join(args.save_dir, "natron_v2.pt")}')


if __name__ == '__main__':
    main()
