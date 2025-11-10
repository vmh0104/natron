"""
Phase 2: Supervised Fine-Tuning
Multi-task learning: Buy/Sell, Direction, Regime prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import os
from tqdm import tqdm
import numpy as np

from model_natron import NatronTransformer
from losses import MultiTaskLoss
from dataset_loader import SequenceCreator
from feature_engine import FeatureEngine
from label_generator import LabelGenerator
import pandas as pd


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train one epoch"""
    model.train()
    total_loss = 0.0
    task_losses = {'buy': 0.0, 'sell': 0.0, 'direction': 0.0, 'regime': 0.0}
    
    for batch in tqdm(dataloader, desc="Training"):
        sequences = batch['sequence'].to(device)
        
        # Prepare targets
        targets = {
            'buy': batch['buy'].to(device),
            'sell': batch['sell'].to(device),
            'direction': batch['direction'].to(device),
            'regime': batch['regime'].to(device)
        }
        
        # Forward pass
        predictions = model(sequences)
        
        # Loss
        losses = criterion(predictions, targets)
        loss = losses['total']
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        for task in task_losses:
            task_losses[task] += losses[task].item()
    
    avg_losses = {task: task_losses[task] / len(dataloader) for task in task_losses}
    avg_losses['total'] = total_loss / len(dataloader)
    return avg_losses


def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0.0
    task_losses = {'buy': 0.0, 'sell': 0.0, 'direction': 0.0, 'regime': 0.0}
    
    # Metrics
    buy_correct = 0
    sell_correct = 0
    direction_correct = 0
    regime_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            sequences = batch['sequence'].to(device)
            
            # Prepare targets
            targets = {
                'buy': batch['buy'].to(device),
                'sell': batch['sell'].to(device),
                'direction': batch['direction'].to(device),
                'regime': batch['regime'].to(device)
            }
            
            # Forward pass
            predictions = model(sequences)
            
            # Loss
            losses = criterion(predictions, targets)
            loss = losses['total']
            
            total_loss += loss.item()
            for task in task_losses:
                task_losses[task] += losses[task].item()
            
            # Accuracy metrics
            batch_size = sequences.size(0)
            total_samples += batch_size
            
            # Buy/Sell (threshold 0.5)
            buy_pred = (predictions['buy'] > 0.5).float()
            buy_correct += (buy_pred.squeeze() == targets['buy'].squeeze()).sum().item()
            
            sell_pred = (predictions['sell'] > 0.5).float()
            sell_correct += (sell_pred.squeeze() == targets['sell'].squeeze()).sum().item()
            
            # Direction
            direction_pred = predictions['direction'].argmax(dim=-1)
            direction_correct += (direction_pred.squeeze() == targets['direction'].squeeze()).sum().item()
            
            # Regime
            regime_pred = predictions['regime'].argmax(dim=-1)
            regime_correct += (regime_pred.squeeze() == targets['regime'].squeeze()).sum().item()
    
    avg_losses = {task: task_losses[task] / len(dataloader) for task in task_losses}
    avg_losses['total'] = total_loss / len(dataloader)
    
    accuracies = {
        'buy': buy_correct / total_samples,
        'sell': sell_correct / total_samples,
        'direction': direction_correct / total_samples,
        'regime': regime_correct / total_samples
    }
    
    return avg_losses, accuracies


def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(config['paths']['model_dir'], exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(config['data']['csv_path'])
    
    # Generate features
    print("Generating features...")
    feature_engine = FeatureEngine()
    features_df = feature_engine.generate_all_features(df)
    
    # Generate labels
    print("Generating labels...")
    label_generator = LabelGenerator(config)
    labels_df = label_generator.generate_labels(df, features_df)
    
    # Create sequences
    print("Creating sequences...")
    sequence_creator = SequenceCreator(config)
    train_dataset, val_dataset, test_dataset = sequence_creator.create_datasets(
        features_df, labels_df
    )
    
    train_loader, val_loader, test_loader = sequence_creator.create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    
    # Save scaler
    sequence_creator.save_scaler(os.path.join(config['paths']['model_dir'], 'scaler.pkl'))
    
    # Initialize model
    print("Initializing model...")
    model = NatronTransformer(
        feature_dim=config['model']['feature_dim'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        activation=config['model']['activation']
    ).to(device)
    
    # Load pretrained encoder if available
    pretrain_path = os.path.join(config['paths']['model_dir'], 'pretrain_best.pt')
    if os.path.exists(pretrain_path):
        print(f"Loading pretrained encoder from {pretrain_path}...")
        checkpoint = torch.load(pretrain_path, map_location=device)
        model.encoder.load_state_dict(checkpoint['model_state_dict'])
        print("Pretrained encoder loaded!")
    else:
        print("No pretrained encoder found, training from scratch.")
    
    # Loss function
    criterion = MultiTaskLoss(config['training']['loss_weights'])
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(config['paths']['logs_dir'], 'supervised'))
    
    # Training loop
    num_epochs = config['training']['num_epochs_supervised']
    best_val_loss = float('inf')
    
    print(f"Starting supervised training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        # Train
        train_losses = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_losses, val_accuracies = validate(model, val_loader, criterion, device)
        
        scheduler.step(val_losses['total'])
        
        # Logging
        writer.add_scalars('Loss/Train', train_losses, epoch)
        writer.add_scalars('Loss/Val', val_losses, epoch)
        writer.add_scalars('Accuracy/Val', val_accuracies, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_losses['total']:.4f} (Buy: {train_losses['buy']:.4f}, "
              f"Sell: {train_losses['sell']:.4f}, Dir: {train_losses['direction']:.4f}, "
              f"Regime: {train_losses['regime']:.4f})")
        print(f"Val Loss: {val_losses['total']:.4f}")
        print(f"Val Accuracies - Buy: {val_accuracies['buy']:.4f}, Sell: {val_accuracies['sell']:.4f}, "
              f"Direction: {val_accuracies['direction']:.4f}, Regime: {val_accuracies['regime']:.4f}")
        
        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['total'],
                'val_accuracies': val_accuracies,
                'config': config
            }
            torch.save(checkpoint, os.path.join(config['paths']['model_dir'], 'natron_v2.pt'))
            print(f"Saved best model (val_loss: {val_losses['total']:.4f})")
    
    # Final test evaluation
    print("\nEvaluating on test set...")
    test_losses, test_accuracies = validate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_losses['total']:.4f}")
    print(f"Test Accuracies - Buy: {test_accuracies['buy']:.4f}, Sell: {test_accuracies['sell']:.4f}, "
          f"Direction: {test_accuracies['direction']:.4f}, Regime: {test_accuracies['regime']:.4f}")
    
    writer.close()
    print("Supervised training complete!")


if __name__ == '__main__':
    main()
