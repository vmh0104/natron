"""
Phase 1: Pretraining - Unsupervised Learning
Masked Modeling and/or Contrastive Learning
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

from model_natron import NatronTransformerEncoder, MaskedModelingHead, ContrastiveHead
from losses import MaskedModelingLoss, AugmentedContrastiveLoss
from dataset_loader import SequenceCreator
import pandas as pd
from feature_engine import FeatureEngine


def create_masked_sequences(features: torch.Tensor, mask_prob: float = 0.15):
    """
    Create masked sequences for pretraining.
    
    Args:
        features: (batch_size, seq_len, feature_dim)
        mask_prob: Probability of masking each token
        
    Returns:
        masked_features: Features with masked tokens
        mask: Boolean mask (True = masked)
        original: Original features
    """
    batch_size, seq_len, feature_dim = features.shape
    device = features.device
    
    # Create random mask
    mask = torch.rand(batch_size, seq_len, device=device) < mask_prob
    
    # Create masked features
    masked_features = features.clone()
    
    # Replace masked positions with zeros (or random noise)
    mask_expanded = mask.unsqueeze(-1).expand_as(masked_features)
    masked_features[mask_expanded] = 0.0
    
    return masked_features, mask, features


def train_epoch_masked_modeling(model, dataloader, optimizer, device, mask_prob=0.15):
    """Train one epoch with masked modeling"""
    model.train()
    total_loss = 0.0
    criterion = MaskedModelingLoss()
    
    for batch in tqdm(dataloader, desc="Training"):
        sequences = batch['sequence'].to(device)
        
        # Create masked sequences
        masked_seq, mask, original = create_masked_sequences(sequences, mask_prob)
        
        # Forward pass
        encoded = model(masked_seq)
        
        # Reconstruction
        reconstructed = model.reconstruction_head(encoded)
        
        # Loss
        loss = criterion(reconstructed, original, mask)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def train_epoch_contrastive(model, contrastive_head, dataloader, optimizer, device):
    """Train one epoch with contrastive learning"""
    model.train()
    contrastive_head.train()
    total_loss = 0.0
    criterion = AugmentedContrastiveLoss(temperature=0.07)
    
    for batch in tqdm(dataloader, desc="Training"):
        sequences = batch['sequence'].to(device)
        
        # Contrastive loss with augmentation
        loss = criterion(model, contrastive_head, sequences)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(contrastive_head.parameters()), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


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
    
    # Create sequences (no labels needed for pretraining)
    print("Creating sequences...")
    sequence_creator = SequenceCreator(config)
    
    # Create dummy labels for dataset creation
    dummy_labels = pd.DataFrame({
        'buy': [0] * len(features_df),
        'sell': [0] * len(features_df),
        'direction': [0] * len(features_df),
        'regime': [0] * len(features_df)
    }, index=features_df.index)
    
    train_dataset, val_dataset, test_dataset = sequence_creator.create_datasets(
        features_df, dummy_labels
    )
    
    train_loader, val_loader, _ = sequence_creator.create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    
    # Save scaler
    sequence_creator.save_scaler(os.path.join(config['paths']['model_dir'], 'scaler.pkl'))
    
    # Initialize model
    print("Initializing model...")
    model = NatronTransformerEncoder(
        feature_dim=config['model']['feature_dim'],
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_layers=config['model']['num_layers'],
        dim_feedforward=config['model']['dim_feedforward'],
        dropout=config['model']['dropout'],
        activation=config['model']['activation']
    ).to(device)
    
    # Choose pretraining method
    use_contrastive = config['training']['pretrain'].get('use_contrastive', True)
    
    if use_contrastive:
        # Contrastive learning
        contrastive_head = ContrastiveHead(
            d_model=config['model']['d_model'],
            projection_dim=128
        ).to(device)
        
        # Add reconstruction head for potential masked modeling
        model.reconstruction_head = MaskedModelingHead(
            d_model=config['model']['d_model'],
            feature_dim=config['model']['feature_dim'],
            dropout=config['model']['dropout']
        ).to(device)
        
        # Combined optimizer
        optimizer = optim.AdamW(
            list(model.parameters()) + list(contrastive_head.parameters()) + list(model.reconstruction_head.parameters()),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
    else:
        # Masked modeling only
        model.reconstruction_head = MaskedModelingHead(
            d_model=config['model']['d_model'],
            feature_dim=config['model']['feature_dim'],
            dropout=config['model']['dropout']
        ).to(device)
        
        optimizer = optim.AdamW(
            list(model.parameters()) + list(model.reconstruction_head.parameters()),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        contrastive_head = None
    
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(config['paths']['logs_dir'], 'pretrain'))
    
    # Training loop
    num_epochs = config['training']['num_epochs_pretrain']
    best_val_loss = float('inf')
    
    print(f"Starting pretraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        if use_contrastive:
            train_loss = train_epoch_contrastive(model, contrastive_head, train_loader, optimizer, device)
        else:
            mask_prob = config['training']['pretrain'].get('mask_probability', 0.15)
            train_loss = train_epoch_masked_modeling(model, train_loader, optimizer, device, mask_prob)
        
        # Validation (use contrastive for now)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequence'].to(device)
                if use_contrastive:
                    loss = AugmentedContrastiveLoss()(model, contrastive_head, sequences)
                else:
                    masked_seq, mask, original = create_masked_sequences(sequences, 0.15)
                    encoded = model(masked_seq)
                    reconstructed = model.reconstruction_head(encoded)
                    loss = MaskedModelingLoss()(reconstructed, original, mask)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        scheduler.step(val_loss)
        
        # Logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }
            if contrastive_head is not None:
                checkpoint['contrastive_head_state_dict'] = contrastive_head.state_dict()
            if hasattr(model, 'reconstruction_head'):
                checkpoint['reconstruction_head_state_dict'] = model.reconstruction_head.state_dict()
            
            torch.save(checkpoint, os.path.join(config['paths']['model_dir'], 'pretrain_best.pt'))
            print(f"Saved best model (val_loss: {val_loss:.4f})")
    
    writer.close()
    print("Pretraining complete!")


if __name__ == '__main__':
    main()
