"""
Training Pipeline for Natron Transformer
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from pathlib import Path

from config import Config
from feature_engine import FeatureEngine
from label_generator import LabelGenerator
from sequence_creator import SequenceCreator
from model import NatronTransformer, PretrainingLoss, MultiTaskLoss


class TradingDataset(Dataset):
    """Dataset for trading sequences"""
    
    def __init__(self, X: np.ndarray, y: dict):
        self.X = torch.FloatTensor(X)
        self.y_buy = torch.LongTensor(y['buy'])
        self.y_sell = torch.LongTensor(y['sell'])
        self.y_direction = torch.LongTensor(y['direction'])
        self.y_regime = torch.LongTensor(y['regime'])
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return {
            'features': self.X[idx],
            'buy': self.y_buy[idx],
            'sell': self.y_sell[idx],
            'direction': self.y_direction[idx],
            'regime': self.y_regime[idx]
        }


def load_data(data_path: str) -> pd.DataFrame:
    """Load OHLCV data"""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Ensure required columns
    required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    # Convert time to datetime if needed
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
    
    print(f"Loaded {len(df)} rows")
    return df


def prepare_data(df: pd.DataFrame) -> tuple:
    """Generate features and labels"""
    print("Generating features...")
    feature_engine = FeatureEngine()
    features_df = feature_engine.generate_features(df)
    
    print(f"Generated {len(feature_engine.feature_names)} features")
    print(f"Feature shape: {features_df.shape}")
    
    print("Generating labels...")
    label_generator = LabelGenerator()
    labels_df = label_generator.generate_labels(df, features_df)
    
    print("Label distribution:")
    print(f"  Buy: {labels_df['buy'].sum()} ({labels_df['buy'].mean()*100:.2f}%)")
    print(f"  Sell: {labels_df['sell'].sum()} ({labels_df['sell'].mean()*100:.2f}%)")
    print(f"  Direction Up: {(labels_df['direction']==1).sum()} ({(labels_df['direction']==1).mean()*100:.2f}%)")
    print("  Regime distribution:")
    for i, name in enumerate(label_generator.regime_names):
        count = (labels_df['regime'] == i).sum()
        print(f"    {name}: {count} ({count/len(labels_df)*100:.2f}%)")
    
    print("Creating sequences...")
    sequence_creator = SequenceCreator(sequence_length=Config.SEQUENCE_LENGTH)
    X, y = sequence_creator.create_sequences(features_df, labels_df)
    
    print(f"Created {len(X)} sequences")
    print(f"X shape: {X.shape}")
    
    return X, y, feature_engine, label_generator


def train_pretrain(model: NatronTransformer,
                   train_loader: DataLoader,
                   device: torch.device,
                   num_epochs: int) -> NatronTransformer:
    """Phase 1: Pretraining"""
    print("\n" + "="*50)
    print("PHASE 1: PRETRAINING")
    print("="*50)
    
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    pretrain_loss_fn = PretrainingLoss(mask_ratio=Config.MASK_RATIO)
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_losses = []
        
        pbar = tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            features = batch['features'].to(device)
            
            optimizer.zero_grad()
            loss, loss_dict = pretrain_loss_fn(model, features)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = np.mean(epoch_losses)
        scheduler.step(avg_loss)
        
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), Config.MODEL_DIR / "natron_pretrain.pt")
    
    print(f"Pretraining complete. Best loss: {best_loss:.6f}")
    return model


def train_supervised(model: NatronTransformer,
                    train_loader: DataLoader,
                    val_loader: DataLoader,
                    device: torch.device,
                    num_epochs: int) -> NatronTransformer:
    """Phase 2: Supervised Fine-tuning"""
    print("\n" + "="*50)
    print("PHASE 2: SUPERVISED FINE-TUNING")
    print("="*50)
    
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    loss_fn = MultiTaskLoss(
        weight_buy=Config.WEIGHT_BUY,
        weight_sell=Config.WEIGHT_SELL,
        weight_direction=Config.WEIGHT_DIRECTION,
        weight_regime=Config.WEIGHT_REGIME
    )
    
    best_val_loss = float('inf')
    train_history = []
    val_history = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            features = batch['features'].to(device)
            targets = {
                'buy': batch['buy'].to(device),
                'sell': batch['sell'].to(device),
                'direction': batch['direction'].to(device),
                'regime': batch['regime'].to(device)
            }
            
            optimizer.zero_grad()
            predictions = model(features)
            loss, loss_dict = loss_fn(predictions, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            pbar.set_postfix(loss_dict)
        
        avg_train_loss = np.mean(train_losses)
        train_history.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_losses = []
        val_metrics = {
            'buy_acc': [], 'sell_acc': [], 'direction_acc': [], 'regime_acc': []
        }
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                targets = {
                    'buy': batch['buy'].to(device),
                    'sell': batch['sell'].to(device),
                    'direction': batch['direction'].to(device),
                    'regime': batch['regime'].to(device)
                }
                
                predictions = model(features)
                loss, loss_dict = loss_fn(predictions, targets)
                val_losses.append(loss.item())
                
                # Calculate accuracies
                buy_pred = (predictions['buy'] > 0.5).long()
                sell_pred = (predictions['sell'] > 0.5).long()
                direction_pred = predictions['direction'].argmax(dim=1)
                regime_pred = predictions['regime'].argmax(dim=1)
                
                val_metrics['buy_acc'].append((buy_pred == targets['buy']).float().mean().item())
                val_metrics['sell_acc'].append((sell_pred == targets['sell']).float().mean().item())
                val_metrics['direction_acc'].append((direction_pred == targets['direction']).float().mean().item())
                val_metrics['regime_acc'].append((regime_pred == targets['regime']).float().mean().item())
        
        avg_val_loss = np.mean(val_losses)
        val_history.append(avg_val_loss)
        
        avg_accs = {k: np.mean(v) for k, v in val_metrics.items()}
        
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}:")
        print(f"  Train Loss: {avg_train_loss:.6f}")
        print(f"  Val Loss: {avg_val_loss:.6f}")
        print(f"  Val Accuracies: Buy={avg_accs['buy_acc']:.4f}, Sell={avg_accs['sell_acc']:.4f}, "
              f"Direction={avg_accs['direction_acc']:.4f}, Regime={avg_accs['regime_acc']:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  ✓ Saved best model (val_loss={best_val_loss:.6f})")
    
    print(f"\nSupervised training complete. Best val loss: {best_val_loss:.6f}")
    return model


def main():
    """Main training pipeline"""
    print("="*70)
    print("NATRON TRANSFORMER - MULTI-TASK FINANCIAL TRADING MODEL")
    print("="*70)
    
    # Setup
    Config.ensure_dirs()
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")
    
    if not torch.cuda.is_available() and device.type == 'cuda':
        print("Warning: CUDA not available, using CPU")
        device = torch.device('cpu')
    
    # Load and prepare data
    df = load_data(Config.DATA_PATH)
    X, y, feature_engine, label_generator = prepare_data(df)
    
    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    print(f"\nTrain samples: {len(X_train)}")
    print(f"Val samples: {len(X_val)}")
    
    # Create datasets
    train_dataset = TradingDataset(X_train, y_train)
    val_dataset = TradingDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2
    )
    
    # Initialize model
    model = NatronTransformer(
        num_features=Config.NUM_FEATURES,
        d_model=Config.D_MODEL,
        n_heads=Config.N_HEADS,
        n_layers=Config.N_LAYERS,
        d_ff=Config.D_FF,
        dropout=Config.DROPOUT,
        sequence_length=Config.SEQUENCE_LENGTH
    ).to(device)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Phase 1: Pretraining
    model = train_pretrain(model, train_loader, device, Config.NUM_EPOCHS_PRETRAIN)
    
    # Phase 2: Supervised Fine-tuning
    model = train_supervised(model, train_loader, val_loader, device, Config.NUM_EPOCHS_SUPERVISED)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print(f"Model saved to: {Config.MODEL_PATH}")
    print("="*70)


if __name__ == "__main__":
    main()
