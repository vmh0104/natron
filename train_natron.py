"""
Natron Training Pipeline
Phase 1: Pretraining (Unsupervised)
Phase 2: Supervised Fine-Tuning
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
from pathlib import Path

from feature_engine import FeatureEngine
from label_generator import LabelGenerator
from dataset_loader import SequenceCreator, create_data_loaders
from model_natron import NatronTransformer, NatronPretrainModel
from losses import MultiTaskLoss, PretrainLoss


class NatronTrainer:
    """Main training class for Natron Transformer"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model directory
        os.makedirs('model', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Initialize components
        self.feature_engine = FeatureEngine()
        self.label_generator = LabelGenerator()
        self.sequence_creator = SequenceCreator(
            sequence_length=self.config['data']['sequence_length'],
            feature_dim=self.config['model']['input_dim']
        )
    
    def load_data(self, data_path: str):
        """Load and preprocess data"""
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Ensure required columns
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        assert all(col in df.columns for col in required_cols), f"Missing columns: {required_cols}"
        
        # Generate features
        print("Generating features...")
        features_df = self.feature_engine.generate_all_features(df)
        print(f"Generated {features_df.shape[1]} features")
        
        # Generate labels
        print("Generating labels...")
        labels_df = self.label_generator.generate_all_labels(df, features_df)
        
        # Create sequences
        print("Creating sequences...")
        X, y_buy, y_sell, y_direction, y_regime = self.sequence_creator.create_sequences(
            features_df, labels_df, fit_scaler=True
        )
        
        print(f"Created {len(X)} sequences")
        print(f"X shape: {X.shape}")
        print(f"Buy signals: {y_buy.sum()}/{len(y_buy)} ({y_buy.mean()*100:.2f}%)")
        print(f"Sell signals: {y_sell.sum()}/{len(y_sell)} ({y_sell.mean()*100:.2f}%)")
        
        # Save scaler
        self.sequence_creator.save_scaler('model/scaler.pkl')
        
        return X, y_buy, y_sell, y_direction, y_regime
    
    def phase1_pretrain(self, train_loader: DataLoader, val_loader: DataLoader):
        """Phase 1: Unsupervised Pretraining"""
        print("\n" + "="*50)
        print("PHASE 1: PRETRAINING (Unsupervised)")
        print("="*50)
        
        # Create pretrain model
        encoder = NatronTransformer(
            input_dim=self.config['model']['input_dim'],
            sequence_length=self.config['data']['sequence_length'],
            d_model=self.config['model']['d_model'],
            nhead=self.config['model']['nhead'],
            num_layers=self.config['model']['num_layers'],
            dim_feedforward=self.config['model']['dim_feedforward'],
            dropout=self.config['model']['dropout']
        ).encoder
        
        pretrain_model = NatronPretrainModel(
            encoder=encoder,
            d_model=self.config['model']['d_model'],
            input_dim=self.config['model']['input_dim'],
            mask_ratio=self.config['pretrain']['mask_ratio']
        ).to(self.device)
        
        # Loss and optimizer
        pretrain_loss_fn = PretrainLoss(
            recon_weight=self.config['pretrain']['recon_weight'],
            contrastive_weight=self.config['pretrain']['contrastive_weight']
        )
        
        optimizer = optim.AdamW(
            pretrain_model.parameters(),
            lr=self.config['pretrain']['lr'],
            weight_decay=self.config['pretrain']['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config['pretrain']['epochs']):
            # Training
            pretrain_model.train()
            train_losses = []
            
            for batch in tqdm(train_loader, desc=f"Pretrain Epoch {epoch+1}/{self.config['pretrain']['epochs']}"):
                features = batch['features'].to(self.device)
                
                optimizer.zero_grad()
                predictions = pretrain_model(features, return_reconstruction=True)
                loss_dict = pretrain_loss_fn(predictions, features)
                loss = loss_dict['total']
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(pretrain_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            pretrain_model.eval()
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    predictions = pretrain_model(features, return_reconstruction=True)
                    loss_dict = pretrain_loss_fn(predictions, features)
                    val_losses.append(loss_dict['total'].item())
            
            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            
            scheduler.step(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(pretrain_model.encoder.state_dict(), 'model/pretrain_encoder.pt')
                print(f"Saved best pretrain encoder (val_loss={best_val_loss:.4f})")
        
        return pretrain_model.encoder
    
    def phase2_supervised(self, train_loader: DataLoader, val_loader: DataLoader,
                         test_loader: DataLoader, pretrained_encoder=None):
        """Phase 2: Supervised Fine-Tuning"""
        print("\n" + "="*50)
        print("PHASE 2: SUPERVISED FINE-TUNING")
        print("="*50)
        
        # Create model
        model = NatronTransformer(
            input_dim=self.config['model']['input_dim'],
            sequence_length=self.config['data']['sequence_length'],
            d_model=self.config['model']['d_model'],
            nhead=self.config['model']['nhead'],
            num_layers=self.config['model']['num_layers'],
            dim_feedforward=self.config['model']['dim_feedforward'],
            dropout=self.config['model']['dropout'],
            freeze_encoder=self.config['supervised']['freeze_encoder']
        ).to(self.device)
        
        # Load pretrained encoder if available
        if pretrained_encoder is not None:
            model.encoder.load_state_dict(pretrained_encoder.state_dict())
            print("Loaded pretrained encoder")
        elif os.path.exists('model/pretrain_encoder.pt'):
            model.encoder.load_state_dict(torch.load('model/pretrain_encoder.pt'))
            print("Loaded pretrained encoder from file")
        
        # Loss and optimizer
        loss_fn = MultiTaskLoss(
            buy_weight=self.config['supervised']['buy_weight'],
            sell_weight=self.config['supervised']['sell_weight'],
            direction_weight=self.config['supervised']['direction_weight'],
            regime_weight=self.config['supervised']['regime_weight'],
            use_focal_loss=self.config['supervised']['use_focal_loss']
        )
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['supervised']['lr'],
            weight_decay=self.config['supervised']['weight_decay']
        )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_val_loss = float('inf')
        best_metrics = {}
        
        for epoch in range(self.config['supervised']['epochs']):
            # Training
            model.train()
            train_losses = {'total': [], 'buy': [], 'sell': [], 'direction': [], 'regime': []}
            
            for batch in tqdm(train_loader, desc=f"Supervised Epoch {epoch+1}/{self.config['supervised']['epochs']}"):
                features = batch['features'].to(self.device)
                targets = {
                    'buy': batch['buy'].to(self.device),
                    'sell': batch['sell'].to(self.device),
                    'direction': batch['direction'].to(self.device),
                    'regime': batch['regime'].to(self.device)
                }
                
                optimizer.zero_grad()
                predictions = model(features)
                loss_dict = loss_fn(predictions, targets)
                
                loss_dict['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                for key in train_losses:
                    train_losses[key].append(loss_dict[key].item())
            
            # Validation
            model.eval()
            val_losses = {'total': [], 'buy': [], 'sell': [], 'direction': [], 'regime': []}
            val_metrics = {'buy_acc': [], 'sell_acc': [], 'direction_acc': [], 'regime_acc': []}
            
            with torch.no_grad():
                for batch in val_loader:
                    features = batch['features'].to(self.device)
                    targets = {
                        'buy': batch['buy'].to(self.device),
                        'sell': batch['sell'].to(self.device),
                        'direction': batch['direction'].to(self.device),
                        'regime': batch['regime'].to(self.device)
                    }
                    
                    predictions = model(features)
                    loss_dict = loss_fn(predictions, targets)
                    
                    for key in val_losses:
                        val_losses[key].append(loss_dict[key].item())
                    
                    # Metrics
                    buy_pred = (predictions['buy'] > 0.5).float()
                    val_metrics['buy_acc'].append((buy_pred == targets['buy']).float().mean().item())
                    
                    sell_pred = (predictions['sell'] > 0.5).float()
                    val_metrics['sell_acc'].append((sell_pred == targets['sell']).float().mean().item())
                    
                    direction_pred = predictions['direction'].argmax(dim=1)
                    val_metrics['direction_acc'].append((direction_pred == targets['direction']).float().mean().item())
                    
                    regime_pred = predictions['regime'].argmax(dim=1)
                    val_metrics['regime_acc'].append((regime_pred == targets['regime']).float().mean().item())
            
            avg_train_loss = {k: np.mean(v) for k, v in train_losses.items()}
            avg_val_loss = {k: np.mean(v) for k, v in val_losses.items()}
            avg_val_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
            
            scheduler.step(avg_val_loss['total'])
            
            print(f"\nEpoch {epoch+1}:")
            print(f"  Train Loss: {avg_train_loss['total']:.4f}")
            print(f"  Val Loss: {avg_val_loss['total']:.4f}")
            print(f"  Val Acc - Buy: {avg_val_metrics['buy_acc']:.4f}, "
                  f"Sell: {avg_val_metrics['sell_acc']:.4f}, "
                  f"Direction: {avg_val_metrics['direction_acc']:.4f}, "
                  f"Regime: {avg_val_metrics['regime_acc']:.4f}")
            
            # Save best model
            if avg_val_loss['total'] < best_val_loss:
                best_val_loss = avg_val_loss['total']
                best_metrics = avg_val_metrics.copy()
                torch.save(model.state_dict(), 'model/natron_v2.pt')
                print(f"  ✓ Saved best model (val_loss={best_val_loss:.4f})")
        
        # Test evaluation
        print("\n" + "="*50)
        print("TEST EVALUATION")
        print("="*50)
        
        model.load_state_dict(torch.load('model/natron_v2.pt'))
        model.eval()
        
        test_metrics = {'buy_acc': [], 'sell_acc': [], 'direction_acc': [], 'regime_acc': []}
        
        with torch.no_grad():
            for batch in test_loader:
                features = batch['features'].to(self.device)
                targets = {
                    'buy': batch['buy'].to(self.device),
                    'sell': batch['sell'].to(self.device),
                    'direction': batch['direction'].to(self.device),
                    'regime': batch['regime'].to(self.device)
                }
                
                predictions = model(features)
                
                buy_pred = (predictions['buy'] > 0.5).float()
                test_metrics['buy_acc'].append((buy_pred == targets['buy']).float().mean().item())
                
                sell_pred = (predictions['sell'] > 0.5).float()
                test_metrics['sell_acc'].append((sell_pred == targets['sell']).float().mean().item())
                
                direction_pred = predictions['direction'].argmax(dim=1)
                test_metrics['direction_acc'].append((direction_pred == targets['direction']).float().mean().item())
                
                regime_pred = predictions['regime'].argmax(dim=1)
                test_metrics['regime_acc'].append((regime_pred == targets['regime']).float().mean().item())
        
        avg_test_metrics = {k: np.mean(v) for k, v in test_metrics.items()}
        print(f"\nTest Metrics:")
        print(f"  Buy Accuracy: {avg_test_metrics['buy_acc']:.4f}")
        print(f"  Sell Accuracy: {avg_test_metrics['sell_acc']:.4f}")
        print(f"  Direction Accuracy: {avg_test_metrics['direction_acc']:.4f}")
        print(f"  Regime Accuracy: {avg_test_metrics['regime_acc']:.4f}")
        
        return model
    
    def train(self, data_path: str):
        """Full training pipeline"""
        # Load data
        X, y_buy, y_sell, y_direction, y_regime = self.load_data(data_path)
        
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders(
            X, y_buy, y_sell, y_direction, y_regime,
            train_ratio=self.config['data']['train_ratio'],
            val_ratio=self.config['data']['val_ratio'],
            batch_size=self.config['training']['batch_size'],
            shuffle=True
        )
        
        # Phase 1: Pretraining
        if self.config['pretrain']['enabled']:
            pretrained_encoder = self.phase1_pretrain(train_loader, val_loader)
        else:
            pretrained_encoder = None
        
        # Phase 2: Supervised fine-tuning
        model = self.phase2_supervised(train_loader, val_loader, test_loader, pretrained_encoder)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETE")
        print("="*50)
        print(f"Model saved to: model/natron_v2.pt")
        print(f"Scaler saved to: model/scaler.pkl")


def main():
    parser = argparse.ArgumentParser(description='Train Natron Transformer')
    parser.add_argument('--data', type=str, default='data_export.csv', help='Path to data CSV')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML')
    args = parser.parse_args()
    
    trainer = NatronTrainer(config_path=args.config)
    trainer.train(data_path=args.data)


if __name__ == '__main__':
    main()
