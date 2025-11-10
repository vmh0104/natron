"""
Natron Main Training Script - End-to-end training pipeline
"""
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import argparse
import yaml
import os

from feature_engine import FeatureEngine
from label_generator import LabelGenerator
from sequence_creator import SequenceCreator
from model import create_model
from trainer import NatronTrainer, FinancialDataset


def load_data(csv_path: str) -> pd.DataFrame:
    """Load OHLCV data from CSV"""
    df = pd.read_csv(csv_path)
    
    # Ensure time column is datetime
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
    else:
        df.index = pd.to_datetime(df.index)
    
    # Ensure required columns exist
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    return df


def prepare_data(
    df: pd.DataFrame,
    sequence_length: int = 96,
    test_size: float = 0.2,
    val_size: float = 0.1
):
    """Prepare data for training"""
    print("Extracting features...")
    feature_engine = FeatureEngine()
    features_df = feature_engine.extract_all_features(df)
    print(f"Features shape: {features_df.shape}")
    
    print("Generating labels...")
    label_generator = LabelGenerator()
    labels_df = label_generator.generate_labels(df, features_df)
    print(f"Labels shape: {labels_df.shape}")
    
    print("Creating sequences...")
    sequence_creator = SequenceCreator(sequence_length=sequence_length)
    X, y, metadata = sequence_creator.create_sequences_from_raw(df, features_df, labels_df)
    print(f"Sequences shape: {X.shape}")
    
    # Split data
    indices = np.arange(len(X))
    train_idx, temp_idx = train_test_split(indices, test_size=test_size + val_size, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=val_size / (test_size + val_size), random_state=42)
    
    X_train, y_train = X[train_idx], {k: v[train_idx] for k, v in y.items()}
    X_val, y_val = X[val_idx], {k: v[val_idx] for k, v in y.items()}
    X_test, y_test = X[test_idx], {k: v[test_idx] for k, v in y.items()}
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), features_df.columns


def main():
    parser = argparse.ArgumentParser(description='Train Natron Transformer')
    parser.add_argument('--data', type=str, default='data_export.csv', help='Path to data CSV')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    parser.add_argument('--phase', type=str, choices=['1', '2', 'both'], default='both', 
                       help='Training phase: 1=pretrain, 2=supervised, both=both')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'model': {
                'd_model': 256,
                'nhead': 8,
                'num_layers': 6,
                'dim_feedforward': 1024,
                'dropout': 0.1
            },
            'training': {
                'batch_size': 32,
                'learning_rate': 1e-4,
                'weight_decay': 1e-5,
                'pretrain_epochs': 10,
                'supervised_epochs': 50
            },
            'data': {
                'sequence_length': 96
            }
        }
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print(f"Loading data from {args.data}...")
    df = load_data(args.data)
    print(f"Data shape: {df.shape}")
    
    # Prepare data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feature_names = prepare_data(
        df,
        sequence_length=config['data']['sequence_length'],
        test_size=0.2,
        val_size=0.1
    )
    
    num_features = len(feature_names)
    print(f"Number of features: {num_features}")
    
    # Create datasets
    train_dataset = FinancialDataset(X_train, y_train)
    val_dataset = FinancialDataset(X_val, y_val)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True if device == 'cuda' else False
    )
    
    # Create model
    if args.phase == '1' or args.phase == 'both':
        print("Creating pretraining model...")
        model = create_model(
            num_features=num_features,
            d_model=config['model']['d_model'],
            nhead=config['model']['nhead'],
            num_layers=config['model']['num_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout'],
            pretrain=True
        )
    else:
        print("Creating supervised model...")
        model = create_model(
            num_features=num_features,
            d_model=config['model']['d_model'],
            nhead=config['model']['nhead'],
            num_layers=config['model']['num_layers'],
            dim_feedforward=config['model']['dim_feedforward'],
            dropout=config['model']['dropout'],
            pretrain=False
        )
    
    # Create trainer
    trainer = NatronTrainer(
        model=model,
        device=device,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Resume if specified
    if args.resume:
        trainer.load_model(args.resume)
    
    # Phase 1: Pretraining
    if args.phase == '1' or args.phase == 'both':
        print("\n" + "="*50)
        print("PHASE 1: PRETRAINING")
        print("="*50)
        
        # Create unsupervised dataset (no labels needed)
        pretrain_dataset = FinancialDataset(X_train)
        pretrain_loader = DataLoader(
            pretrain_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True if device == 'cuda' else False
        )
        
        pretrain_losses = trainer.train_phase1_pretrain(
            pretrain_loader,
            epochs=config['training']['pretrain_epochs']
        )
        
        # Save pretrained model
        os.makedirs('models', exist_ok=True)
        trainer.save_model('models/natron_pretrained.pt')
    
    # Phase 2: Supervised Fine-tuning
    if args.phase == '2' or args.phase == 'both':
        print("\n" + "="*50)
        print("PHASE 2: SUPERVISED FINE-TUNING")
        print("="*50)
        
        # If we did pretraining, switch to supervised mode
        if args.phase == 'both':
            # Extract base model if wrapped
            if hasattr(model, 'base_model'):
                base_model = model.base_model
            else:
                base_model = model
            
            # Create new supervised model with same architecture
            model = create_model(
                num_features=num_features,
                d_model=config['model']['d_model'],
                nhead=config['model']['nhead'],
                num_layers=config['model']['num_layers'],
                dim_feedforward=config['model']['dim_feedforward'],
                dropout=config['model']['dropout'],
                pretrain=False
            )
            
            # Copy encoder weights
            if hasattr(base_model, 'transformer'):
                model.transformer.load_state_dict(base_model.transformer.state_dict())
                model.input_projection.load_state_dict(base_model.input_projection.state_dict())
                model.pos_encoder.load_state_dict(base_model.pos_encoder.state_dict())
                print("Loaded pretrained encoder weights")
            
            trainer = NatronTrainer(
                model=model,
                device=device,
                learning_rate=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay']
            )
        
        history = trainer.train_phase2_supervised(
            train_loader,
            val_loader,
            epochs=config['training']['supervised_epochs']
        )
        
        # Save final model
        os.makedirs('models', exist_ok=True)
        trainer.save_model('models/natron_v2.pt')
        
        print("\nTraining completed!")
        print(f"Final train loss: {history['train_loss'][-1]:.4f}")
        if history['val_loss']:
            print(f"Final val loss: {history['val_loss'][-1]:.4f}")


if __name__ == '__main__':
    main()
