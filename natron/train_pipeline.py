"""
Natron End-to-End Training Pipeline
"""
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import argparse
import os
import yaml
from pathlib import Path

from feature_engine import FeatureEngine
from label_generator import LabelGenerator
from sequence_creator import SequenceCreator
from model import create_model
from training import PretrainingTrainer, SupervisedTrainer, SequenceDataset
from torch.utils.data import DataLoader


def load_config(config_path: str = 'config.yaml') -> dict:
    """Load configuration from YAML file."""
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def main():
    parser = argparse.ArgumentParser(description='Train Natron Transformer Model')
    parser.add_argument('--data', type=str, default='data_export.csv', help='Path to data CSV')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config YAML')
    parser.add_argument('--phase', type=str, choices=['pretrain', 'supervised', 'both'], 
                       default='both', help='Training phase')
    parser.add_argument('--pretrain_epochs', type=int, default=10, help='Pretraining epochs')
    parser.add_argument('--supervised_epochs', type=int, default=50, help='Supervised epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line args
    config.setdefault('training', {})
    config['training']['pretrain_epochs'] = args.pretrain_epochs
    config['training']['supervised_epochs'] = args.supervised_epochs
    config['training']['batch_size'] = args.batch_size
    config['training']['device'] = args.device
    
    print("=" * 80)
    print("🧠 Natron Transformer - Multi-Task Financial Trading Model")
    print("=" * 80)
    
    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Step 1: Load data
    print("\n[1/5] Loading data...")
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Data file not found: {args.data}")
    
    df = pd.read_csv(args.data)
    print(f"Loaded {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Step 2: Feature Engineering
    print("\n[2/5] Generating features...")
    feature_engine = FeatureEngine()
    df_features = feature_engine.generate_features(df)
    feature_columns = feature_engine.get_feature_columns(df_features)
    print(f"Generated {len(feature_columns)} features")
    
    # Step 3: Label Generation
    print("\n[3/5] Generating labels...")
    label_generator = LabelGenerator()
    df_labeled = label_generator.generate_labels(df_features)
    print(f"Label distribution:")
    print(f"  Buy: {df_labeled['buy'].sum()} ({df_labeled['buy'].mean()*100:.2f}%)")
    print(f"  Sell: {df_labeled['sell'].sum()} ({df_labeled['sell'].mean()*100:.2f}%)")
    print(f"  Direction Up: {(df_labeled['direction']==1).sum()} ({(df_labeled['direction']==1).mean()*100:.2f}%)")
    print(f"  Regime distribution:")
    for regime_id in range(6):
        count = (df_labeled['regime'] == regime_id).sum()
        name = label_generator.get_regime_name(regime_id)
        print(f"    {regime_id} ({name}): {count} ({count/len(df_labeled)*100:.2f}%)")
    
    # Step 4: Sequence Creation
    print("\n[4/5] Creating sequences...")
    sequence_creator = SequenceCreator(sequence_length=96)
    X, y = sequence_creator.create_sequences(df_labeled, feature_columns)
    print(f"Created {len(X)} sequences of shape {X.shape}")
    
    # Train/Val split
    indices = np.arange(len(X))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, shuffle=True)
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train = {k: v[train_idx] for k, v in y.items()}
    y_val = {k: v[val_idx] for k, v in y.items()}
    
    print(f"Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Create datasets
    train_dataset = SequenceDataset(X_train, y_train)
    val_dataset = SequenceDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Step 5: Model Creation
    print("\n[5/5] Creating model...")
    model_config = config.get('model', {})
    model_config.setdefault('d_model', 128)
    model_config.setdefault('nhead', 8)
    model_config.setdefault('num_layers', 6)
    model_config.setdefault('dim_feedforward', 512)
    model_config.setdefault('dropout', 0.1)
    model_config.setdefault('num_features', len(feature_columns))
    model_config.setdefault('max_seq_len', 96)
    
    model = create_model(model_config)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Phase 1: Pretraining
    if args.phase in ['pretrain', 'both']:
        print("\n" + "=" * 80)
        print("🧩 Phase 1: Pretraining (Masked Modeling)")
        print("=" * 80)
        
        pretrain_trainer = PretrainingTrainer(model, device, mask_prob=0.15)
        model = pretrain_trainer.train(
            train_loader=train_loader,
            num_epochs=args.pretrain_epochs,
            lr=config['training'].get('pretrain_lr', 1e-4),
            weight_decay=config['training'].get('weight_decay', 1e-5)
        )
        print("✓ Pretraining completed")
    
    # Phase 2: Supervised Fine-tuning
    if args.phase in ['supervised', 'both']:
        print("\n" + "=" * 80)
        print("🧩 Phase 2: Supervised Fine-Tuning")
        print("=" * 80)
        
        supervised_trainer = SupervisedTrainer(model, device)
        model = supervised_trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=args.supervised_epochs,
            lr=config['training'].get('supervised_lr', 1e-4),
            weight_decay=config['training'].get('weight_decay', 1e-5),
            patience=config['training'].get('patience', 5),
            save_path=config.get('model_path', 'model/natron_v2.pt')
        )
        print("✓ Supervised training completed")
    
    print("\n" + "=" * 80)
    print("✅ Training pipeline completed!")
    print(f"Model saved to: {config.get('model_path', 'model/natron_v2.pt')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
