"""
Main Training Pipeline: End-to-End Training Orchestrator
"""
import pandas as pd
import numpy as np
import argparse
import os
import yaml
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from feature_engine import FeatureEngine
from label_generator import LabelGenerator
from sequence_creator import SequenceCreator
from train_phase1_pretrain import Pretrainer, MaskedDataset, ContrastiveDataset
from train_phase2_supervised import SupervisedTrainer, TradingDataset
from train_phase3_rl import PPOTrainer, TradingEnvironment


class NatronTrainingPipeline:
    """End-to-end training pipeline"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize components
        self.feature_engine = FeatureEngine()
        self.label_generator = LabelGenerator()
        self.sequence_creator = SequenceCreator(
            sequence_length=self.config['data']['sequence_length']
        )
        
        # Create output directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
    
    def load_data(self, data_path: str) -> pd.DataFrame:
        """Load and preprocess data"""
        print(f"Loading data from {data_path}...")
        
        df = pd.read_csv(data_path)
        
        # Ensure required columns exist
        required_cols = ['time', 'open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert time to datetime if needed
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        # Sort by time
        df = df.sort_values('time').reset_index(drop=True)
        
        print(f"Loaded {len(df)} candles")
        return df
    
    def prepare_features_and_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate features and labels"""
        print("Generating features...")
        df_features = self.feature_engine.generate_features(df)
        
        print("Generating labels...")
        df_labeled = self.label_generator.generate_labels(df_features)
        
        # Get feature columns
        feature_columns = self.feature_engine.get_feature_columns(df_labeled)
        print(f"Generated {len(feature_columns)} features")
        
        return df_labeled, feature_columns
    
    def create_sequences(self, df: pd.DataFrame, feature_columns: list):
        """Create sequences and splits"""
        print("Creating sequences...")
        
        X_splits, y_splits = self.sequence_creator.create_sequences_with_splits(
            df=df,
            feature_columns=feature_columns,
            label_columns=['buy', 'sell', 'direction', 'regime'],
            train_ratio=self.config['data']['train_ratio'],
            val_ratio=self.config['data']['val_ratio']
        )
        
        print(f"Train sequences: {len(X_splits['train'])}")
        print(f"Val sequences: {len(X_splits['val'])}")
        print(f"Test sequences: {len(X_splits['test'])}")
        
        return X_splits, y_splits
    
    def phase1_pretrain(self, X_splits: dict):
        """Phase 1: Pretraining"""
        if not self.config['training']['phase1']['enabled']:
            print("Phase 1 pretraining disabled")
            return None
        
        print("\n" + "="*50)
        print("PHASE 1: PRETRAINING")
        print("="*50)
        
        pretrainer = Pretrainer(
            input_dim=self.config['model']['input_dim'],
            d_model=self.config['model']['d_model'],
            nhead=self.config['model']['nhead'],
            num_layers=self.config['model']['num_layers']
        )
        
        # Masked modeling
        if self.config['training']['phase1']['masked_modeling']['enabled']:
            print("\nTraining with Masked Modeling...")
            train_dataset = MaskedDataset(
                X_splits['train'],
                mask_prob=self.config['training']['phase1']['masked_modeling']['mask_prob']
            )
            val_dataset = MaskedDataset(
                X_splits['val'],
                mask_prob=self.config['training']['phase1']['masked_modeling']['mask_prob']
            )
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['training']['phase1']['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['training']['phase1']['batch_size'],
                shuffle=False
            )
            
            pretrainer.train_masked_modeling(
                train_loader,
                val_loader,
                epochs=self.config['training']['phase1']['masked_modeling']['epochs'],
                lr=self.config['training']['phase1']['lr'],
                save_path='models/pretrain_masked.pt'
            )
        
        # Contrastive learning
        if self.config['training']['phase1']['contrastive']['enabled']:
            print("\nTraining with Contrastive Learning...")
            train_dataset = ContrastiveDataset(X_splits['train'])
            val_dataset = ContrastiveDataset(X_splits['val'])
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config['training']['phase1']['batch_size'],
                shuffle=True
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config['training']['phase1']['batch_size'],
                shuffle=False
            )
            
            pretrainer.train_contrastive(
                train_loader,
                val_loader,
                epochs=self.config['training']['phase1']['contrastive']['epochs'],
                lr=self.config['training']['phase1']['lr'],
                save_path='models/pretrain_contrastive.pt'
            )
        
        return 'models/pretrain_contrastive.pt'  # Return path to pretrained encoder
    
    def phase2_supervised(self, X_splits: dict, y_splits: dict, pretrained_path: str = None):
        """Phase 2: Supervised Fine-Tuning"""
        if not self.config['training']['phase2']['enabled']:
            print("Phase 2 supervised training disabled")
            return None
        
        print("\n" + "="*50)
        print("PHASE 2: SUPERVISED FINE-TUNING")
        print("="*50)
        
        trainer = SupervisedTrainer(
            input_dim=self.config['model']['input_dim'],
            d_model=self.config['model']['d_model'],
            nhead=self.config['model']['nhead'],
            num_layers=self.config['model']['num_layers'],
            pretrained_encoder_path=pretrained_path
        )
        
        # Create datasets
        train_dataset = TradingDataset(X_splits['train'], y_splits['train'])
        val_dataset = TradingDataset(X_splits['val'], y_splits['val'])
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['phase2']['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['phase2']['batch_size'],
            shuffle=False
        )
        
        # Train
        trainer.train(
            train_loader,
            val_loader,
            epochs=self.config['training']['phase2']['epochs'],
            lr=self.config['training']['phase2']['lr'],
            weight_decay=self.config['training']['phase2']['weight_decay'],
            save_path=self.config['training']['phase2']['save_path'],
            freeze_encoder=self.config['training']['phase2']['freeze_encoder']
        )
        
        return self.config['training']['phase2']['save_path']
    
    def phase3_rl(self, X_splits: dict, y_splits: dict, model_path: str):
        """Phase 3: Reinforcement Learning"""
        if not self.config['training']['phase3']['enabled']:
            print("Phase 3 RL training disabled")
            return
        
        print("\n" + "="*50)
        print("PHASE 3: REINFORCEMENT LEARNING")
        print("="*50)
        
        # Load model
        from model import NatronModel
        model = NatronModel(
            input_dim=self.config['model']['input_dim'],
            d_model=self.config['model']['d_model'],
            nhead=self.config['model']['nhead'],
            num_layers=self.config['model']['num_layers']
        )
        
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Create RL trainer
        rl_trainer = PPOTrainer(
            model=model,
            lr=self.config['training']['phase3']['lr'],
            gamma=self.config['training']['phase3']['gamma'],
            eps_clip=self.config['training']['phase3']['eps_clip']
        )
        
        # Use validation set for RL (simulated trading)
        # Extract prices from sequences (assuming close price is in features)
        # This is simplified - in practice, you'd need to extract actual prices
        
        print("RL training requires price data - skipping for now")
        print("To enable RL training, provide price sequences separately")
    
    def run(self, data_path: str):
        """Run complete training pipeline"""
        print("="*50)
        print("NATRON TRANSFORMER TRAINING PIPELINE")
        print("="*50)
        
        # Load data
        df = self.load_data(data_path)
        
        # Generate features and labels
        df_labeled, feature_columns = self.prepare_features_and_labels(df)
        
        # Create sequences
        X_splits, y_splits = self.create_sequences(df_labeled, feature_columns)
        
        # Phase 1: Pretraining
        pretrained_path = None
        if self.config['training']['phase1']['enabled']:
            pretrained_path = self.phase1_pretrain(X_splits)
        
        # Phase 2: Supervised Fine-Tuning
        model_path = None
        if self.config['training']['phase2']['enabled']:
            model_path = self.phase2_supervised(X_splits, y_splits, pretrained_path)
        
        # Phase 3: Reinforcement Learning
        if self.config['training']['phase3']['enabled'] and model_path:
            self.phase3_rl(X_splits, y_splits, model_path)
        
        print("\n" + "="*50)
        print("TRAINING COMPLETE")
        print("="*50)
        if model_path:
            print(f"Model saved to: {model_path}")


if __name__ == '__main__':
    import torch
    
    parser = argparse.ArgumentParser(description='Natron Transformer Training Pipeline')
    parser.add_argument('--data', type=str, default='data_export.csv',
                       help='Path to data CSV file')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to config YAML file')
    
    args = parser.parse_args()
    
    # Check if config exists, create default if not
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        print("Please create config.yaml or use --config to specify path")
        exit(1)
    
    # Run pipeline
    pipeline = NatronTrainingPipeline(config_path=args.config)
    pipeline.run(data_path=args.data)
