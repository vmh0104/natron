"""
Natron V2 Main Training Script
End-to-end training pipeline: Pretrain -> Supervised -> RL (optional)
"""

import yaml
import argparse
import torch
import os

from src.data.dataset_loader import DataProcessor
from src.training.pretrain import PretrainManager
from src.training.train_supervised import SupervisedTrainer
from src.training.train_rl import RLTrainer


def main():
    parser = argparse.ArgumentParser(description='Natron V2 Training Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/data_export.csv',
                       help='Path to OHLCV CSV data')
    parser.add_argument('--phase', type=str, default='all',
                       choices=['all', 'pretrain', 'supervised', 'rl'],
                       help='Training phase to execute')
    parser.add_argument('--pretrain-checkpoint', type=str, default=None,
                       help='Path to pretrain checkpoint (for supervised)')
    parser.add_argument('--supervised-checkpoint', type=str, default=None,
                       help='Path to supervised checkpoint (for RL)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    parser.add_argument('--skip-pretrain', action='store_true',
                       help='Skip pretraining phase')
    parser.add_argument('--skip-rl', action='store_true',
                       help='Skip reinforcement learning phase')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override device if specified
    if args.device == 'cpu' or not torch.cuda.is_available():
        config['deployment']['device'] = 'cpu'
        device = 'cpu'
    else:
        device = 'cuda'
    
    print("=" * 80)
    print("NATRON V2 TRAINING PIPELINE")
    print("=" * 80)
    print(f"Configuration: {args.config}")
    print(f"Data: {args.data}")
    print(f"Device: {device}")
    print(f"Phase: {args.phase}")
    print("=" * 80)
    
    # Create directories
    os.makedirs('model', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # =========================================================================
    # DATA LOADING & PREPROCESSING
    # =========================================================================
    print("\n" + "=" * 80)
    print("STAGE 1: DATA LOADING & PREPROCESSING")
    print("=" * 80)
    
    data_processor = DataProcessor(config)
    train_data, val_data, test_data = data_processor.load_and_process(args.data)
    
    # Save scaler
    data_processor.save_scaler('model/scaler.pkl')
    
    # Create dataloaders
    batch_size = config['training']['batch_size']
    train_loader, val_loader, test_loader = data_processor.create_dataloaders(
        train_data, val_data, test_data, batch_size
    )
    
    print(f"\n✓ Data preprocessing complete")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")
    print(f"  Test batches: {len(test_loader)}")
    
    # =========================================================================
    # PHASE 1: PRETRAINING (Optional)
    # =========================================================================
    pretrain_checkpoint = args.pretrain_checkpoint
    
    if args.phase in ['all', 'pretrain'] and not args.skip_pretrain:
        print("\n" + "=" * 80)
        print("STAGE 2: PHASE 1 - PRETRAINING")
        print("=" * 80)
        
        pretrain_loader = data_processor.create_pretrain_dataloader(
            train_data,
            batch_size,
            config['pretrain']['mask_ratio']
        )
        
        pretrain_manager = PretrainManager(config, device)
        pretrain_manager.train(
            pretrain_loader,
            num_epochs=config['training']['epochs_pretrain'],
            use_contrastive=True
        )
        
        pretrain_checkpoint = 'model/pretrain_best.pt'
        print(f"\n✓ Pretraining complete. Checkpoint: {pretrain_checkpoint}")
    else:
        print("\n⊗ Skipping pretraining phase")
        if pretrain_checkpoint:
            print(f"  Using existing checkpoint: {pretrain_checkpoint}")
    
    # =========================================================================
    # PHASE 2: SUPERVISED TRAINING
    # =========================================================================
    supervised_checkpoint = args.supervised_checkpoint
    
    if args.phase in ['all', 'supervised']:
        print("\n" + "=" * 80)
        print("STAGE 3: PHASE 2 - SUPERVISED TRAINING")
        print("=" * 80)
        
        supervised_trainer = SupervisedTrainer(
            config,
            device,
            pretrain_checkpoint=pretrain_checkpoint
        )
        
        # Train with optional encoder freezing
        freeze_epochs = 10 if pretrain_checkpoint else 0
        
        supervised_trainer.train(
            train_loader,
            val_loader,
            num_epochs=config['training']['epochs_supervised'],
            freeze_encoder_epochs=freeze_epochs
        )
        
        supervised_checkpoint = 'model/supervised_best.pt'
        
        # Evaluate on test set
        print("\n" + "=" * 80)
        print("EVALUATING ON TEST SET")
        print("=" * 80)
        test_metrics = supervised_trainer.validate(test_loader, epoch=0)
        print(f"\nTest Results:")
        for metric, value in test_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Save as final model
        import shutil
        shutil.copy(supervised_checkpoint, 'model/natron_v2.pt')
        print(f"\n✓ Model saved to model/natron_v2.pt")
    else:
        print("\n⊗ Skipping supervised training phase")
        if supervised_checkpoint:
            print(f"  Using existing checkpoint: {supervised_checkpoint}")
    
    # =========================================================================
    # PHASE 3: REINFORCEMENT LEARNING (Optional)
    # =========================================================================
    if args.phase in ['all', 'rl'] and not args.skip_rl:
        print("\n" + "=" * 80)
        print("STAGE 4: PHASE 3 - REINFORCEMENT LEARNING")
        print("=" * 80)
        
        if not supervised_checkpoint:
            supervised_checkpoint = 'model/supervised_best.pt'
        
        rl_trainer = RLTrainer(
            config,
            device,
            supervised_checkpoint=supervised_checkpoint
        )
        
        rl_trainer.train(
            train_data,
            num_episodes=config['training']['epochs_rl'],
            steps_per_episode=1000
        )
        
        # Optionally save RL model as final model
        import shutil
        shutil.copy('model/rl_best.pt', 'model/natron_v2_rl.pt')
        print(f"\n✓ RL model saved to model/natron_v2_rl.pt")
    else:
        print("\n⊗ Skipping reinforcement learning phase")
    
    # =========================================================================
    # TRAINING COMPLETE
    # =========================================================================
    print("\n" + "=" * 80)
    print("🎉 NATRON V2 TRAINING PIPELINE COMPLETE 🎉")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Start the inference server:")
    print("   python src/inference/socket_server.py")
    print("   or")
    print("   python src/inference/flask_api.py")
    print("\n2. Connect MetaTrader 5 with the Natron EA (mql5/natron_ea.mq5)")
    print("\n3. Monitor performance and iterate!")
    print("=" * 80)


if __name__ == '__main__':
    main()
