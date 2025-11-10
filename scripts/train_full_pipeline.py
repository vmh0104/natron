#!/usr/bin/env python3
"""
Natron Full Training Pipeline
Runs all three training phases sequentially:
1. Unsupervised Pretraining
2. Supervised Multi-Task Training
3. Reinforcement Learning (Optional)

Author: Natron AI System
"""

import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.pretrain import NatronPretrainer
from src.training.train_supervised import NatronSupervisedTrainer
from src.training.train_rl import train_rl
from src.data.dataset_loader import NatronDataModule


def train_phase1_pretrain(config_path: str, skip_if_exists: bool = True):
    """
    Phase 1: Unsupervised Pretraining
    """
    print("\n" + "=" * 80)
    print("🚀 PHASE 1: UNSUPERVISED PRETRAINING")
    print("=" * 80)
    
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    checkpoint_path = config['pretrain']['checkpoint_path'].replace('.pt', '_best.pt')
    
    if skip_if_exists and Path(checkpoint_path).exists():
        print(f"⚠️ Pretrained model already exists at {checkpoint_path}")
        print("   Skipping Phase 1. Use --no-skip to retrain.")
        return True
    
    try:
        # Load data
        print("\n📂 Loading and preparing data...")
        data_module = NatronDataModule(config_path=config_path)
        pipeline_data = data_module.prepare_full_pipeline(mode='contrastive')
        
        # Create and train pretrainer
        pretrainer = NatronPretrainer(config_path=config_path)
        pretrainer.create_model(num_features=pipeline_data['num_features'])
        pretrainer.create_optimizer()
        
        pretrainer.train(
            train_loader=pipeline_data['train_loader'],
            val_loader=pipeline_data['val_loader']
        )
        
        print("\n✅ Phase 1 completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_phase2_supervised(config_path: str, skip_if_exists: bool = True, use_pretrained: bool = True):
    """
    Phase 2: Supervised Multi-Task Training
    """
    print("\n" + "=" * 80)
    print("🚀 PHASE 2: SUPERVISED MULTI-TASK TRAINING")
    print("=" * 80)
    
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    checkpoint_path = config['supervised']['checkpoint_path'].replace('.pt', '_best.pt')
    
    if skip_if_exists and Path(checkpoint_path).exists():
        print(f"⚠️ Supervised model already exists at {checkpoint_path}")
        print("   Skipping Phase 2. Use --no-skip to retrain.")
        return True
    
    try:
        # Load data
        print("\n📂 Loading and preparing data...")
        data_module = NatronDataModule(config_path=config_path)
        pipeline_data = data_module.prepare_full_pipeline(mode='supervised')
        
        # Create and train supervised trainer
        trainer = NatronSupervisedTrainer(config_path=config_path)
        trainer.create_model(
            num_features=pipeline_data['num_features'],
            load_pretrained=use_pretrained
        )
        trainer.create_optimizer()
        
        trainer.train(
            train_loader=pipeline_data['train_loader'],
            val_loader=pipeline_data['val_loader']
        )
        
        print("\n✅ Phase 2 completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_phase3_rl(config_path: str, skip_if_exists: bool = True):
    """
    Phase 3: Reinforcement Learning (Optional)
    """
    print("\n" + "=" * 80)
    print("🚀 PHASE 3: REINFORCEMENT LEARNING")
    print("=" * 80)
    
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    checkpoint_path = config['reinforcement']['checkpoint_path'].replace('.pt', '_best.pt')
    
    if skip_if_exists and Path(checkpoint_path).exists():
        print(f"⚠️ RL model already exists at {checkpoint_path}")
        print("   Skipping Phase 3. Use --no-skip to retrain.")
        return True
    
    try:
        train_rl(config_path=config_path)
        print("\n✅ Phase 3 completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Phase 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Natron Full Training Pipeline')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--phases', type=str, default='1,2,3',
                        help='Phases to run (comma-separated): 1=pretrain, 2=supervised, 3=rl')
    parser.add_argument('--no-skip', action='store_true',
                        help='Force retraining even if checkpoints exist')
    parser.add_argument('--no-pretrained', action='store_true',
                        help='Train supervised from scratch (skip loading pretrained weights)')
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("🧠 NATRON TRANSFORMER - FULL TRAINING PIPELINE")
    print("=" * 80)
    print(f"\n📋 Configuration:")
    print(f"   Config file: {args.config}")
    print(f"   Phases: {args.phases}")
    print(f"   Skip existing: {not args.no_skip}")
    print(f"   Use pretrained: {not args.no_pretrained}")
    
    skip_existing = not args.no_skip
    use_pretrained = not args.no_pretrained
    phases = [int(p.strip()) for p in args.phases.split(',')]
    
    results = {}
    
    # Phase 1: Pretraining
    if 1 in phases:
        results['phase1'] = train_phase1_pretrain(args.config, skip_if_exists=skip_existing)
        if not results['phase1']:
            print("\n❌ Pipeline stopped due to Phase 1 failure")
            return
    
    # Phase 2: Supervised
    if 2 in phases:
        results['phase2'] = train_phase2_supervised(
            args.config, 
            skip_if_exists=skip_existing,
            use_pretrained=use_pretrained
        )
        if not results['phase2']:
            print("\n❌ Pipeline stopped due to Phase 2 failure")
            return
    
    # Phase 3: Reinforcement Learning
    if 3 in phases:
        results['phase3'] = train_phase3_rl(args.config, skip_if_exists=skip_existing)
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TRAINING PIPELINE SUMMARY")
    print("=" * 80)
    
    for phase, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {phase.upper()}: {status}")
    
    if all(results.values()):
        print("\n🎉 All training phases completed successfully!")
        print("\n📁 Model checkpoints saved in: model/")
        print("\n🚀 Next steps:")
        print("   1. Start the inference server: python src/server/api_server.py")
        print("   2. Deploy the MQL5 EA to MetaTrader 5")
        print("   3. Monitor performance and iterate!")
    else:
        print("\n⚠️ Some phases failed. Please check the logs above.")


if __name__ == "__main__":
    main()
