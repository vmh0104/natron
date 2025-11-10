#!/usr/bin/env python3
"""
Natron Transformer - Main Training Orchestrator
Runs complete training pipeline: Phase 1 (Pretrain) → Phase 2 (Supervised)

Usage:
    python train_natron.py --config config.yaml
    python train_natron.py --phase 1  # Only pretrain
    python train_natron.py --phase 2  # Only supervised
"""

import argparse
import yaml
import os
import sys
import torch

# Add src to path
sys.path.append('src')

from train_pretrain import main as run_pretrain
from train_supervised import main as run_supervised


def print_banner():
    """Print Natron banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║   ███╗   ██╗ █████╗ ████████╗██████╗  ██████╗ ███╗   ██╗       ║
║   ████╗  ██║██╔══██╗╚══██╔══╝██╔══██╗██╔═══██╗████╗  ██║       ║
║   ██╔██╗ ██║███████║   ██║   ██████╔╝██║   ██║██╔██╗ ██║       ║
║   ██║╚██╗██║██╔══██║   ██║   ██╔══██╗██║   ██║██║╚██╗██║       ║
║   ██║ ╚████║██║  ██║   ██║   ██║  ██║╚██████╔╝██║ ╚████║       ║
║   ╚═╝  ╚═══╝╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝       ║
║                                                                  ║
║        Multi-Task Transformer for Financial Trading             ║
║                      Version 2.0                                 ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_requirements():
    """Check if all requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA version: {torch.version.cuda}")
    else:
        print("⚠️  CUDA not available, will use CPU (training will be slower)")
    
    # Check data file
    config_path = 'config.yaml'
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    data_path = config['data']['csv_path']
    if not os.path.exists(data_path):
        print(f"⚠️  Data file not found: {data_path}")
        print(f"   Please place your OHLCV data at: {data_path}")
        print(f"   Required columns: time, open, high, low, close, volume")
        return False
    
    print(f"✅ Data file found: {data_path}")
    
    # Create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    os.makedirs(config['pretrain']['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['supervised']['checkpoint_dir'], exist_ok=True)
    
    print("✅ All directories created")
    
    return True


def run_full_pipeline():
    """Run complete training pipeline"""
    print("\n" + "="*70)
    print("🚀 Starting Natron Complete Training Pipeline")
    print("="*70)
    
    # Phase 1: Pretraining
    print("\n" + "="*70)
    print("📍 PHASE 1: Unsupervised Pretraining")
    print("="*70)
    
    try:
        run_pretrain()
        print("\n✅ Phase 1 complete!")
    except Exception as e:
        print(f"\n❌ Phase 1 failed: {e}")
        return False
    
    # Phase 2: Supervised Fine-tuning
    print("\n" + "="*70)
    print("📍 PHASE 2: Supervised Fine-Tuning")
    print("="*70)
    
    try:
        run_supervised()
        print("\n✅ Phase 2 complete!")
    except Exception as e:
        print(f"\n❌ Phase 2 failed: {e}")
        return False
    
    # Success
    print("\n" + "="*70)
    print("🎉 NATRON TRAINING COMPLETE!")
    print("="*70)
    print("\n✅ Model saved to: models/natron_v2.pt")
    print("✅ Scaler saved to: models/scaler.pkl")
    print("\n📊 Next steps:")
    print("   1. Test the model: python inference.py")
    print("   2. Start API server: python server_natron.py")
    print("   3. Connect MQL5 EA to MetaTrader 5")
    print("\n" + "="*70)
    
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Natron Transformer Training Pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--phase',
        type=int,
        choices=[1, 2],
        help='Run specific phase only (1=pretrain, 2=supervised)'
    )
    parser.add_argument(
        '--skip-check',
        action='store_true',
        help='Skip requirements check'
    )
    
    args = parser.parse_args()
    
    # Print banner
    print_banner()
    
    # Check requirements
    if not args.skip_check:
        if not check_requirements():
            print("\n❌ Requirements check failed. Please fix issues above.")
            return 1
    
    # Run training
    try:
        if args.phase == 1:
            # Only pretrain
            print("\n🎯 Running Phase 1 only (Pretraining)")
            run_pretrain()
            
        elif args.phase == 2:
            # Only supervised
            print("\n🎯 Running Phase 2 only (Supervised)")
            run_supervised()
            
        else:
            # Full pipeline
            success = run_full_pipeline()
            if not success:
                return 1
        
        print("\n✅ Training completed successfully!")
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
