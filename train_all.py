"""
Main Training Orchestrator
Runs all training phases sequentially: Pretrain → Supervised → RL
"""

import os
import sys
import yaml
from train_pretrain import main as train_pretrain
from train_supervised import main as train_supervised
from train_rl import main as train_rl


def main():
    """Run all training phases"""
    print("=" * 80)
    print("🧠 Natron Transformer - Complete Training Pipeline")
    print("=" * 80)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Check if data exists
    data_path = config['data']['csv_path']
    if not os.path.exists(data_path):
        print(f"❌ Error: Data file not found: {data_path}")
        print("Please ensure data_export.csv exists in the project directory.")
        sys.exit(1)
    
    print(f"✅ Data file found: {data_path}")
    print()
    
    # Phase 1: Pretraining
    print("=" * 80)
    print("PHASE 1: PRETRAINING (Unsupervised Learning)")
    print("=" * 80)
    try:
        train_pretrain()
        print("✅ Phase 1 completed successfully!")
    except Exception as e:
        print(f"❌ Phase 1 failed: {e}")
        print("Continuing to Phase 2 anyway...")
    print()
    
    # Phase 2: Supervised Fine-tuning
    print("=" * 80)
    print("PHASE 2: SUPERVISED FINE-TUNING (Multi-Task Learning)")
    print("=" * 80)
    try:
        train_supervised()
        print("✅ Phase 2 completed successfully!")
    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
        print("Skipping Phase 3...")
        sys.exit(1)
    print()
    
    # Phase 3: Reinforcement Learning (optional)
    print("=" * 80)
    print("PHASE 3: REINFORCEMENT LEARNING (Optional)")
    print("=" * 80)
    response = input("Do you want to run Phase 3 (RL training)? [y/N]: ")
    if response.lower() == 'y':
        try:
            train_rl()
            print("✅ Phase 3 completed successfully!")
        except Exception as e:
            print(f"❌ Phase 3 failed: {e}")
    else:
        print("Skipping Phase 3 (RL training).")
    print()
    
    print("=" * 80)
    print("🎉 Training Pipeline Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Start API server: python api_server.py")
    print("2. Start Socket server: python socket_server.py")
    print("3. Load EA in MetaTrader 5: natron_ea.mq5")
    print()


if __name__ == '__main__':
    main()
