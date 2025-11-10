"""
Natron Transformer - Main Training Script
Orchestrates all three training phases
"""

import torch
import yaml
import argparse
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from model_natron import create_model, save_checkpoint
from dataset_loader import NatronDataLoader
from pretrain import run_pretraining
from supervised import run_supervised_training
from reinforcement import run_reinforcement_learning


def setup_logging(config: dict):
    """Setup logging directory"""
    logs_dir = config['paths']['logs_dir']
    os.makedirs(logs_dir, exist_ok=True)
    
    # Setup other directories
    for path_key in ['model_dir', 'checkpoint_dir', 'cache_dir']:
        os.makedirs(config['paths'][path_key], exist_ok=True)


def train_phase_all(config_path: str, csv_path: str = None):
    """
    Run all training phases sequentially
    
    Args:
        config_path: Path to config file
        csv_path: Path to CSV data (optional)
    """
    print("=" * 80)
    print("🧠 NATRON TRANSFORMER - END-TO-END TRAINING")
    print("=" * 80)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    setup_logging(config)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n🖥️  Device: {device}")
    
    if device == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # === STEP 1: Data Loading ===
    print("\n" + "=" * 80)
    print("📂 STEP 1: DATA LOADING & PREPROCESSING")
    print("=" * 80)
    
    loader = NatronDataLoader(config_path)
    dataloaders = loader.load_and_prepare(csv_path)
    
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    pretrain_loader = dataloaders['pretrain']
    
    # === STEP 2: Model Creation ===
    print("\n" + "=" * 80)
    print("🏗️  STEP 2: MODEL CREATION")
    print("=" * 80)
    
    model = create_model(config, device)
    
    # === STEP 3: Phase 1 - Pretraining ===
    if config['training']['pretrain']['enabled']:
        print("\n" + "=" * 80)
        print("🔥 STEP 3: PHASE 1 - UNSUPERVISED PRETRAINING")
        print("=" * 80)
        
        model = run_pretraining(model, pretrain_loader, config, device)
        
        # Save pretrained model
        model_path = os.path.join(config['paths']['model_dir'], 'natron_pretrained.pt')
        torch.save(model.state_dict(), model_path)
        print(f"\n💾 Pretrained model saved: {model_path}")
    
    # === STEP 4: Phase 2 - Supervised Training ===
    print("\n" + "=" * 80)
    print("🎯 STEP 4: PHASE 2 - SUPERVISED MULTI-TASK TRAINING")
    print("=" * 80)
    
    model = run_supervised_training(model, train_loader, val_loader, config, device)
    
    # Save supervised model
    model_path = os.path.join(config['paths']['model_dir'], 'natron_supervised.pt')
    torch.save(model.state_dict(), model_path)
    print(f"\n💾 Supervised model saved: {model_path}")
    
    # === STEP 5: Phase 3 - Reinforcement Learning ===
    if config['training']['reinforcement']['enabled']:
        print("\n" + "=" * 80)
        print("🚀 STEP 5: PHASE 3 - REINFORCEMENT LEARNING")
        print("=" * 80)
        
        # Get sequences and prices for RL
        sequences = loader.features.values[:-config['data']['sequence_length']]
        prices = loader.df['close'].values[config['data']['sequence_length']:]
        
        # Create sequences
        n_samples = len(sequences) - config['data']['sequence_length']
        seq_array = []
        price_array = []
        
        for i in range(n_samples):
            seq_array.append(sequences[i:i+config['data']['sequence_length']])
            price_array.append(prices[i+config['data']['sequence_length']-1])
        
        seq_array = np.array(seq_array)
        price_array = np.array(price_array)
        
        model = run_reinforcement_learning(model, seq_array, price_array, config, device)
        
        # Save RL model
        model_path = os.path.join(config['paths']['model_dir'], 'natron_rl.pt')
        torch.save(model.state_dict(), model_path)
        print(f"\n💾 RL model saved: {model_path}")
    
    # === STEP 6: Final Model ===
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE")
    print("=" * 80)
    
    final_model_path = os.path.join(config['paths']['model_dir'], 'natron_v2.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_model_path)
    print(f"\n🎉 Final model saved: {final_model_path}")
    
    # === STEP 7: Evaluation ===
    print("\n" + "=" * 80)
    print("📊 FINAL EVALUATION ON TEST SET")
    print("=" * 80)
    
    evaluate_model(model, test_loader, config, device)
    
    print("\n" + "=" * 80)
    print("🎊 ALL PHASES COMPLETE!")
    print("=" * 80)


def evaluate_model(model, test_loader, config, device):
    """Evaluate model on test set"""
    from supervised import SupervisedTrainer
    from losses import compute_accuracy
    
    model.eval()
    
    total_buy_acc = 0
    total_sell_acc = 0
    total_dir_acc = 0
    total_reg_acc = 0
    num_batches = 0
    
    print("\n🧪 Running evaluation...")
    
    with torch.no_grad():
        for batch in test_loader:
            sequences, buy_labels, sell_labels, direction_labels, regime_labels = batch
            
            sequences = sequences.to(device)
            buy_labels = buy_labels.to(device)
            sell_labels = sell_labels.to(device)
            direction_labels = direction_labels.to(device)
            regime_labels = regime_labels.to(device)
            
            predictions = model(sequences)
            
            total_buy_acc += compute_accuracy(predictions['buy'], buy_labels)
            total_sell_acc += compute_accuracy(predictions['sell'], sell_labels)
            total_dir_acc += compute_accuracy(predictions['direction'], direction_labels)
            total_reg_acc += compute_accuracy(predictions['regime'], regime_labels)
            
            num_batches += 1
    
    print("\n📈 Test Set Results:")
    print(f"  Buy Accuracy: {total_buy_acc / num_batches:.4f}")
    print(f"  Sell Accuracy: {total_sell_acc / num_batches:.4f}")
    print(f"  Direction Accuracy: {total_dir_acc / num_batches:.4f}")
    print(f"  Regime Accuracy: {total_reg_acc / num_batches:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Train Natron Transformer')
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--phase',
        type=str,
        choices=['pretrain', 'supervised', 'reinforcement', 'all'],
        default='all',
        help='Training phase to run'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to CSV data file (overrides config)'
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    setup_logging(config)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load data
    loader = NatronDataLoader(args.config)
    dataloaders = loader.load_and_prepare(args.data)
    
    # Create model
    model = create_model(config, device)
    
    # Load checkpoint if provided
    if args.checkpoint:
        print(f"📂 Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Run specified phase
    if args.phase == 'pretrain' or args.phase == 'all':
        if config['training']['pretrain']['enabled']:
            print("\n🔥 Running Pretraining...")
            model = run_pretraining(model, dataloaders['pretrain'], config, device)
            
            model_path = os.path.join(config['paths']['model_dir'], 'natron_pretrained.pt')
            torch.save(model.state_dict(), model_path)
            print(f"💾 Model saved: {model_path}")
    
    if args.phase == 'supervised' or args.phase == 'all':
        print("\n🎯 Running Supervised Training...")
        model = run_supervised_training(
            model, dataloaders['train'], dataloaders['val'], config, device
        )
        
        model_path = os.path.join(config['paths']['model_dir'], 'natron_supervised.pt')
        torch.save(model.state_dict(), model_path)
        print(f"💾 Model saved: {model_path}")
    
    if args.phase == 'reinforcement' or args.phase == 'all':
        if config['training']['reinforcement']['enabled']:
            print("\n🚀 Running Reinforcement Learning...")
            
            # Prepare RL data
            sequences = loader.features.values
            prices = loader.df['close'].values
            
            model = run_reinforcement_learning(model, sequences, prices, config, device)
            
            model_path = os.path.join(config['paths']['model_dir'], 'natron_rl.pt')
            torch.save(model.state_dict(), model_path)
            print(f"💾 Model saved: {model_path}")
    
    # Save final model
    final_model_path = os.path.join(config['paths']['model_dir'], 'natron_v2.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config
    }, final_model_path)
    print(f"\n🎉 Final model saved: {final_model_path}")
    
    # Evaluate
    print("\n📊 Evaluating on test set...")
    evaluate_model(model, dataloaders['test'], config, device)
    
    print("\n✅ Training complete!")


if __name__ == "__main__":
    main()
