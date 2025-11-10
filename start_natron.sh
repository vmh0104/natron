#!/bin/bash
# Natron Trading System - Startup Script

set -e

echo "🚀 Starting Natron Transformer Trading System..."

# Check if CUDA is available
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  CUDA not detected, will use CPU"
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if data file exists
if [ ! -f "data_export.csv" ]; then
    echo "⚠️  Warning: data_export.csv not found. Please ensure data file is present."
fi

# Create directories
mkdir -p models logs data

# Check if model exists
if [ ! -f "models/natron_v2.pt" ]; then
    echo "⚠️  Warning: Trained model not found. Please train the model first:"
    echo "   python train_pretrain.py  # Phase 1"
    echo "   python train_supervised.py  # Phase 2"
    echo "   python train_rl.py  # Phase 3 (optional)"
fi

# Parse command line arguments
MODE=${1:-api}  # Default to API mode

case $MODE in
    api)
        echo "📡 Starting Flask API server..."
        python api_server.py --host 0.0.0.0 --port 5000
        ;;
    socket)
        echo "🔌 Starting Socket server..."
        python socket_server.py --host 127.0.0.1 --port 8888
        ;;
    train-pretrain)
        echo "🧠 Starting Phase 1: Pretraining..."
        python train_pretrain.py
        ;;
    train-supervised)
        echo "📚 Starting Phase 2: Supervised Fine-tuning..."
        python train_supervised.py
        ;;
    train-rl)
        echo "🎮 Starting Phase 3: Reinforcement Learning..."
        python train_rl.py
        ;;
    train-all)
        echo "🔄 Training all phases..."
        python train_pretrain.py
        python train_supervised.py
        python train_rl.py
        ;;
    *)
        echo "Usage: $0 [api|socket|train-pretrain|train-supervised|train-rl|train-all]"
        exit 1
        ;;
esac
