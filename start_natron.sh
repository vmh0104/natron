#!/bin/bash

# Natron Transformer Startup Script
# Starts the inference server for MQL5 integration

set -e

echo "🧠 Starting Natron Transformer Inference Server..."

# Check if model exists
if [ ! -f "model/natron_v2.pt" ]; then
    echo "❌ Error: Model file not found at model/natron_v2.pt"
    echo "   Please train the model first: python train_natron.py"
    exit 1
fi

# Check if scaler exists
if [ ! -f "model/scaler.pkl" ]; then
    echo "❌ Error: Scaler file not found at model/scaler.pkl"
    echo "   Please train the model first: python train_natron.py"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found"
    exit 1
fi

# Check CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Start server
echo "🚀 Starting server..."
python3 server_natron.py \
    --config config.yaml \
    --model model/natron_v2.pt \
    --flask-port 5000 \
    --socket-port 8888 \
    --socket-host 0.0.0.0
