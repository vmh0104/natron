#!/bin/bash
#
# Natron V2 Startup Script
# Starts the socket server for real-time trading
#

set -e

echo "=========================================="
echo "  NATRON V2 - AI TRADING SYSTEM"
echo "=========================================="
echo ""

# Configuration
CONFIG_PATH="${CONFIG_PATH:-config/config.yaml}"
MODEL_PATH="${MODEL_PATH:-model/natron_v2.pt}"
SERVER_TYPE="${SERVER_TYPE:-socket}"  # socket or flask

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo "Please train the model first using: python train_natron.py"
    exit 1
fi

# Check if config exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "ERROR: Config not found at $CONFIG_PATH"
    exit 1
fi

echo "✓ Model found: $MODEL_PATH"
echo "✓ Config found: $CONFIG_PATH"
echo ""

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
    echo ""
else
    echo "⚠ GPU not detected - running on CPU"
    echo ""
fi

# Start appropriate server
if [ "$SERVER_TYPE" = "flask" ]; then
    echo "Starting Flask API server..."
    python src/inference/flask_api.py --config "$CONFIG_PATH" --model "$MODEL_PATH"
else
    echo "Starting Socket server for MQL5 communication..."
    python src/inference/socket_server.py --config "$CONFIG_PATH" --model "$MODEL_PATH"
fi
