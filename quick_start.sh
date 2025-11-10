#!/bin/bash
# Natron Quick Start Script

set -e

echo "🧠 Natron Transformer - Quick Start"
echo "===================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check CUDA availability
echo "Checking CUDA..."
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')" || echo "PyTorch not installed"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Create directories
mkdir -p model
mkdir -p data

# Check if data file exists
if [ ! -f "data_export.csv" ]; then
    echo ""
    echo "⚠️  Warning: data_export.csv not found!"
    echo "Please place your OHLCV data file in the workspace root."
    echo "Expected format: time,open,high,low,close,volume"
    exit 1
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Train the model:"
echo "   python natron/train_pipeline.py --data data_export.csv --phase both"
echo ""
echo "2. Start API server:"
echo "   python natron/api_server.py"
echo ""
echo "3. Or start socket server (for MQL5):"
echo "   python natron/socket_server.py --port 8888"
