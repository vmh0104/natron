#!/bin/bash
# Quick Start Script for Natron Transformer

set -e

echo "=========================================="
echo "Natron Transformer - Quick Start"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
python3 --version

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

# Generate sample data if data_export.csv doesn't exist
if [ ! -f "data_export.csv" ]; then
    echo ""
    echo "Generating sample data..."
    python3 generate_sample_data.py --num_candles 2000 --output data_export.csv
fi

# Test pipeline components
echo ""
echo "Testing pipeline components..."
python3 test_pipeline.py

# Create models directory
mkdir -p models

echo ""
echo "=========================================="
echo "Quick Start Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Review config.yaml and adjust if needed"
echo "2. Run training: python3 train.py --data data_export.csv"
echo "3. Start API server: python3 api_server.py"
echo "4. Start MQL5 socket server: python3 mql5_socket_server.py"
echo ""
