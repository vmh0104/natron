#!/bin/bash
# Natron Transformer Training Script

echo "=========================================="
echo "Natron Transformer - Training Pipeline"
echo "=========================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if data exists
if [ ! -f "data_export.csv" ]; then
    echo "⚠ Warning: data_export.csv not found"
    echo "Generating sample data..."
    python3 create_sample_data.py
fi

# Check CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Create model directory
mkdir -p model

# Run training
echo ""
echo "Starting training..."
python3 train.py

echo ""
echo "Training complete!"
echo "Model saved to: model/natron_v2.pt"
