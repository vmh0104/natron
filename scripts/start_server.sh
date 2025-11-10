#!/bin/bash
#
# Natron Server Startup Script
# Starts the Flask API and Socket server
#

echo "================================================================="
echo "           NATRON AI INFERENCE SERVER - STARTING                "
echo "================================================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
fi

# Check if model exists
if [ ! -f "model/natron_supervised_best.pt" ]; then
    echo "❌ Error: Trained model not found at model/natron_supervised_best.pt"
    echo "   Please train the model first using: python scripts/train_full_pipeline.py"
    exit 1
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0  # Use first GPU

echo ""
echo "🚀 Starting Natron Inference Server..."
echo "   Config: config/config.yaml"
echo "   Model: model/natron_supervised_best.pt"
echo ""
echo "📡 Server will be available at:"
echo "   REST API: http://0.0.0.0:5000"
echo "   Socket: 0.0.0.0:9090"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start server
python src/server/api_server.py

echo ""
echo "Server stopped."
