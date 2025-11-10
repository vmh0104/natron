#!/bin/bash
# Natron Startup Script

set -e

echo "🚀 Starting Natron Transformer Trading System..."

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ CUDA detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "⚠️  CUDA not detected, using CPU"
fi

# Check Python version
python3 --version

# Check if model exists
MODEL_PATH="./checkpoints/natron_v2.pt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "⚠️  Model not found at $MODEL_PATH"
    echo "   Please train a model first using:"
    echo "   python3 train_supervised.py --data_path data_export.csv"
    exit 1
fi

# Start API server (background)
echo "📡 Starting Flask API server..."
python3 api_server.py --model_path "$MODEL_PATH" --host 0.0.0.0 --port 5000 &
API_PID=$!
echo "   API server PID: $API_PID"

# Wait a moment
sleep 2

# Start Socket server (background)
echo "🔌 Starting Socket server for MQL5..."
python3 socket_server.py --model_path "$MODEL_PATH" --host localhost --port 8888 &
SOCKET_PID=$!
echo "   Socket server PID: $SOCKET_PID"

# Wait for servers to start
sleep 3

# Health check
echo "🏥 Health check..."
curl -s http://localhost:5000/health || echo "   API server not responding"

echo ""
echo "✅ Natron system started!"
echo "   - API Server: http://localhost:5000"
echo "   - Socket Server: localhost:8888"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for interrupt
trap "echo '🛑 Stopping servers...'; kill $API_PID $SOCKET_PID 2>/dev/null; exit" INT TERM

wait
