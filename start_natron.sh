#!/bin/bash
################################################################################
# Natron Transformer - Production Startup Script
# Starts the Natron API server with proper configuration
################################################################################

set -e

echo "================================================================="
echo "🚀 Starting Natron Transformer Server"
echo "================================================================="

# Check if in correct directory
if [ ! -f "config.yaml" ]; then
    echo "❌ Error: config.yaml not found"
    echo "   Please run this script from the Natron root directory"
    exit 1
fi

# Check if model exists
if [ ! -f "models/natron_v2.pt" ]; then
    echo "❌ Error: Trained model not found at models/natron_v2.pt"
    echo "   Please train the model first: python train_natron.py"
    exit 1
fi

# Check if scaler exists
if [ ! -f "models/scaler.pkl" ]; then
    echo "⚠️  Warning: Scaler not found at models/scaler.pkl"
fi

# Create directories
mkdir -p logs
mkdir -p models

# Activate virtual environment if exists
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Check Python dependencies
echo "🔍 Checking dependencies..."
python -c "import torch, flask, pandas, numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
fi

# Display configuration
echo ""
echo "📋 Configuration:"
echo "   Config: config.yaml"
echo "   Model: models/natron_v2.pt"
echo "   Logs: logs/"
echo ""

# Check if port is already in use
PORT=$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['api']['port'])")
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "⚠️  Port $PORT is already in use"
    echo "   Attempting to kill existing process..."
    lsof -ti:$PORT | xargs kill -9 2>/dev/null || true
    sleep 2
fi

# Start server
echo "🌐 Starting server..."
echo ""

# Option 1: Development mode (with auto-reload)
if [ "$1" == "--dev" ]; then
    echo "🔧 Running in DEVELOPMENT mode"
    python server_natron.py
    
# Option 2: Production mode with Gunicorn
elif [ "$1" == "--prod" ]; then
    echo "🏭 Running in PRODUCTION mode with Gunicorn"
    WORKERS=$(python -c "import multiprocessing; print(multiprocessing.cpu_count() * 2 + 1)")
    gunicorn -w $WORKERS -b 0.0.0.0:$PORT --timeout 120 --access-logfile logs/access.log --error-logfile logs/error.log server_natron:app
    
# Option 3: Background mode
elif [ "$1" == "--background" ]; then
    echo "🔙 Running in BACKGROUND mode"
    nohup python server_natron.py > logs/natron.log 2>&1 &
    PID=$!
    echo $PID > logs/natron.pid
    echo "✅ Server started with PID: $PID"
    echo "   Log file: logs/natron.log"
    echo "   To stop: kill \$(cat logs/natron.pid)"
    
# Default: Standard mode
else
    python server_natron.py
fi
