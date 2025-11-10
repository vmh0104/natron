#!/bin/bash
# Natron Transformer - Startup Script
# Comprehensive startup for all services

set -e

echo "=========================================="
echo "🧠 NATRON TRANSFORMER STARTUP"
echo "=========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo -e "${YELLOW}⚠️  Running as root. Consider using a non-root user.${NC}"
fi

# Check CUDA availability
echo ""
echo "🔍 Checking CUDA..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo -e "${GREEN}✅ CUDA available${NC}"
else
    echo -e "${YELLOW}⚠️  CUDA not detected. Will use CPU.${NC}"
fi

# Create directories
echo ""
echo "📁 Creating directories..."
mkdir -p data models checkpoints logs cache runs
echo -e "${GREEN}✅ Directories created${NC}"

# Check Python version
echo ""
echo "🐍 Checking Python..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python version: $PYTHON_VERSION"

REQUIRED_VERSION="3.10"
if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo -e "${RED}❌ Python 3.10+ required${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Python version OK${NC}"

# Check dependencies
echo ""
echo "📦 Checking dependencies..."
if ! python3 -c "import torch" &> /dev/null; then
    echo -e "${YELLOW}⚠️  PyTorch not found. Installing dependencies...${NC}"
    pip3 install -r requirements.txt
fi
echo -e "${GREEN}✅ Dependencies OK${NC}"

# Parse arguments
MODE="all"
CONFIG="config.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo ""
echo "⚙️  Configuration:"
echo "   Mode: $MODE"
echo "   Config: $CONFIG"

# Start services based on mode
case $MODE in
    train)
        echo ""
        echo "🔥 Starting training..."
        python3 src/train_natron.py --config $CONFIG --phase all
        ;;
    
    api)
        echo ""
        echo "🌐 Starting API server..."
        python3 api/api_server.py --config $CONFIG &
        API_PID=$!
        
        echo "   API Server PID: $API_PID"
        echo "   API URL: http://localhost:5000"
        echo ""
        echo "Press Ctrl+C to stop"
        
        trap "kill $API_PID" EXIT
        wait $API_PID
        ;;
    
    socket)
        echo ""
        echo "🔌 Starting socket server..."
        python3 api/socket_server.py --config $CONFIG &
        SOCKET_PID=$!
        
        echo "   Socket Server PID: $SOCKET_PID"
        echo "   Socket: localhost:9090"
        echo ""
        echo "Press Ctrl+C to stop"
        
        trap "kill $SOCKET_PID" EXIT
        wait $SOCKET_PID
        ;;
    
    all)
        echo ""
        echo "🚀 Starting all services..."
        
        # Start API server
        echo "   Starting API server..."
        python3 api/api_server.py --config $CONFIG &
        API_PID=$!
        sleep 3
        
        # Start socket server
        echo "   Starting socket server..."
        python3 api/socket_server.py --config $CONFIG &
        SOCKET_PID=$!
        sleep 3
        
        # Start TensorBoard
        echo "   Starting TensorBoard..."
        tensorboard --logdir=runs --host=0.0.0.0 --port=6006 &
        TB_PID=$!
        
        echo ""
        echo -e "${GREEN}✅ All services started${NC}"
        echo ""
        echo "📡 Service endpoints:"
        echo "   API Server: http://localhost:5000"
        echo "   Socket Server: localhost:9090"
        echo "   TensorBoard: http://localhost:6006"
        echo ""
        echo "Press Ctrl+C to stop all services"
        
        # Trap to kill all on exit
        trap "kill $API_PID $SOCKET_PID $TB_PID 2>/dev/null" EXIT
        
        # Wait for any process to exit
        wait -n
        ;;
    
    docker)
        echo ""
        echo "🐳 Starting Docker services..."
        docker-compose -f deployment/docker-compose.yml up -d
        
        echo ""
        echo -e "${GREEN}✅ Docker services started${NC}"
        echo ""
        echo "To view logs:"
        echo "   docker-compose -f deployment/docker-compose.yml logs -f"
        echo ""
        echo "To stop:"
        echo "   docker-compose -f deployment/docker-compose.yml down"
        ;;
    
    *)
        echo -e "${RED}❌ Unknown mode: $MODE${NC}"
        echo ""
        echo "Available modes:"
        echo "   train   - Run training only"
        echo "   api     - Run API server only"
        echo "   socket  - Run socket server only"
        echo "   all     - Run all services"
        echo "   docker  - Run with Docker Compose"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "🎉 NATRON TRANSFORMER READY"
echo "=========================================="
