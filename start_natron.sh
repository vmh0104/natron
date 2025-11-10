#!/bin/bash

# Natron Trading System Startup Script
# Starts API server, socket server, and monitoring

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
MODEL_PATH="./models/natron_v2.pt"
API_PORT=5000
SOCKET_PORT=8888
LOG_DIR="./logs"

# Create directories
mkdir -p models logs

echo -e "${GREEN}Starting Natron Trading System...${NC}"

# Check if model exists
if [ ! -f "$MODEL_PATH" ]; then
    echo -e "${YELLOW}Warning: Model file not found at $MODEL_PATH${NC}"
    echo "Please train the model first using: python train_natron.py"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import torch, flask, numpy, pandas" 2>/dev/null || {
    echo -e "${RED}Missing dependencies. Installing...${NC}"
    pip install -r requirements.txt
}

# Start API server in background
echo -e "${GREEN}Starting API server on port $API_PORT...${NC}"
python3 api_server.py --model "$MODEL_PATH" --port $API_PORT > "$LOG_DIR/api_server.log" 2>&1 &
API_PID=$!
echo "API Server PID: $API_PID"

# Wait for API server to start
sleep 3

# Check if API server is running
if ! kill -0 $API_PID 2>/dev/null; then
    echo -e "${RED}API server failed to start. Check $LOG_DIR/api_server.log${NC}"
    exit 1
fi

# Start socket server in background
echo -e "${GREEN}Starting socket server on port $SOCKET_PORT...${NC}"
python3 socket_server.py --host localhost --port $SOCKET_PORT --model "$MODEL_PATH" > "$LOG_DIR/socket_server.log" 2>&1 &
SOCKET_PID=$!
echo "Socket Server PID: $SOCKET_PID"

# Wait for socket server to start
sleep 3

# Check if socket server is running
if ! kill -0 $SOCKET_PID 2>/dev/null; then
    echo -e "${RED}Socket server failed to start. Check $LOG_DIR/socket_server.log${NC}"
    kill $API_PID 2>/dev/null
    exit 1
fi

# Start monitoring in background
echo -e "${GREEN}Starting monitoring...${NC}"
python3 monitor_natron.py --interval 60 > "$LOG_DIR/monitor.log" 2>&1 &
MONITOR_PID=$!
echo "Monitor PID: $MONITOR_PID"

# Save PIDs to file
echo $API_PID > "$LOG_DIR/natron.pid"
echo $SOCKET_PID >> "$LOG_DIR/natron.pid"
echo $MONITOR_PID >> "$LOG_DIR/natron.pid"

echo -e "${GREEN}Natron system started successfully!${NC}"
echo ""
echo "Services:"
echo "  - API Server: http://localhost:$API_PORT"
echo "  - Socket Server: localhost:$SOCKET_PORT"
echo "  - Monitor: Running"
echo ""
echo "Logs:"
echo "  - API: $LOG_DIR/api_server.log"
echo "  - Socket: $LOG_DIR/socket_server.log"
echo "  - Monitor: $LOG_DIR/monitor.log"
echo ""
echo "To stop: ./stop_natron.sh"
echo "To check status: python3 monitor_natron.py --once"

# Keep script running
wait
