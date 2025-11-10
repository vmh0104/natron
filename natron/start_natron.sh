#!/bin/bash
# Natron Startup Script
# Auto-start with restart-on-failure

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

LOG_FILE="$LOG_DIR/natron_$(date +%Y%m%d_%H%M%S).log"
PID_FILE="./natron.pid"

# Function to cleanup on exit
cleanup() {
    echo "Shutting down Natron..."
    if [ -f "$PID_FILE" ]; then
        kill $(cat "$PID_FILE") 2>/dev/null || true
        rm -f "$PID_FILE"
    fi
    exit 0
}

trap cleanup SIGINT SIGTERM

# Function to start Natron server
start_server() {
    echo "Starting Natron Server..."
    python natron_server_v5.py realtime_config.yaml >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Natron Server started with PID $(cat $PID_FILE)"
}

# Function to check if server is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            return 0
        else
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

# Main loop with restart-on-failure
echo "Natron V1.0 - Galaxy-class"
echo "Starting with auto-restart enabled..."
echo "Logs: $LOG_FILE"

while true; do
    if ! is_running; then
        start_server
    fi
    
    # Wait a bit before checking again
    sleep 10
    
    # Check if process is still running
    if ! is_running; then
        echo "Server crashed, restarting in 5 seconds..."
        sleep 5
    fi
done
