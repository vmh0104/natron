#!/bin/bash
# Natron Startup Script
# Auto-start with restart-on-failure

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
CONFIG_PATH="${CONFIG_PATH:-configs/realtime_config.yaml}"
LOG_DIR="${LOG_DIR:-logs}"
MAX_RESTARTS=10
RESTART_DELAY=5

# Create log directory
mkdir -p "$LOG_DIR"

# Log file
LOG_FILE="$LOG_DIR/natron_server.log"
ERROR_LOG="$LOG_DIR/natron_server_error.log"

# Function to start server
start_server() {
    echo "$(date): Starting Natron Server V5..." | tee -a "$LOG_FILE"
    python -m src.realtime.natron_server_v5 "$CONFIG_PATH" 2>&1 | tee -a "$LOG_FILE"
}

# Function to check if server is running
check_server() {
    # Check if port is listening (simplified check)
    netstat -tuln 2>/dev/null | grep -q ":8888" || return 1
    return 0
}

# Main loop with restart logic
RESTART_COUNT=0
while [ $RESTART_COUNT -lt $MAX_RESTARTS ]; do
    echo "$(date): Attempt $((RESTART_COUNT + 1))/$MAX_RESTARTS" | tee -a "$LOG_FILE"
    
    # Start server
    if start_server; then
        echo "$(date): Server exited normally" | tee -a "$LOG_FILE"
        break
    else
        EXIT_CODE=$?
        echo "$(date): Server crashed with exit code $EXIT_CODE" | tee -a "$ERROR_LOG"
        RESTART_COUNT=$((RESTART_COUNT + 1))
        
        if [ $RESTART_COUNT -lt $MAX_RESTARTS ]; then
            echo "$(date): Restarting in $RESTART_DELAY seconds..." | tee -a "$LOG_FILE"
            sleep $RESTART_DELAY
        else
            echo "$(date): Max restarts reached. Exiting." | tee -a "$ERROR_LOG"
            exit 1
        fi
    fi
done

# If we get here, server stopped
echo "$(date): Natron Server stopped" | tee -a "$LOG_FILE"
