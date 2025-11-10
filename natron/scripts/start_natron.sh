#!/bin/bash
# Natron Startup Script
# Auto-start with restart-on-failure

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Configuration
CONFIG_FILE="${CONFIG_FILE:-configs/realtime_config.yaml}"
LOG_DIR="${LOG_DIR:-logs}"
MAX_RESTARTS=10
RESTART_DELAY=5

# Create log directory
mkdir -p "$LOG_DIR"

# Log file
LOG_FILE="$LOG_DIR/natron_server.log"
ERROR_LOG="$LOG_DIR/natron_errors.log"

# Function to log messages
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to start server
start_server() {
    log "Starting Natron Server..."
    log "Config: $CONFIG_FILE"
    log "Version: Natron V1.0 - Galaxy-class"
    
    python -m src.realtime.natron_server_v5 --config "$CONFIG_FILE" 2>&1 | tee -a "$LOG_FILE"
}

# Main loop with restart logic
restart_count=0
while [ $restart_count -lt $MAX_RESTARTS ]; do
    log "=========================================="
    log "Natron Server - Attempt $((restart_count + 1))/$MAX_RESTARTS"
    log "=========================================="
    
    if start_server; then
        log "Server exited normally"
        break
    else
        exit_code=$?
        log "Server crashed with exit code $exit_code" | tee -a "$ERROR_LOG"
        restart_count=$((restart_count + 1))
        
        if [ $restart_count -lt $MAX_RESTARTS ]; then
            log "Restarting in $RESTART_DELAY seconds..."
            sleep $RESTART_DELAY
        else
            log "Max restarts reached. Exiting."
            exit 1
        fi
    fi
done

log "Natron Server stopped"
