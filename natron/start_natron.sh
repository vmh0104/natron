#!/bin/bash
# Natron AI Trading System - Startup Script
# Version: Natron V1.0 - Galaxy-class

set -e

echo "=========================================="
echo "  NATRON AI TRADING SYSTEM"
echo "  Version: V1.0 - Galaxy-class"
echo "=========================================="
echo ""

# Configuration
NATRON_DIR="/app"
LOG_DIR="${NATRON_DIR}/logs"
PID_FILE="${LOG_DIR}/natron.pid"
LOG_FILE="${LOG_DIR}/natron.log"
MAX_RESTARTS=10
RESTART_DELAY=5

# Create log directory
mkdir -p "${LOG_DIR}"

# Function to start Natron server
start_natron() {
    echo "[$(date)] Starting Natron Server..."
    cd "${NATRON_DIR}"
    
    # Start server in background
    nohup python natron_server_v5.py \
        --config realtime_config.yaml \
        >> "${LOG_FILE}" 2>&1 &
    
    echo $! > "${PID_FILE}"
    echo "[$(date)] Natron Server started with PID: $(cat ${PID_FILE})"
}

# Function to stop Natron server
stop_natron() {
    if [ -f "${PID_FILE}" ]; then
        PID=$(cat "${PID_FILE}")
        if ps -p "${PID}" > /dev/null 2>&1; then
            echo "[$(date)] Stopping Natron Server (PID: ${PID})..."
            kill "${PID}"
            wait "${PID}" 2>/dev/null || true
            echo "[$(date)] Natron Server stopped"
        fi
        rm -f "${PID_FILE}"
    fi
}

# Function to check if server is running
is_running() {
    if [ -f "${PID_FILE}" ]; then
        PID=$(cat "${PID_FILE}")
        if ps -p "${PID}" > /dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# Function to restart server
restart_natron() {
    echo "[$(date)] Restarting Natron Server..."
    stop_natron
    sleep "${RESTART_DELAY}"
    start_natron
}

# Trap signals for graceful shutdown
trap 'stop_natron; exit 0' SIGTERM SIGINT

# Main loop with restart-on-failure
restart_count=0

while true; do
    if ! is_running; then
        if [ "${restart_count}" -lt "${MAX_RESTARTS}" ]; then
            restart_count=$((restart_count + 1))
            echo "[$(date)] Restart attempt ${restart_count}/${MAX_RESTARTS}"
            start_natron
        else
            echo "[$(date)] Maximum restart attempts reached. Exiting."
            exit 1
        fi
    fi
    
    # Wait and check again
    sleep 10
    
    # Check if process is still running
    if ! is_running; then
        echo "[$(date)] Process died, will restart..."
    fi
done
