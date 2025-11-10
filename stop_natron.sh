#!/bin/bash

# Natron Trading System Stop Script

LOG_DIR="./logs"
PID_FILE="$LOG_DIR/natron.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found. System may not be running."
    exit 1
fi

echo "Stopping Natron system..."

# Read PIDs and kill processes
while read pid; do
    if kill -0 $pid 2>/dev/null; then
        echo "Stopping process $pid..."
        kill $pid
    fi
done < "$PID_FILE"

# Wait for processes to stop
sleep 2

# Force kill if still running
while read pid; do
    if kill -0 $pid 2>/dev/null; then
        echo "Force killing process $pid..."
        kill -9 $pid
    fi
done < "$PID_FILE"

# Remove PID file
rm -f "$PID_FILE"

echo "Natron system stopped."
