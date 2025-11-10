#!/bin/bash
################################################################################
# Natron Transformer - Stop Script
# Gracefully stops the Natron server
################################################################################

echo "⏹️  Stopping Natron server..."

# Check PID file
if [ -f "logs/natron.pid" ]; then
    PID=$(cat logs/natron.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "🔴 Killing process $PID..."
        kill -TERM $PID
        sleep 2
        
        # Force kill if still running
        if ps -p $PID > /dev/null 2>&1; then
            echo "⚠️  Process still running, force killing..."
            kill -9 $PID
        fi
        
        rm -f logs/natron.pid
        echo "✅ Server stopped"
    else
        echo "⚠️  Process $PID not found"
        rm -f logs/natron.pid
    fi
else
    echo "⚠️  PID file not found, searching for process..."
    
    # Find by port
    PORT=$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['api']['port'])" 2>/dev/null || echo "8888")
    
    PIDS=$(lsof -ti:$PORT 2>/dev/null)
    if [ -n "$PIDS" ]; then
        echo "🔴 Found processes on port $PORT: $PIDS"
        echo $PIDS | xargs kill -9
        echo "✅ Processes killed"
    else
        echo "ℹ️  No Natron server processes found"
    fi
fi

echo "Done."
