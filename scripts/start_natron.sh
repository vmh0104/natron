#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-configs/natron_config.yaml}
LOG_DIR=${2:-logs}

mkdir -p "$LOG_DIR"

echo "[Natron] Launching inference server with config: $CONFIG_PATH"
python server_natron.py --config "$CONFIG_PATH" >"$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!

sleep 5
echo "[Natron] Launching monitor"
python monitor_natron.py --config "$CONFIG_PATH" --interval 60 --log-path "$LOG_DIR/monitor.csv" >"$LOG_DIR/monitor.log" 2>&1 &
MONITOR_PID=$!

cleanup() {
  echo "[Natron] Shutting down services"
  kill "$SERVER_PID" "$MONITOR_PID" 2>/dev/null || true
  wait "$SERVER_PID" "$MONITOR_PID" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

wait "$SERVER_PID" "$MONITOR_PID"
