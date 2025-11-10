#!/usr/bin/env bash
# Launches the Natron inference server and optional monitor process.

set -euo pipefail

CONFIG_PATH=${1:-configs/natron_config.yaml}
LOG_DIR=${LOG_DIR:-logs}
MONITOR=${MONITOR:-true}

mkdir -p "$LOG_DIR"

export NATRON_CONFIG="$CONFIG_PATH"

read_host_port() {
  python3 - <<'PY'
import os
import sys
import yaml

config_path = os.environ.get("NATRON_CONFIG", "configs/natron_config.yaml")
with open(config_path, "r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh)
server_cfg = cfg.get("server", {})
print(server_cfg.get("host", "0.0.0.0"))
print(server_cfg.get("port", 8000))
PY
}

HOST_PORT=($(read_host_port))
HOST=${HOST_PORT[0]}
PORT=${HOST_PORT[1]}

echo "Starting Natron server with config $CONFIG_PATH (host=$HOST, port=$PORT)"
python3 server_natron.py >"$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "Natron server PID: $SERVER_PID"

if [[ "$MONITOR" == "true" ]]; then
  echo "Starting monitor..."
  python3 monitor_natron.py --host "$HOST" --port "$PORT" >"$LOG_DIR/monitor.log" 2>&1 &
  MONITOR_PID=$!
  echo "Monitor PID: $MONITOR_PID"
fi

trap 'echo "Stopping..." ; kill $SERVER_PID >/dev/null 2>&1; [[ -n "${MONITOR_PID:-}" ]] && kill $MONITOR_PID >/dev/null 2>&1; wait' INT TERM

wait $SERVER_PID
