#!/usr/bin/env bash

set -euo pipefail

MODE=${1:-train}
CONFIG=${2:-configs/natron_config.yaml}
PYTHON=${PYTHON:-python}

case "$MODE" in
  train)
    echo "[Natron] Starting training pipeline with config ${CONFIG}"
    ${PYTHON} train_natron.py --config "${CONFIG}"
    ;;
  infer)
    echo "[Natron] Launching inference server"
    ${PYTHON} server_natron.py --config "${CONFIG}" &
    SERVER_PID=$!
    sleep 2
    echo "[Natron] Starting monitor"
    ${PYTHON} monitor_natron.py --config "${CONFIG}" &
    MONITOR_PID=$!
    trap "echo 'Stopping processes'; kill ${SERVER_PID} ${MONITOR_PID}" SIGINT SIGTERM
    wait ${SERVER_PID} ${MONITOR_PID}
    ;;
  *)
    echo "Usage: ./start_natron.sh [train|infer] [config_path]"
    exit 1
    ;;
esac
