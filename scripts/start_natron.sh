#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH=${1:-configs/natron_config.yaml}

echo "[Natron] Activating virtual environment..."
if [ -d ".venv" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
else
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
fi

echo "[Natron] Launching inference server..."
exec python server_natron.py --config "${CONFIG_PATH}"
