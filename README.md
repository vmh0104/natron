# Natron V2 – Multi-Task Financial Trading Transformer

Natron V2 is an end-to-end, GPU-accelerated trading intelligence platform that combines self-supervised sequence modeling, supervised multi-task learning, and reinforcement learning to deliver actionable buy/sell signals and market regimes from raw OHLCV data.

## Project Layout

- `natron/` – Core Python package (data, model, RL, losses).
- `configs/` – Experiment and deployment YAML files.
- `docs/` – Architecture whitepaper and notes.
- `scripts/` – Runtime helpers (`start_natron.sh`).
- `server_natron.py` – Flask + TCP inference bridge.
- `monitor_natron.py` – Runtime telemetry monitor.
- `rl_train_natron.py` – PPO agent training loop.
- `natron_ea.mq5` – MetaTrader 5 Expert Advisor integration.
- `Dockerfile` – CUDA runtime container specification.

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train_natron.py --config configs/natron_config.yaml
```

### Optional: Reinforcement Learning Policy

```bash
python rl_train_natron.py --config configs/natron_config.yaml --total-steps 200000
```

### Serve Inference + Socket Bridge

```bash
./scripts/start_natron.sh configs/natron_config.yaml
# or
python server_natron.py --config configs/natron_config.yaml
```

### Monitor Runtime Health

```bash
python monitor_natron.py --host http://127.0.0.1 --port 8081
```

### MetaTrader 5 Deployment

1. Copy `natron_ea.mq5` into your MT5 `MQL5/Experts` folder.
2. Compile inside MetaEditor.
3. Attach the EA to a chart and configure host/port thresholds.
4. Ensure the Natron Python server is running before enabling auto-trading.

## Status

Development in progress. Refer to `docs/whitepaper.md` for detailed architecture, learning strategies, and deployment notes.
