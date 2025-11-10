Natron V2 Integration Guide
===========================

This guide connects the Natron Transformer stack with MetaTrader 5 for realtime automated trading.

Prerequisites
-------------
- Linux GPU server (CUDA-enabled) with the Natron repository cloned.
- Python ≥3.10, CUDA-compatible PyTorch (see `requirements.txt`).
- MetaTrader 5 terminal (Windows build ≥3500) with algorithmic trading enabled.
- Network reachability between MT5 terminal and Natron server (default ports 7600/TCP, 8080/HTTP).

1. Train & Export Artifacts
--------------------------
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python train_natron.py --config configs/natron_config.yaml
```

Artifacts produced in `models/`:
- `natron_encoder.pt` – pretrained encoder state.
- `natron_v2.pt` – supervised multi-task checkpoint.
- `scaler.pkl` – feature scaler used during training.
- `feature_columns.json` – deterministic feature ordering.

Optional (RL):
```bash
python train_natron.py --config configs/natron_config.yaml --skip-pretrain --skip-finetune
```
This will call `rl_trainer.py` and generate `natron_policy.zip`.

2. Launch Inference Stack
-------------------------
```bash
./scripts/start_natron.sh configs/natron_config.yaml logs
```
Services started:
- `server_natron.py` – REST `/predict` and TCP JSON bridge.
- `monitor_natron.py` – health checks, latency recorder (`logs/monitor.csv`).

Review logs in `logs/server.log` and `logs/monitor.log`. Containers can be built with:
```bash
docker build -t natron-transformer .
docker run --gpus all -p 8080:8080 -p 7600:7600 natron-transformer
```

3. Configure MetaTrader 5 Expert Advisor
----------------------------------------
1. Copy `natron_ea.mq5` to `MQL5/Experts/Natron/`.
2. Compile inside MetaEditor.
3. Attach the EA to the desired chart/timeframe.
4. Inputs:
   - `InpHost` – Natron server IP.
   - `InpTcpPort` – TCP bridge port (7600).
   - `InpHttpPort` – REST fallback port (8080).
   - `InpUseHttp` – set `true` if socket access blocked; add host to **Tools → Options → Expert Advisors → Allow WebRequest**.
   - Thresholds (`InpBuyThreshold`, `InpSellThreshold`) and risk parameters (`InpLots`, `InpMagic`).
5. The EA pushes the latest 96-bar OHLCV window, parses JSON response, renders overlay via `Comment`, and submits market orders when probabilities exceed thresholds.

4. Data Contract
----------------
Request payload (EA → server):
```json
{
  "candles": [
    {"time":"2024-10-10 00:00:00","open":1.0642,"high":1.0651,"low":1.0630,"close":1.0647,"volume":2150},
    // ... 96 entries ...
  ]
}
```

Response payload (server → EA):
```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_up": 0.69,
  "direction_down": 0.31,
  "regime": "BULL_WEAK",
  "regime_probs": [0.55,0.31,0.04,0.03,0.02,0.05],
  "confidence": 0.82
}
```

5. Visualization & Monitoring
-----------------------------
- `visualization.py` renders offline regime maps and signal heatmaps from CSV prediction logs:
  ```bash
  python visualization.py --predictions logs/predictions.csv --plot regime
  ```
- `monitor_natron.py` writes latency metrics to CSV; integrate with Grafana/Prometheus by tailing the file or forwarding to a metrics gateway.

6. Recovery & Safety
--------------------
- `monitor_natron.py` latency spikes >50 ms trigger log warnings; integrate a webhook by setting `monitoring.alert_webhook` in YAML (extend script to send payload).
- EA auto-reconnects to TCP bridge; when both TCP and HTTP fail, it disables trading until connectivity restored.
- Ensure fail-safe order handling (e.g., add stop-loss/TP logic) before production deployment.

7. Extending the Pipeline
-------------------------
- Deploy gRPC/REST gateway for multiple client EAs.
- Stream predictions to Kafka/Redis for fleet distribution.
- Integrate RL policy (PPO) outputs for continuous action adjustments.
- Use `stable-baselines3` saved policy inside Python for live decision overlays.

Support
-------
For research questions or onboarding, reference `docs/NATRON_WHITEPAPER.md` and the inline docstrings across the codebase.
