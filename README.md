# Natron V2 – End-to-End Multi-Task Financial Trading System

Natron V2 is a production-grade trading intelligence stack that learns market structure, supervised signals, and reinforcement policies from high-frequency OHLCV data. The system delivers multi-task predictions (buy, sell, direction, regime) and exposes them via a low-latency GPU inference server with MetaTrader 5 (MQL5) integration.

## Key Components
- **Phase 1 – Structure Understanding:** Masked-sequence reconstruction and contrastive pretraining over 96-candle windows to learn robust market embeddings.
- **Phase 2 – Signal Recognition:** Multi-head Transformer fine-tuned to predict buy/sell probabilities, directional movement, and market regime classification simultaneously.
- **Phase 3 – Behavioral Adaptation:** Optional PPO training loop that optimizes a risk-aware reward (`profit - α * turnover - β * drawdown`) on simulated order executions.
- **Realtime Execution:** Asynchronous TCP socket server serving JSON signals, monitored health checks, and MQL5 Expert Advisor for automated order routing.

## Repository Layout
- `dataset_loader.py` – Feature engineering, labeling logic, and sequence dataset builders.
- `model_natron.py` – Transformer encoder/decoder blocks, multi-task heads, and projection layers for pretraining.
- `losses.py` – Composite losses for masked modeling, contrastive learning, and multi-task supervised training.
- `train_natron.py` – End-to-end trainer covering pretraining, fine-tuning, and reinforcement phases.
- `reinforcement.py` – PPO agent and trading environment abstractions.
- `feature_engine.py` – Technical feature generation (~100 engineered features).
- `labeling.py` – Rule-based buy/sell/direction/regime labels aligned with institutional heuristics.
- `utils/` – Shared utilities: metrics, logging, checkpoints.
- `server_natron.py` – GPU-backed inference and TCP relay for MetaTrader.
- `monitor_natron.py` – Runtime monitor and alert hooks.
- `start_natron.sh` – Bootstrap script (training or inference modes).
- `Dockerfile` – CUDA-ready container definition for deployment.
- `natron_ea.mq5` – MetaTrader 5 Expert Advisor for socket communication and chart overlays.
- `visualize_natron.py` – Matplotlib/Seaborn dashboard generator for prediction logs.
- `whitepaper.md` – System architecture and research notes curated by Sonnet 4.5.
- `configs/natron_config.yaml` – YAML hyperparameters and workflow toggles.

## Getting Started
1. **Install Requirements**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Prepare Data**  
   Place `data_export.csv` inside `data/`. The pipeline expects columns: `time, open, high, low, close, volume`.

3. **Run Pretraining + Supervised Training**
   ```bash
   python train_natron.py --config configs/natron_config.yaml
   ```
   Use `--phase` to limit to `pretrain`, `supervised`, or `rl`.

4. **Launch Inference Server**
   ```bash
   python server_natron.py --config configs/natron_config.yaml
   ```

5. **Start Monitoring & EA**
   ```bash
   python monitor_natron.py --config configs/natron_config.yaml &
   # Load natron_ea.mq5 inside MetaTrader 5 terminal
   ```

## Realtime Execution
- Launch inference + monitor in one command:
  ```bash
  ./start_natron.sh infer configs/natron_config.yaml
  ```
- REST endpoint: `POST http://<host>:8080/predict` with payload `{"candles": [...]}`
- TCP socket: connect to `<host>:8765` and stream newline-delimited JSON.
- Metadata used by the server is stored in `outputs/data_metadata.json` and automatically generated after training.

## MetaTrader 5 Integration
- Copy `natron_ea.mq5` into your `MQL5/Experts` directory and compile.
- In MetaTrader 5, allow WebRequest access to `http://<natron_host>:8080` via **Tools → Options → Expert Advisors**.
- Attach the EA to a chart; it will pull the latest `SequenceLength` candles, call the Natron API, display probabilities, and submit market orders when thresholds are met. Tune thresholds with the EA inputs panel.

## Visualization
- Generate heatmaps and probability plots from a CSV of predictions:
  ```bash
  python visualize_natron.py --predictions data/predictions.csv --output figures/natron_signals.png
  ```
- CSV must include columns `time,buy_prob,sell_prob,direction_up,regime,confidence`.

## Development Notes
- Built for **Python 3.10+** and **PyTorch 2.x (CUDA)**.
- Modular design allows custom feature sets, label strategies, and RL objectives.
- Logging via JSONL and TensorBoard.
- Checkpoints saved to `model/` (default `model/natron_v2.pt`).

## License
Proprietary – internal research use only unless granted explicit permission.
