## Natron Transformer V2

Natron is a multi-phase GPU-first trading intelligence stack that learns market structure, supervised signals, and reinforcement policies end-to-end from OHLCV sequences. The system ships with a production-ready data pipeline, Transformer backbone, pretraining + fine-tuning loops, PPO reinforcement adapter, realtime inference bridge, monitoring utilities, and an MQL5 Expert Advisor.

### Repository Layout
- `configs/natron_config.yaml` – master parameters for data, training, RL, and serving.
- `dataset_loader.py` – feature engineering, institutional labeling, sequence builders, and `NatronDataModule`.
- `model_natron.py` – Transformer encoder, multi-task heads, masked/contrastive blocks, and init helpers.
- `losses.py` – masked reconstruction, InfoNCE, and multi-task classification losses.
- `train_natron.py` – orchestrated pretraining, supervised fine-tuning, and optional PPO reinforcement.
- `rl_trainer.py` – trading environment wrapper and PPO trainer built on the Natron encoder.
- `server_natron.py` – Flask + TCP hybrid inference server that scores sequences and returns JSON signals.
- `monitor_natron.py` – lightweight system/GPU/health monitor for deployed servers.
- `start_natron.sh` – boot script that launches the server (and optional monitor) with a given config.
- `natron_ea.mq5` – MetaTrader 5 Expert Advisor that streams candles to the socket bridge and manages trades.
- `Dockerfile`, `requirements.txt` – containerized GPU runtime for training and inference.
- `docs/natron_v2_whitepaper.md` – architecture, research rationale, and operating philosophy.

### Quick Start
1. **Environment Setup**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install torch==2.1.0 --extra-index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```
2. **Prepare Data**
   - Place `data_export.csv` under `data/`.
   - Update `paths` and `data` sections of `configs/natron_config.yaml` if required.
3. **Train the Model**
   ```bash
   python train_natron.py --config configs/natron_config.yaml
   ```
   - Phase 1: masked modelling + contrastive pretraining over engineered features.
   - Phase 2: supervised multi-task fine-tuning (buy / sell / direction / regime).
   - Phase 3 (optional): PPO reinforcement tuning (`rl.enabled: true`).
   - Checkpoints and the feature scaler are saved under `model/`.
4. **Serve Inference**
   ```bash
   bash start_natron.sh configs/natron_config.yaml
   ```
   - REST endpoint: `POST /predict` with the latest 96 OHLCV candles.
   - TCP bridge: newline-delimited JSON on the configured socket (`server.socket`).
   - Monitor logs stream CPU/GPU health while the service runs.
5. **Connect MetaTrader**
   - Load `natron_ea.mq5` into the MetaTrader 5 terminal.
   - Set the EA inputs to match the socket host/port and thresholds.
   - Enable algo-trading; the EA will poll Natron and manage positions.

### Training Pipeline
- **Feature Matrix (100+ signals)** – trend, momentum, volume, volatility, structure, smart money concept, market profile, and calendar embeddings.
- **Label Suite** – buy/sell heuristics, binary direction, and six-state market regime following institutional rules.
- **Sequences** – sliding windows of `(96, 100)` features aligned with next-step labels.
- **Phase 1 (Pretraining)** – random masking + InfoNCE on jittered/time-warped augmentations to learn latent structure.
- **Phase 2 (Supervised)** – multi-task heads with AdamW + ReduceLROnPlateau, encoder freezing warm-up, and early stopping.
- **Phase 3 (RL)** – PPO actor-critic on top of the shared encoder, optimizing profit minus turnover/drawdown penalties.

### Deployment and Ops
- **REST JSON Output**
  ```json
  {
    "buy_prob": 0.71,
    "sell_prob": 0.24,
    "direction_up": 0.69,
    "regime": "BULL_WEAK",
    "confidence": 0.82
  }
  ```
- **Socket Protocol** – Send a JSON payload with `candles` array terminated by `\n`; receive a single-line JSON response.
- **Monitoring** – `monitor_natron.py` polls `/health`, reports CPU/RAM/GPU usage, and can be run standalone or via `start_natron.sh`.
- **Docker Build**
  ```bash
  docker build -t natron:v2 .
  docker run --gpus all -p 8000:8000 -p 9000:9000 natron:v2
  ```

### Extending Natron
- Update `FeatureEngineer` to add domain-specific indicators; the scaler automatically adapts via caching.
- Customize loss weights or schedules in the YAML config.
- Swap PPO for SAC by adding a new trainer module; the environment is self-contained.
- Integrate alternative brokers by mirroring the EA’s socket contract.

For a deeper explanation of the architecture, data philosophy, and experimentation roadmap, read `docs/natron_v2_whitepaper.md`.
