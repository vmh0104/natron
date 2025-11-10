Natron Transformer – End-to-End Multi-Task Trading System
=========================================================

Overview
--------
- **Natron Transformer** is a GPU-first, multi-task learning system for institutional-grade financial trading.
- Processes rolling windows of 96 OHLCV candles enriched by 100+ engineered technical features.
- Learns three intelligence layers: self-supervised market structure, supervised signal recognition, and reinforcement-driven behavioral adaptation.
- Ships with automated data engineering, configurable training pipeline, real-time inference server, MT5 Expert Advisor integration, monitoring, and deployment assets.

Repository Layout
-----------------
- `configs/` – YAML configs for data, model, training, and deployment pipelines.
- `docs/` – Whitepaper (`NATRON_WHITEPAPER.md`), experiment templates, and operational notes.
- `data/` – Place `data_export.csv` and cached feature tensors.
- `models/` – Saved checkpoints (`models/natron_v2.pt`, encoder pretrain, RL policy).
- `scripts/` – Operational scripts (`start_natron.sh`, `monitor_natron.py`, Docker assets).
- Root Python modules – Core source (`dataset_loader.py`, `feature_engineering.py`, `model_natron.py`, `train_natron.py`, `rl_trainer.py`, `server_natron.py`, etc.).

Quickstart
----------
1. `python -m venv .venv && source .venv/bin/activate`
2. `pip install -U pip && pip install -r requirements.txt`
3. Place raw candles in `data/data_export.csv`
4. `python train_natron.py --config configs/natron_config.yaml`
5. Launch inference server: `python server_natron.py --config configs/natron_config.yaml`
6. Connect MT5 EA (`natron_ea.mq5`) to the socket gateway for live trading.

Key Capabilities
----------------
- FeatureEngine module synthesizes ~100 technical + statistical descriptors automatically.
- Self-supervised Transformer pretraining (masked modeling + InfoNCE contrastive pairs).
- Multi-task fine-tuning (buy, sell, direction, regime) with adaptive loss weighting.
- Optional PPO reinforcement loop optimizing profit, turnover, drawdown trade-offs.
- Low-latency socket bridge and Flask REST API for downstream consumers.
- Monitoring, logging, restart automation, and containerized deployment boilerplate.

Conventions
-----------
- Python ≥3.10, PyTorch 2.x with CUDA.
- Configuration-first: override YAML via CLI flags or environment variables.
- All modules log to `logs/` using structured JSON logging (`rich` console tails optional).
- Tests & notebooks omitted for brevity; integrate with `pytest` and `wandb` as needed.

Documentation
-------------
- `docs/NATRON_WHITEPAPER.md` – Architecture, training philosophy, and technical deep dive.
- Inline docstrings and type hints across modules.
- `configs/experiments/` (optional) for sweeps and ablations.

License
-------
Internal R&D – add explicit license before distribution.
