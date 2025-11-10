Natron V2 Transformer – End-to-End AI Trading Whitepaper
========================================================

Author: Sonnet 4.5 (Logic Designer)  
Revision: 2025-11-10

Abstract
--------
Natron V2 is an institutional-grade trading intelligence framework that unifies self-supervised learning, supervised multi-task prediction, and reinforcement-driven decision optimization on GPU infrastructure. The system ingests multi-timeframe OHLCV candles, synthesizes 100+ technical descriptors, and produces actionable buy/sell, directional, and market regime insights for automated execution pipelines such as MetaTrader 5.

1. Philosophy
-------------
- **Structure Understanding**: Masked-token reconstruction and contrastive metric learning reveal latent market grammars from unlabeled sequences.
- **Signal Recognition**: Supervised multi-head objectives align latent embeddings with institutional heuristics for buy/sell momentum, directional shifts, and six-state regimes.
- **Behavioral Adaptation**: Reinforcement optimization adapts to evolving liquidity, transaction costs, and risk constraints using PPO under live or simulated conditions.

2. Data & Feature Stack
-----------------------
- Source: `data_export.csv` with fields `time, open, high, low, close, volume`.
- Pre-processing: Outlier winsorization, timezone normalization, missing data forward-fill.
- Feature families (≈100 total):
  - Moving averages & slopes (EMA, WMA, crossovers, price distance ratios).
  - Momentum oscillators (RSI, ROC, Stochastic, MACD, Awesome oscillator).
  - Volatility envelopes (ATR, Bollinger, Donchian, Keltner, realized volatility).
  - Volume analytics (OBV, VWAP deltas, Money Flow Index, volume pressure).
  - Price pattern heuristics (candlestick anatomy, gaps, engulfing signals).
  - Returns (log returns, intraday range, cumulative regimes).
  - Trend strength (ADX, DI±, Aroon).
  - Statistical textures (skew, kurtosis, z-score, Hurst exponent).
  - Support & resistance distances (swing-high/low footprints).
  - Smart Money Concepts (break of structure, change of character).
  - Market profile metrics (POC, value area, entropy).

3. Label Engineering
--------------------
Two-tier logic: institutional heuristics for buy/sell triggers and macro regime stratification. Label smoothing and hysteresis prevent churning around thresholds.

- **Buy Signal**: Activate when ≥2 bullish heuristics align (MA ladder, RSI momentum, Bollinger/volume confirmation, MACD trend, candle position).
- **Sell Signal**: Mirror bearish heuristics triggered when ≥2 are met.
- **Direction**: Binary (up/down) derived from forward returns over `delta=4` candles.
- **Regime**: Six classes defined by directional drift, ADX, and volatility anomalies. Volatility class overrides directional classes when ATR or volume exceed 90th percentile.

4. Model Architecture
---------------------
- Input: `(batch, seq_len=96, feature_dim≈100)`.
- Positional Encoding: Learnable rotary embeddings for temporal robustness.
- Encoder: Deep Transformer stack (6 layers, `d_model=256`, `n_heads=8`) with stochastic depth, pre-norm residuals, and adaptive layer-drop during pretraining.
- Projection Heads:
  - **Masked Reconstruction Head**: Linear decoder with feature-specific covariance regularization.
  - **Contrastive Head**: Two-layer projector (`256→128→128`) with L2-normalized outputs.
  - **Task Heads**: Shared MLP conditioner feeding:
    - `buy_head`: Sigmoid output.
    - `sell_head`: Sigmoid output.
    - `direction_head`: Softmax over 2.
    - `regime_head`: Softmax over 6.
- Regularization: Dropout, mixup/cutmix on feature space, uncertainty-weighted loss aggregation.

5. Training Curriculum
----------------------
### Phase 1 – Self-Supervised Encoder Pretraining
- Masking: Randomly mask 20% of tokens × 20% of feature dimensions.
- Loss: Sum of MSE reconstruction and InfoNCE contrastive objective over augmented positive pairs (time warp, scaling, jitter).
- Optimizer: AdamW, cosine warm restarts, gradient accumulation for long sequences.
- Checkpoint: `models/natron_encoder.pt`.

### Phase 2 – Supervised Fine-Tuning
- Initialize from pretrained encoder; optionally freeze bottom `k` layers for first `N` epochs.
- Multi-task loss `L = Σ w_t * L_t`, with adaptive uncertainty weighting modulating `w_t`.
- Form of `L_t`:
  - `L_buy`, `L_sell`: Binary cross entropy with focal modulation.
  - `L_direction`: Cross-entropy + calibration regularizer.
  - `L_regime`: Class-balanced cross-entropy + temporal coherence penalty.
- Validation: Matthews correlation for classification, regime F1, calibration ECE.
- Early stopping via plateau on composite score.
- Checkpoint: `models/natron_v2.pt`.

### Phase 3 – Reinforcement Learning (PPO)
- Environment: Historical replay or paper-trading stream using latest policy.
- State: Recent Transformer embeddings, current position, PnL, regime context.
- Action: Continuous target position ∈ [-1, +1] or discrete buy/hold/sell.
- Reward: `profit - α*turnover - β*drawdown`, tunable via YAML.
- Risk: CVaR constraints enforced via Lagrangian penalty.
- Output: `models/natron_policy.zip`.

6. Inference & Deployment
-------------------------
- Socket server (`server_natron.py`): JSON/TCP bridge for MT5 EA and other clients (<50 ms latency goal).
- Flask API (`flask_app.py`): `/predict` endpoint returning buy/sell probabilities, direction distribution, and regime label with confidence.
- Monitoring (`monitor_natron.py`): p95 latency, queue depth, heartbeat health-check.
- Dockerfile: CUDA runtime base, installs requirements, exposes Flask + socket ports, config via environment.
- Start script: `scripts/start_natron.sh` orchestrates services and background RL actor.

7. MetaTrader 5 Integration
---------------------------
- EA `natron_ea.mq5` maintains persistent TCP socket, streams latest 96-bar buffer, and parses JSON response.
- Execution: Position sizing via Kelly fraction scaling, optional hedging toggles, UI overlays (regime banner, buy/sell heatmap).
- Failover: Revert to safety mode if server unresponsive > `timeout_ms`.

8. Experimentation & Logging
----------------------------
- Logging: `rich` console + structured JSON. Optional TensorBoard and Weights & Biases.
- Experiments: Config-driven (Hydra/OmegaConf). Support for reproducibility via seeded dataloaders and deterministic CuDNN toggles.
- Evaluation: Backtest metrics (Sharpe, Sortino, max drawdown, win rate) logged per epoch.

9. Roadmap
----------
- Multi-asset joint training with cross-asset attention.
- Meta-learning for regime shift rapid adaptation.
- On-device distillation for edge deployment (Jetson, Raspberry Pi).
- Graph neural augmentation for order book features.

Appendix
--------
- Hyperparameters: See `configs/natron_config.yaml`.
- Glossary: BOS = Break of Structure, CHOCH = Change of Character, POC = Point of Control, ATR = Average True Range, ADX = Average Directional Movement Index.
