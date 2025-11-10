## Natron V2 End-to-End Whitepaper

### 1. Vision
Natron V2 is designed to learn the grammar of financial markets through successive layers of intelligence: structural understanding, supervised signal recognition, and behavioural adaptation. The system consumes 96-candle OHLCV sequences, builds ~100 handcrafted features rooted in institutional technical analysis, and trains a transformer encoder to extract a latent market representation deployed for multi-task decisioning and reinforcement learning.

### 2. Data Manifold
- **Source:** `data_export.csv` (time, open, high, low, close, volume).
- **Cadence:** M15/H1 candles; the pipeline is timeframe agnostic as long as the sampling stays consistent.
- **Feature Engine:** 15 families including moving averages, volatility envelopes, oscillators, volume statistics, smart money concepts, market profile, and cyclical embeddings. The FeatureEngineer outputs a 100+ dimensional matrix aligned to each candle.
- **Labels:** Institutional heuristics produce buy/sell triggers, direction (up/down), and six market regimes (BULL_STRONG, BULL_WEAK, RANGE, BEAR_WEAK, BEAR_STRONG, VOLATILE). Labels are balanced through thresholding and confirmatory checks.
- **Sequences:** Sliding windows of 96 steps result in tensors shaped `(N, 96, 100)` paired with the label snapshot at the window endpoint.

### 3. Architectural Overview
| Stage | Objective | Methodology |
|-------|-----------|-------------|
| Pretraining | Learn structural representations | Masked token reconstruction + InfoNCE contrastive learning on augmented views |
| Supervised | Multi-task decisioning | Transformer encoder with sigmoid/softmax heads for buy, sell, direction, regime |
| Reinforcement | Behavioural adaptation | PPO actor-critic anchored on encoder embeddings; reward = profit − α·turnover − β·drawdown |

### 4. Encoder Backbone
- **Embedding:** Linear projection from 100-dim features to 256-dim hidden space with learned CLS token.
- **Positional Encoding:** Sinusoidal embeddings up to 256 positions.
- **Transformer:** 6 encoder layers, 8 heads, 512-d feed-forward blocks, dropout 0.1, layer norm pre-activation.
- **Pooling:** CLS token by default; mean pooling optional via config.
- **Heads:** 
  - Multi-task classification/regression heads for buy/sell/direction/regime.
  - Masked reconstruction MLP for denoising.
  - Projection head (128-dim) normalized for contrastive loss.

### 5. Pretraining Regimen
1. **Masking:** Apply Bernoulli mask (default 15%) across tokens, forcing the model to reconstruct corrupted features.
2. **Augmentations:** Jitter, feature dropout, time warp to generate twin views for contrastive training.
3. **Objectives:** Weighted combination of masked MSE and InfoNCE. AdamW with grad clipping ensures stability.
4. **Outcome:** Encoder learns seasonality, volatility clustering, momentum phases, and latent market regimes without label exposure.

### 6. Supervised Fine-Tuning
- **Losses:** BCE with logits for buy/sell, categorical cross-entropy for direction/regime. Weighting configurable via YAML.
- **Optimization:** AdamW (1e-4, wd 1e-5), ReduceLROnPlateau scheduler on direction accuracy, encoder freezing warm-up, early stopping.
- **Metrics:** Direction accuracy, regime accuracy, buy/sell precision (monitored via logging).
- **Checkpoints:** Best model saved with scaler for deterministic deployment.

### 7. Reinforcement Layer
- **Environment:** Uses the same sequence dataset with access to raw prices; agent decides position (`-1`, `0`, `+1`) per timestep.
- **Reward:** `pnl - α * turnover - β * drawdown` encourages profitable yet conservative policies.
- **Actor-Critic:** Shares transformer encoder; policy/value heads fine-tuned via PPO updates (clip ratio 0.2, target KL 0.02).
- **Usage:** Optional stage that can be toggled. Ideal for continual learning on live or simulated feeds.

### 8. Serving Topology
```
MetaTrader 5 EA  ⇄  Natron TCP Bridge (JSON sockets)
                       │
                       ├─ Flask REST API (/predict)
                       └─ Natron Transformer (GPU)
```
- **REST:** Accepts last 96 candles and returns JSON predictions for integrations.
- **Socket:** Lightweight newline-delimited JSON channel for trading terminals (e.g., MQL5 EA).
- **Monitoring:** CPU/RAM/GPU telemetry + health pings via `monitor_natron.py`.
- **Deployment:** Docker image with CUDA runtime; `start_natron.sh` bootstraps both server and monitor.

### 9. MetaTrader Integration
- `natron_ea.mq5` streams candles at configurable intervals, parses the JSON response, renders overlay metrics, and places trades using `CTrade`.
- Threshold-based decisioning with ATR-derived protective stops and a confidence-driven flattening rule.
- Easily extensible to visual dashboards by leveraging the returned `regime` and `confidence`.

### 10. Experimentation Framework
- **Configuration:** Single YAML governs seeds, loaders, model depth, losses, RL hyperparameters, and serving ports.
- **Caching:** Feature and label caches (Parquet / Torch) accelerate iterative experimentation.
- **Extensibility:** Swap modules – e.g., add alternative heads, custom losses, or new RL algorithms – without touching the core pipeline.
- **Reproducibility:** Seeds, scaling artefacts, and checkpoints saved in `model/`, ensuring deterministic inference.

### 11. Monitoring & Ops
- Regular logs capture pretraining losses, supervised metrics, LR adjustments, and RL reward curves.
- Deployment monitor records resource usage; alerts can be layered by watching `monitor.log`.
- MQL5 EA prints live probabilities and regime; can trigger alerts or overlay chart artefacts.

### 12. Roadmap
- **Portfolio Context:** Extend to multi-symbol baskets with cross-asset encoders.
- **Risk Conditioning:** Integrate VaR/CVaR estimators into the reward function.
- **Market Microstructure:** Incorporate order book / tick-level features when available.
- **Explainability:** Attach SHAP/attention visualizers to decode key drivers behind signals.

Natron V2 provides a holistic research-to-production workflow for institutional-grade algorithmic trading, combining engineered domain knowledge, transformer learning, and reinforcement fine-tuning under a single GPU-optimized roof.
