# Natron V2 End-to-End Architecture

*Author: Sonnet 4.5 – Logic Designer & Documentation Specialist*  
*Revision: 2025-11-10*

---

## 1. Vision
Natron V2 is a three-stage learning system that internalises the grammar of market microstructure, recognises tradeable signals, and adapts behaviour to optimise P&L under risk controls. It operates on 96-candle OHLCV sequences and produces four synchronized outputs: buy probability, sell probability, directional likelihood, and market regime classification.

---

## 2. Data Substrates
- **Input Source:** `data_export.csv` containing `time, open, high, low, close, volume`.
- **Sampling:** 15min or 1h timeframe normalised to UTC.
- **Windowing:** Rolling windows of length 96 with stride 1.
- **Caching:** Intermediate features stored in `artifacts/cache/`.

### 2.1 Feature Engine
Feature groups (∼100 features):
- Moving averages & slopes (SMA/EMA/SMMA/KAMA, crossovers, price-to-MA).
- Momentum suite (RSI, ROC, CCI, Stochastic K/D, MACD components).
- Volatility curves (ATR, Bollinger, Keltner, Donchian, realised volatility).
- Volume analytics (OBV, VWAP, volume ratio, money flow index).
- Price patterns (candlestick bodies, shadows, gap ratios, doji tests).
- Returns (log, intraday, cumulative, rolling Sharpe).
- Trend strength (ADX, DI channels, Aroon).
- Statistical signatures (z-score, kurtosis, skewness, Hurst exponent).
- Support/resistance distances (rolling highs/lows).
- Smart Money Concepts (swing points, break-of-structure, change-of-character).
- Market profile (POC, VAH, VAL, profile entropy, TPO ratios).

### 2.2 Label Synthesis
- **Buy / Sell triggers:** Weighted set of confluences (MA stack, RSI regime, Bollinger alignment, volume spikes, MACD momentum, candle location).
- **Direction:** Binary log-return over lookahead horizon.
- **Regime:** 6-class taxonomy (strong/weak bull, range, weak/strong bear, volatile) derived from multi-scale trend & ADX plus volatility overrides.

---

## 3. Transformer Anatomy
- **Backbone:** 8-layer encoder with pre-norm, GELU, multi-head attention (8 heads), feed-forward width 1024, dropout 0.1.
- **Tokenisation:** Continuous features projected via learnable linear stem, positional encodings (sinusoidal + learned bias).
- **Masked Modelling Head:** Predicts masked tokens with denoising linear head.
- **Contrastive Head:** Projections (128-dim) for InfoNCE across augmented views.
- **Supervised Heads:**  
  - `buy_head`: linear → sigmoid.  
  - `sell_head`: linear → sigmoid.  
  - `direction_head`: linear → softmax(2).  
  - `regime_head`: linear → softmax(6).  
  All operate on `[CLS]` embedding.

---

## 4. Training Curriculum
1. **Phase 1 – Pretraining**
   - Mask 25% of tokens.
   - Augment views via stochastic feature dropout & jitter.
   - Optimise combination of masked MSE loss and InfoNCE.
2. **Phase 2 – Supervised Fine-Tuning**
   - Warm-start from pretrained encoder.
   - Weighted multi-task BCE + CE losses.
   - ReduceLROnPlateau scheduler keyed on validation macro-F1.
3. **Phase 3 – Reinforcement Adaption (Optional)**
   - PPO agent acting on environment fed by historical sequences or simulated stream.
   - Reward `profit - α * turnover - β * drawdown`.
   - Policy initialised with supervised heads, updates small step-size to preserve calibration.

---

## 5. Deployment Topology
- **Inference server:** Async TCP + Flask REST (predict + health endpoints) on GPU node.
- **Monitoring:** Latency histograms, throughput, exception counts; optional Slack alerts.
- **MQL5 EA:** Streams latest candles to server, receives JSON decisions, annotates chart, and executes orders with safeguards.
- **Docker:** CUDA base image, installs PyTorch + deps, exposes training and inference entry points.

---

## 6. Experiment Protocol
- **Metrics:** ROC-AUC, PR-AUC for buy/sell; accuracy & F1 for direction/regime; Sharpe/Sortino for RL.
- **Cross-validation:** Walk-forward with anchored expanding windows.
- **Logging:** TensorBoard scalars, JSONL for sample-level predictions, optional MLflow integration.
- **Checkpointing:** Best validation composite score, plus periodic EMA snapshots.

---

## 7. Future Extensions
- Online self-supervised adaptation with streaming unlabeled data.
- Dynamic risk budgeting integration with broker margin APIs.
- Hierarchical regime modelling (macro regime → micro action).
- Federated training across multiple brokers/exchanges.

---

## 8. References
- Vaswani et al., "Attention Is All You Need" (2017).
- Oord et al., "Representation Learning with Contrastive Predictive Coding" (2018).
- Sutton et al., "Policy Gradient Methods for Reinforcement Learning" (2000).
- Institutional trading heuristics aggregated from sell-side research (2015-2024).
