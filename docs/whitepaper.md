# Natron V2 Whitepaper (Draft)

## 1. Executive Summary

Natron V2 is a multi-stage trading intelligence system that combines self-supervised market representation learning, supervised multi-task classification, and reinforcement learning for behavioral adaptation. This document outlines the architecture, training methodology, and deployment strategy.

## 2. Pipeline Overview

1. **Feature Extraction** – Transform OHLCV candles into a 100-dimensional feature space capturing momentum, volatility, trend, and market microstructure.
2. **Sequence Modeling** – Consume sequences of 96 candles to model temporal dependencies.
3. **Pretraining** – Masked reconstruction and contrastive objectives to learn latent market grammar.
4. **Supervised Fine-Tuning** – Jointly optimize buy/sell, directional bias, and regime classification heads.
5. **Reinforcement Learning** – Optimize real-world trading metrics using PPO against a market simulator.

## 3. Data Specification

- Input CSV columns: `time, open, high, low, close, volume`
- Sequence length: 96 candles (supports M15/H1)
- Feature groups: Moving Average, Momentum, Volatility, Volume, Price Pattern, Returns, Trend Strength, Statistical, Support/Resistance, SMC, Market Profile

## 4. Model Architecture

- Transformer encoder with learnable positional embeddings and gated residual connections
- Shared encoder for all tasks
- Task-specific heads:
  - Buy/Sell logits (sigmoid outputs)
  - Directional prediction (2-class softmax)
  - Regime classification (6-class softmax)

## 5. Training Strategy

### Pretraining

- Randomly mask 15% of tokens and reconstruct features (L1 + cosine loss)
- Contrast between augmented views of the same sequence using an NT-Xent loss

### Supervised

- Weighted binary cross-entropy (buy/sell)
- Cross-entropy for direction and regime
- Focal/entropy regularizers to balance class imbalance

### Reinforcement

- Actor-critic PPO over simulated PnL, turnover, and drawdown penalties
- Rolling window recalibration with continual learning hooks

## 6. Deployment

- TorchScript model served via Flask + TCP bridge
- MetaTrader 5 Expert Advisor communicates via JSON over sockets
- Monitoring stack captures latency, hit ratio, and error rates

## 7. Reinforcement Learning Integration

- Environment: rolling window simulator that feeds Natron feature sequences and closes into PPO agent.
- Action space: {flat, long, short} with position transitions penalised for turnover.
- Reward: `R = profit - 0.1 * turnover - 0.2 * drawdown`, scaled for numerical stability.
- Policy: Actor-critic feed-forward network (flattened embeddings) trained with PPO and GAE.
- Deployment: PPO weights exported separately and optionally blended into signal post-processing.

## 8. Realtime Execution Path

1. MT5 EA streams latest OHLCV block to Python server (or precomputed features).
2. `server_natron.py` computes/normalizes features, runs the Transformer, and responds with JSON.
3. EA visualises probabilities, applies thresholds, and dispatches orders (with deconfliction logic).
4. `monitor_natron.py` watches health endpoints, CPU/GPU utilisation, and latency metrics.

## 9. Reliability & Monitoring

- Flask health endpoint for liveness.
- Structured logging with latency annotations.
- Optional GPU telemetry reporting via PyTorch APIs.
- Auto-reconnect logic in EA to handle transient socket drops.

## 10. Next Steps

- Expand feature coverage with adaptive lookbacks and cross-asset signals.
- Integrate PPO policy blending with supervised logits for production deployment.
- Introduce ensemble uncertainty estimation for risk-aware order sizing.
