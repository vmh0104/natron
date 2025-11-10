# 🧠 Natron Transformer - System Overview

## Complete End-to-End AI Trading System

This document provides a high-level overview of the Natron Transformer system architecture and components.

## 🎯 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                            │
├─────────────────────────────────────────────────────────────┤
│  data_export.csv → FeatureEngine → LabelGenerator          │
│  (OHLCV)         (~100 features)  (buy/sell/regime)         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                  SEQUENCE CONSTRUCTION                        │
├─────────────────────────────────────────────────────────────┤
│  SequenceCreator: (N, 96, 100) sequences                   │
│  Labels: buy, sell, direction, regime                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              TRAINING PIPELINE (3 PHASES)                    │
├─────────────────────────────────────────────────────────────┤
│  Phase 1: Pretraining (Unsupervised)                        │
│    - Masked Modeling + Contrastive Learning                 │
│    - Learns market structure                                │
│                                                              │
│  Phase 2: Supervised Fine-Tuning                            │
│    - Multi-task learning                                    │
│    - Buy/Sell, Direction, Regime prediction                 │
│                                                              │
│  Phase 3: Reinforcement Learning (Optional)                 │
│    - PPO algorithm                                           │
│    - Optimizes trading decisions                            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT                                │
├─────────────────────────────────────────────────────────────┤
│  Option A: Flask REST API (port 5000)                      │
│    - /predict endpoint                                       │
│    - JSON request/response                                   │
│                                                              │
│  Option B: TCP Socket Server (port 8888)                   │
│    - Real-time MQL5 integration                             │
│    - Bidirectional communication                            │
│                                                              │
│  Option C: MetaTrader 5 Expert Advisor                      │
│    - Automated order execution                              │
│    - Live trading                                           │
└─────────────────────────────────────────────────────────────┘
```

## 📦 Component Breakdown

### 1. Feature Engineering (`feature_engine.py`)
- **Purpose**: Generate ~100 technical indicators from OHLCV data
- **Categories**:
  - Moving Averages (13)
  - Momentum (13)
  - Volatility (15)
  - Volume (9)
  - Price Patterns (8)
  - Returns (8)
  - Trend Strength (6)
  - Statistical (6)
  - Support/Resistance (4)
  - SMC (6)
  - Market Profile (10)

### 2. Label Generation (`label_generator.py`)
- **Purpose**: Create multi-task labels
- **Outputs**:
  - Buy signal (binary, ≥2 conditions)
  - Sell signal (binary, ≥2 conditions)
  - Direction (up/down)
  - Regime (6 classes)

### 3. Dataset Loader (`dataset_loader.py`)
- **Purpose**: Create PyTorch datasets from processed data
- **Features**:
  - Sequence construction (96 candles)
  - Train/val/test splitting
  - DataLoader creation

### 4. Model Architecture (`model_natron.py`)
- **Type**: Transformer Encoder
- **Specs**:
  - 6 layers, 8 heads, 256-dim embeddings
  - Multi-task heads (buy, sell, direction, regime)
  - Pretraining model (masked modeling + contrastive)

### 5. Loss Functions (`losses.py`)
- **MultiTaskLoss**: Combined loss for supervised training
- **PretrainLoss**: Reconstruction + contrastive loss
- **RLLoss**: PPO loss for reinforcement learning

### 6. Training Scripts
- **`train_pretrain.py`**: Phase 1 unsupervised pretraining
- **`train_natron.py`**: Phase 2 supervised fine-tuning
- **`train_rl.py`**: Phase 3 reinforcement learning

### 7. Deployment Servers
- **`api_server.py`**: Flask REST API (port 5000)
- **`socket_server.py`**: TCP socket server (port 8888)

### 8. MQL5 Integration
- **`natron_ea.mq5`**: MetaTrader 5 Expert Advisor
- Connects to Python socket server
- Executes trades based on AI predictions

### 9. Monitoring & Utilities
- **`monitor_natron.py`**: System health monitoring
- **`test_natron.py`**: Component testing
- **`start_natron.sh`**: Automated startup
- **`stop_natron.sh`**: Graceful shutdown

## 🔄 Data Flow

### Training Flow
```
Raw OHLCV → Features → Labels → Sequences → Model Training
```

### Inference Flow
```
Live Candles → Features → Model → Predictions → Trading Signals
```

### Real-Time Trading Flow
```
MQL5 EA → Socket Server → Model → Predictions → MQL5 EA → Orders
```

## 🎓 Learning Philosophy

### Layer 1: Structure Understanding
- **Method**: Masked modeling + contrastive learning
- **Goal**: Learn hidden market dynamics
- **Output**: Encoder that understands market structure

### Layer 2: Signal Recognition
- **Method**: Multi-task supervised learning
- **Goal**: Detect patterns leading to outcomes
- **Output**: Buy/sell/direction/regime predictions

### Layer 3: Behavioral Adaptation
- **Method**: Reinforcement learning (PPO)
- **Goal**: Optimize profit vs risk
- **Output**: Trading policy optimized for returns

## 📊 Model Output Format

```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_up": 0.69,
  "regime": "BULL_WEAK",
  "confidence": 0.82,
  "regime_probs": [0.1, 0.4, 0.2, 0.1, 0.1, 0.1],
  "timestamp": "2024-01-01T12:00:00"
}
```

## 🔧 Configuration

All settings in `config.yaml`:
- Model architecture
- Training hyperparameters
- API/socket server settings
- MQL5 EA parameters

## 🚀 Quick Start

1. **Install**: `pip install -r requirements.txt`
2. **Prepare Data**: Place `data_export.csv` in workspace
3. **Train**: `python train_natron.py --data data_export.csv`
4. **Deploy**: `./start_natron.sh`
5. **Trade**: Connect MQL5 EA

## 📈 Performance Metrics

- **Buy/Sell Accuracy**: Classification performance
- **Direction Accuracy**: Up/down prediction
- **Regime Accuracy**: Market state classification
- **Multi-task Loss**: Combined loss across tasks

## 🛡️ Production Considerations

- **GPU Acceleration**: CUDA support for training
- **Error Handling**: Comprehensive try/except blocks
- **Logging**: File and console logging
- **Monitoring**: Health checks and metrics
- **Scalability**: Batch processing support

## 📚 Key Files Reference

| File | Purpose |
|------|---------|
| `feature_engine.py` | Technical indicator generation |
| `label_generator.py` | Multi-task label creation |
| `model_natron.py` | Transformer architecture |
| `train_natron.py` | Main training script |
| `api_server.py` | REST API deployment |
| `socket_server.py` | MQL5 integration |
| `natron_ea.mq5` | MetaTrader 5 EA |
| `config.yaml` | System configuration |

## 🎯 Use Cases

1. **Research**: Study market patterns and regimes
2. **Backtesting**: Evaluate trading strategies
3. **Live Trading**: Automated execution via MQL5
4. **Signal Generation**: API-based predictions
5. **Market Analysis**: Regime classification

---

**System Status**: ✅ Complete and Ready for Deployment

For detailed usage, see `README.md` and `QUICKSTART.md`.
