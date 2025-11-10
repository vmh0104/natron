# 🧠 Natron Transformer – Multi-Task Financial Trading Model

**End-to-End GPU Pipeline for Algorithmic Trading**

---

## 🎯 Overview

Natron is a sophisticated Transformer-based AI system that learns market dynamics through three phases of intelligence:

1. **Structure Understanding** (Pretraining) - Learn hidden market patterns
2. **Signal Recognition** (Supervised) - Detect actionable trading opportunities
3. **Behavioral Adaptation** (Reinforcement) - Optimize for profit and risk balance

### Key Features

- **Multi-Task Learning**: Simultaneous prediction of buy/sell signals, direction, and market regime
- **100+ Technical Features**: Comprehensive market representation
- **GPU-Optimized**: PyTorch 2.x with CUDA acceleration
- **Real-Time Integration**: MQL5 Expert Advisor for MetaTrader 5
- **Production-Ready**: Flask API + Socket server for live trading

---

## 🏗️ Architecture

```
Input: 96 OHLCV Candles (M15/H1)
    ↓
Feature Engineering (~100 indicators)
    ↓
Transformer Encoder (6 layers, 8 heads)
    ↓
Multi-Task Heads:
├── Buy Probability (sigmoid)
├── Sell Probability (sigmoid)
├── Direction (up/down softmax)
└── Market Regime (6-class softmax)
```

### Market Regimes

| ID | Regime | Condition |
|----|--------|-----------|
| 0 | BULL_STRONG | Trend > +2%, ADX > 25 |
| 1 | BULL_WEAK | 0 < Trend ≤ 2%, ADX ≤ 25 |
| 2 | RANGE | Lateral market |
| 3 | BEAR_WEAK | -2% ≤ Trend < 0, ADX ≤ 25 |
| 4 | BEAR_STRONG | Trend < -2%, ADX > 25 |
| 5 | VOLATILE | ATR > 90th percentile |

---

## 🚀 Quick Start

### Prerequisites

- Ubuntu/Debian VM with NVIDIA GPU
- CUDA 12.1+ and cuDNN 8.9+
- Python 3.10+
- MetaTrader 5 (for live trading)

### Installation

```bash
# Clone repository
cd /workspace

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data models checkpoints logs cache
```

### Data Preparation

Place your OHLCV data in `data/data_export.csv` with columns:
```
time, open, high, low, close, volume
```

### Training Pipeline

```bash
# Phase 1: Pretraining (unsupervised)
python train_natron.py --phase pretrain --config config.yaml

# Phase 2: Supervised fine-tuning
python train_natron.py --phase supervised --config config.yaml

# Phase 3: Reinforcement learning (optional)
python train_natron.py --phase reinforcement --config config.yaml

# Or run all phases sequentially
python train_natron.py --phase all --config config.yaml
```

### Inference & Deployment

```bash
# Start Flask API server
python api_server.py --config config.yaml --port 5000

# Start Socket server for MQL5
python socket_server.py --config config.yaml --port 9090

# Test inference
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

---

## 📂 Project Structure

```
/workspace/
├── config.yaml                 # Main configuration
├── requirements.txt            # Python dependencies
├── natron_init.py             # Initialization script
│
├── src/
│   ├── feature_engine.py      # 100+ technical indicators
│   ├── labeling.py            # Buy/sell/regime labeling
│   ├── dataset_loader.py      # Sequence creation & data loading
│   ├── model_natron.py        # Transformer architecture
│   ├── losses.py              # Multi-task loss functions
│   ├── train_natron.py        # Training orchestrator
│   ├── pretrain.py            # Phase 1: Unsupervised learning
│   ├── supervised.py          # Phase 2: Multi-task training
│   └── reinforcement.py       # Phase 3: RL optimization
│
├── api/
│   ├── api_server.py          # Flask REST API
│   ├── socket_server.py       # Real-time socket server
│   └── inference.py           # Inference engine
│
├── mql5/
│   └── natron_ea.mq5          # MetaTrader 5 Expert Advisor
│
├── deployment/
│   ├── Dockerfile             # Container setup
│   ├── docker-compose.yml     # Multi-service orchestration
│   ├── start_natron.sh        # Startup script
│   └── monitor_natron.py      # System monitoring
│
├── tests/
│   ├── test_features.py
│   ├── test_model.py
│   └── test_api.py
│
├── data/
│   └── data_export.csv        # Input OHLCV data
│
├── models/
│   └── natron_v2.pt           # Trained model weights
│
├── checkpoints/               # Training checkpoints
├── logs/                      # Training & inference logs
└── cache/                     # Feature cache
```

---

## 🔥 Training Phases

### Phase 1: Pretraining (Unsupervised)

**Objective**: Learn latent market representations

**Methods**:
- Masked token modeling (15% random masking)
- Contrastive learning (InfoNCE loss)

**Output**: Pretrained encoder understanding market structure

### Phase 2: Supervised Fine-Tuning

**Objective**: Multi-task prediction

**Tasks**:
- Buy signal probability
- Sell signal probability
- Directional forecast (up/down)
- Market regime classification

**Loss**: Weighted combination of BCE + CrossEntropy

### Phase 3: Reinforcement Learning (Optional)

**Objective**: Maximize trading performance

**Algorithm**: Proximal Policy Optimization (PPO)

**Reward Function**:
```
R = profit - α × turnover - β × drawdown
```

---

## 🧮 API Response Format

```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_up": 0.69,
  "direction_down": 0.31,
  "regime": "BULL_WEAK",
  "regime_id": 1,
  "confidence": 0.82,
  "timestamp": "2025-11-10T15:30:00Z"
}
```

---

## ⚡ MQL5 Integration

### Architecture

```
MetaTrader 5 (MQL5 EA) ⇄ Python Socket Server ⇄ Natron AI Model (GPU)
        ↑                                              ↓
   Tick/Candle Data                        JSON Response (signals)
```

### Communication Protocol

1. **MT5 → Python**: Send last 96 OHLCV candles as JSON
2. **Python → MT5**: Return trading signals + probabilities
3. **MT5**: Execute orders based on AI recommendations

---

## 🐳 Docker Deployment

```bash
# Build image
docker build -t natron-transformer -f deployment/Dockerfile .

# Run container with GPU support
docker run --gpus all -p 5000:5000 -p 9090:9090 \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/models:/workspace/models \
  natron-transformer

# Or use docker-compose
docker-compose -f deployment/docker-compose.yml up -d
```

---

## 📊 Monitoring

```bash
# Start monitoring dashboard
python deployment/monitor_natron.py

# View TensorBoard logs
tensorboard --logdir=runs --port=6006

# Check system health
curl http://localhost:5000/health
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v --cov=src

# Test specific components
pytest tests/test_features.py
pytest tests/test_model.py
pytest tests/test_api.py
```

---

## 📈 Performance Metrics

During training, monitor:

- **Pretraining**: Reconstruction loss, contrastive accuracy
- **Supervised**: Multi-task losses, per-task accuracy/F1
- **Reinforcement**: Cumulative reward, Sharpe ratio, max drawdown

---

## 🛠️ Configuration

Edit `config.yaml` to customize:

- Model architecture (layers, heads, dimensions)
- Training hyperparameters
- Feature engineering settings
- API/socket server ports
- Logging verbosity

---

## 🤝 Contributing

This is a complete production system. Extend by:

1. Adding custom technical indicators in `feature_engine.py`
2. Implementing new labeling strategies in `labeling.py`
3. Experimenting with model architectures in `model_natron.py`
4. Creating new reward functions for RL in `reinforcement.py`

---

## 📝 License

Proprietary - Natron AI Trading System

---

## 🎓 Citation

```bibtex
@software{natron_transformer_2025,
  title={Natron Transformer: Multi-Task Financial Trading Model},
  author={Natron AI Team},
  year={2025},
  version={2.0}
}
```

---

## 📧 Support

For technical support or feature requests, contact the development team.

---

**Built with ⚡ by Claude Sonnet 4.5 × GPT-5 Codex High × Opus 4.1 × Composer 1**
