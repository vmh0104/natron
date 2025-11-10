# Natron End-to-End AI Trading System

**Version:** Natron V1.0 – Galaxy-class  
**Inspired by:** QuantConnect, DeepMind AlphaTrade, MetaTrader MQL5

---

## 🎯 Overview

Natron is an advanced AI trading system that combines:
- **Multi-head Transformer Models** for regime classification, context scoring, and price forecasting
- **Realtime Execution Engine** with Python + MQL5 integration
- **Self-Learning Capabilities** that adapt from trading feedback
- **Production-Ready Deployment** with Docker and monitoring

---

## 📁 Project Structure

```
natron/
├── Phase 1 - Data Pipeline/
│   ├── data_pipeline.py          # Main pipeline orchestrator
│   ├── feature_engineering.py    # 50-70 technical indicators
│   └── regime_labeler.py          # 6-class regime labeling
│
├── Phase 2 - Model Training/
│   ├── model_natron.py           # Multi-head transformer architecture
│   ├── dataset_loader.py          # PyTorch Dataset & DataLoader
│   ├── train_model.py             # Training script with mixed precision
│   └── config_train.yaml          # Training configuration
│
├── Phase 3 - Evaluation/
│   ├── evaluate_model.py          # Model evaluation & metrics
│   └── plot_regime_distribution.py # Visualization tools
│
├── Phase 4 - Realtime Engine/
│   ├── natron_server_v5.py       # Python trading server
│   ├── realtime_socket.py         # Socket client for data feed
│   ├── natron_ea.mq5              # MetaTrader 5 Expert Advisor
│   └── realtime_config.yaml       # Realtime configuration
│
├── Phase 5 - Feedback Learning/
│   ├── auto_feedback_engine.py   # Adaptive weight adjustment
│   ├── log_analyzer.py            # Trade outcome analysis
│   └── adaptive_config.yaml       # Adaptive configuration
│
├── Phase 6 - Deployment/
│   ├── Dockerfile                 # Docker container
│   ├── start_natron.sh            # Auto-start script
│   ├── monitor_natron.py          # System monitoring
│   └── requirements.txt            # Python dependencies
│
└── README_Natron_EndToEnd.md      # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (T4 / A100 recommended) for training
- MetaTrader 5 (for live trading)
- Docker (optional, for containerized deployment)

### Installation

1. **Clone and setup:**
```bash
cd natron
pip install -r requirements.txt
```

2. **Prepare data:**
```bash
# Place your OHLCV CSV file (columns: time, open, high, low, close, volume)
python data_pipeline.py data/raw_data.csv data/processed_data.csv
```

3. **Train model:**
```bash
python train_model.py config_train.yaml
```

4. **Evaluate:**
```bash
python evaluate_model.py config_train.yaml checkpoints/natron_v6.pt
python plot_regime_distribution.py
```

---

## 📊 Phase-by-Phase Guide

### Phase 1: Data Pipeline

**Goal:** Preprocess OHLCV data and create 50-70 features.

```bash
python data_pipeline.py input.csv processed_data.csv
```

**Output:**
- `processed_data.csv` with features and regime labels
- Features include: ATR, EMA(20,50,200), RSI, MACD, Bollinger Bands, candle patterns, etc.
- Regime labels: BULL_STRONG, BULL_WEAK, BEAR_STRONG, BEAR_WEAK, RANGE, VOLATILE

---

### Phase 2: Model Training

**Goal:** Train multi-head transformer with 3 output heads.

```bash
python train_model.py config_train.yaml
```

**Architecture:**
- Input: 96 candles × features
- Transformer Encoder (6 layers, 8 heads, d_model=256)
- Outputs:
  - Regime classification (6 classes)
  - Context strength (1 value: |bull_p - bear_p|)
  - Forecast direction (2 classes: up/down)

**Training features:**
- Mixed precision (FP16) for GPU acceleration
- Cosine annealing LR scheduler
- Weighted loss function
- Best model saved as `checkpoints/natron_v6.pt`

---

### Phase 3: Evaluation

**Goal:** Evaluate model and generate reports.

```bash
python evaluate_model.py config_train.yaml checkpoints/natron_v6.pt
python plot_regime_distribution.py
```

**Outputs:**
- `natron_predictions.csv` - All predictions
- `metrics_report.json` - Accuracy, F1, precision/recall
- `regime_distribution.png` - Visualizations

---

### Phase 4: Realtime Execution

**Goal:** Deploy live trading engine.

**Setup:**

1. **Python Server:**
```bash
python natron_server_v5.py realtime_config.yaml
```

2. **MetaTrader 5 EA:**
- Compile `natron_ea.mq5` in MT5
- Attach to chart
- Configure socket port (default: 8888)

**Features:**
- Real-time candle processing
- Entry/exit signal generation
- Stop loss & take profit calculation
- Trailing stop management
- Event logging to `realtime_events.csv`

---

### Phase 5: Feedback Learning

**Goal:** Learn from trading outcomes and adapt.

```bash
python auto_feedback_engine.py adapt
```

**Mechanism:**
- Analyzes `realtime_events.csv` vs `natron_predictions.csv`
- Updates `adaptive_config.yaml` weights:
  - Confidence scaling
  - Entry thresholds
  - Regime-specific weights
- Generates learning curve

---

### Phase 6: Deployment

**Goal:** Production deployment with monitoring.

**Docker:**
```bash
docker build -t natron:latest .
docker run -d --name natron --restart unless-stopped -p 8888:8888 natron:latest
```

**Manual:**
```bash
chmod +x start_natron.sh
./start_natron.sh  # Auto-restart on failure
```

**Monitoring:**
```bash
python monitor_natron.py monitor_config.yaml
```

**Features:**
- Auto-restart on crash
- Daily summary reports
- Telegram alerts (optional)
- System health monitoring

---

## ⚙️ Configuration Files

### `config_train.yaml`
Training hyperparameters, model architecture, data paths.

### `realtime_config.yaml`
Trading parameters, socket settings, risk management.

### `adaptive_config.yaml`
Auto-updated by feedback engine. Confidence scaling, regime weights.

---

## 📈 Example Training Run Log

```
2024-01-15 10:00:00 - INFO - DataPipeline initialized
2024-01-15 10:00:05 - INFO - Loaded 10000 candles
2024-01-15 10:00:10 - INFO - Engineered 65 features
2024-01-15 10:00:15 - INFO - Regime distribution:
BULL_STRONG    1500
BULL_WEAK      1200
BEAR_STRONG    1300
BEAR_WEAK      1100
RANGE          2800
VOLATILE       2100

2024-01-15 10:05:00 - INFO - Starting training for 100 epochs...
Epoch 1/100 | Train Loss: 2.3456 | Val Loss: 2.1234 | LR: 0.000100
Epoch 2/100 | Train Loss: 1.9876 | Val Loss: 1.8765 | LR: 0.000099
...
Epoch 50/100 | Train Loss: 0.5432 | Val Loss: 0.6123 | LR: 0.000050
...
Epoch 100/100 | Train Loss: 0.3210 | Val Loss: 0.3456 | LR: 0.000010
2024-01-15 12:30:00 - INFO - Training completed!
2024-01-15 12:30:00 - INFO - Best validation loss: 0.3456
```

---

## 🔧 Running on Google Colab / Vertex AI

### Google Colab (Free GPU)

1. **Upload project:**
```python
from google.colab import files
files.upload()  # Upload natron folder
```

2. **Install dependencies:**
```bash
!pip install -r natron/requirements.txt
```

3. **Run training:**
```bash
!cd natron && python train_model.py config_train.yaml
```

### Vertex AI Studio

1. **Create custom container:**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/natron
```

2. **Submit training job:**
```bash
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name="natron-training" \
  --config=vertex_config.yaml
```

---

## 🗺️ Roadmap: G9–G13 Features

### G9: Forecast Fusion
- Ensemble multiple forecast horizons (5, 10, 20 candles)
- Weighted fusion based on regime confidence

### G10: Order Flow Analysis
- Level 2 order book integration
- Volume profile and market microstructure

### G11: Sentiment Integration
- News sentiment analysis
- Social media sentiment (Twitter, Reddit)
- Economic calendar events

### G12: Strategy Selector
- Multi-strategy ensemble
- Regime-aware strategy switching
- Performance-based strategy weighting

### G13: Reinforcement Learning
- RL agent for position sizing
- Adaptive risk management
- Multi-objective optimization (Sharpe, Sortino, Calmar)

---

## 📝 Notes

- **Backtesting:** Use historical data with `data_pipeline.py` → `train_model.py` → `evaluate_model.py`
- **Paper Trading:** Run `natron_server_v5.py` with demo account
- **Live Trading:** Ensure proper risk management and start with small position sizes
- **Monitoring:** Always run `monitor_natron.py` in production

---

## ⚠️ Disclaimer

This system is for educational and research purposes. Trading involves risk. Always:
- Test thoroughly on historical data
- Start with paper trading
- Use proper risk management
- Monitor system continuously
- Never risk more than you can afford to lose

---

## 📧 Support

For issues, questions, or contributions, please refer to the project repository.

---

**Natron V1.0 – Galaxy-class**  
*Advanced AI Trading System*
