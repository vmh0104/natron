# Natron End-to-End AI Trading System

**Version:** Natron V1.0 – Galaxy-class

An advanced AI trading system inspired by QuantConnect, DeepMind AlphaTrade, and MetaTrader MQL5 integration. Natron combines multi-head transformer models with realtime execution and self-learning capabilities.

---

## 🏗️ Architecture Overview

Natron consists of 6 integrated phases:

1. **Data Pipeline** - Preprocesses OHLCV data and computes 50-70 technical features
2. **Model Training** - Multi-head transformer with regime, context, and forecast heads
3. **Evaluation** - Comprehensive metrics and visualization
4. **Realtime Execution** - Python server + MQL5 integration for live trading
5. **Feedback Learning** - Self-adaptation from trading outcomes
6. **Deployment** - Docker containerization and monitoring

---

## 📁 Project Structure

```
natron/
├── data/
│   ├── raw/                    # Raw market data CSV files
│   └── processed/              # Processed data ready for ML
├── models/                     # Trained model checkpoints
├── configs/                    # YAML configuration files
│   ├── config_train.yaml       # Training configuration
│   ├── realtime_config.yaml    # Realtime trading config
│   └── adaptive_config.yaml    # Auto-adapted config
├── logs/                       # Training logs, events, predictions
│   ├── training/               # Training history
│   ├── evaluation/             # Evaluation results
│   └── realtime_events.csv     # Trading events log
├── src/
│   ├── data_pipeline/          # Phase 1: Data preprocessing
│   │   ├── data_pipeline.py
│   │   ├── feature_engineering.py
│   │   └── regime_labeler.py
│   ├── training/               # Phase 2: Model training
│   │   ├── train_model.py
│   │   ├── model_natron.py
│   │   └── dataset_loader.py
│   ├── evaluation/             # Phase 3: Evaluation
│   │   ├── evaluate_model.py
│   │   └── plot_regime_distribution.py
│   ├── realtime/               # Phase 4: Realtime engine
│   │   ├── natron_server_v5.py
│   │   ├── realtime_socket.py
│   │   └── natron_ea.mq5
│   ├── feedback/               # Phase 5: Feedback learning
│   │   ├── auto_feedback_engine.py
│   │   └── log_analyzer.py
│   └── deployment/            # Phase 6: Deployment
│       └── monitor_natron.py
├── scripts/
│   └── start_natron.sh         # Startup script
├── Dockerfile                  # Docker container definition
├── requirements.txt            # Python dependencies
└── README_Natron_EndToEnd.md   # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: T4 / A100) for training
- MetaTrader 5 (for realtime trading)
- Docker (optional, for containerized deployment)

### Installation

1. **Clone and setup:**
```bash
cd natron
pip install -r requirements.txt
```

2. **Prepare data:**
   - Place your OHLCV CSV file in `data/raw/market_data.csv`
   - Format: `time,open,high,low,close,volume`

3. **Run Phase 1 - Data Pipeline:**
```bash
python -m src.data_pipeline.data_pipeline
```

4. **Run Phase 2 - Train Model:**
```bash
python -m src.training.train_model configs/config_train.yaml
```

5. **Run Phase 3 - Evaluate:**
```bash
python -m src.evaluation.evaluate_model models/natron_v6.pt configs/config_train.yaml logs/evaluation
```

---

## 📖 Detailed Phase Documentation

### Phase 1: Data Pipeline

**Purpose:** Preprocess raw OHLCV data into ML-ready features.

**Features Computed:**
- Technical indicators: ATR, EMA(20,50,200), RSI, MACD, Bollinger Bands
- Candle features: body_pct, wick_ratio, direction
- Volatility features: ATR percentile, price/volume volatility
- Momentum features: ROC, momentum, price position
- Trend features: EMA slopes, crossovers, trend strength
- Volume features: OBV, VPT, volume ratios

**Regime Labeling:**
- 6 classes: BULL_STRONG, BULL_WEAK, BEAR_STRONG, BEAR_WEAK, RANGE, VOLATILE
- Based on trend slope, ATR percentile, and breakout logic

**Usage:**
```python
from src.data_pipeline.data_pipeline import DataPipeline

pipeline = DataPipeline()
processed_df = pipeline.process(
    input_file="data/raw/market_data.csv",
    output_file="data/processed/processed_data.csv",
    create_target=True,
    forecast_horizon=5
)
```

---

### Phase 2: Model Training

**Architecture:**
- **Input:** Sequence of 96 candles × features
- **Model:** Transformer Encoder (6 layers, 8 heads, 256 dim)
- **Heads:**
  - `regime_head`: 6-class classification
  - `context_head`: Context strength (|bull_p - bear_p|)
  - `forecast_head`: 2-class direction prediction

**Training Features:**
- Mixed precision training (FP16)
- AdamW optimizer + Cosine LR scheduler
- Weighted loss function
- GPU acceleration

**Usage:**
```bash
python -m src.training.train_model configs/config_train.yaml
```

**Output:**
- Model checkpoint: `models/natron_v6.pt`
- Training history: `logs/training/training_history.json`

---

### Phase 3: Evaluation & Analysis

**Metrics Computed:**
- Regime classification: Accuracy, F1, precision/recall per class
- Forecast direction: Accuracy, confusion matrix
- Context strength: MSE, MAE, R²

**Visualizations:**
- Regime distribution over time
- Regime statistics and transitions
- Forecast probability distributions

**Usage:**
```bash
python -m src.evaluation.evaluate_model \
    models/natron_v6.pt \
    configs/config_train.yaml \
    logs/evaluation

python -m src.evaluation.plot_regime_distribution \
    logs/evaluation/natron_predictions.csv \
    logs/evaluation
```

**Output:**
- `logs/evaluation/metrics_report.json`
- `logs/evaluation/natron_predictions.csv`
- `logs/evaluation/regime_distribution.png`
- `logs/evaluation/regime_statistics.png`

---

### Phase 4: Realtime Execution Engine

**Components:**
1. **Python Server** (`natron_server_v5.py`): Main trading engine
2. **Socket Communication** (`realtime_socket.py`): MQL5 bridge
3. **MQL5 EA** (`natron_ea.mq5`): MetaTrader Expert Advisor

**Trading Logic:**
- Entry: Based on forecast probability + context strength + regime
- Exit: Stop loss, take profit, trailing stop
- Risk management: ATR-based position sizing

**Usage:**

1. **Start Python server:**
```bash
python -m src.realtime.natron_server_v5 configs/realtime_config.yaml
```

2. **Load EA in MetaTrader 5:**
   - Compile `src/realtime/natron_ea.mq5`
   - Attach to chart
   - Configure server host/port

3. **Monitor events:**
   - Events logged to `logs/realtime_events.csv`

---

### Phase 5: Feedback Learning

**Mechanism:**
- Analyzes `natron_predictions.csv` + `realtime_events.csv`
- Compares predicted vs actual outcomes
- Adjusts confidence thresholds, context requirements
- Updates regime-specific parameters

**Usage:**
```bash
python -m src.feedback.auto_feedback_engine \
    logs/evaluation/natron_predictions.csv \
    logs/realtime_events.csv \
    configs/realtime_config.yaml \
    configs/adaptive_config.yaml
```

**Output:**
- Updated config: `configs/adaptive_config.yaml`
- Learning curve: `configs/learning_curve.json`

---

### Phase 6: Deployment

**Docker Deployment:**
```bash
# Build image
docker build -t natron:latest .

# Run container
docker run -d \
    --name natron \
    -p 8888:8888 \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    natron:latest
```

**Monitoring:**
```bash
python -m src.deployment.monitor_natron localhost 8888 60
```

**Startup Script:**
```bash
./scripts/start_natron.sh
```

---

## 🧪 Running on Google Colab / Vertex AI Studio

### Google Colab Setup

1. **Upload project:**
```python
from google.colab import files
# Upload natron.zip
!unzip natron.zip
```

2. **Install dependencies:**
```python
!pip install -r natron/requirements.txt
```

3. **Upload data:**
```python
# Upload CSV to natron/data/raw/market_data.csv
```

4. **Run phases:**
```python
# Phase 1
!cd natron && python -m src.data_pipeline.data_pipeline

# Phase 2 (training)
!cd natron && python -m src.training.train_model configs/config_train.yaml

# Phase 3 (evaluation)
!cd natron && python -m src.evaluation.evaluate_model models/natron_v6.pt configs/config_train.yaml logs/evaluation
```

### Vertex AI Studio Setup

1. **Create custom training job:**
```bash
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name="natron-training" \
    --config=vertex_config.yaml
```

2. **Deploy model endpoint:**
```bash
gcloud ai models deploy natron-model \
    --model-dir=gs://your-bucket/models/natron_v6.pt \
    --region=us-central1
```

---

## 📊 Example Training Run Log

```
2024-01-15 10:00:00 - INFO - Loading data from data/processed/processed_data.csv
2024-01-15 10:00:05 - INFO - Found 67 features
2024-01-15 10:00:05 - INFO - Data split: Train=7000, Val=1500, Test=1500
2024-01-15 10:00:10 - INFO - Building model...
2024-01-15 10:00:10 - INFO - Model parameters: 2,847,392
2024-01-15 10:00:15 - INFO - Starting training...
Epoch 1/100 | Train Loss: 2.3456 | Val Loss: 2.1234 | Regime Acc: 0.4523 | Forecast Acc: 0.5234 | LR: 0.000100
Epoch 2/100 | Train Loss: 1.9876 | Val Loss: 1.8765 | Regime Acc: 0.5234 | Forecast Acc: 0.5678 | LR: 0.000099
...
Epoch 50/100 | Train Loss: 0.8234 | Val Loss: 0.9123 | Regime Acc: 0.7234 | Forecast Acc: 0.6789 | LR: 0.000050
...
Epoch 100/100 | Train Loss: 0.6543 | Val Loss: 0.7890 | Regime Acc: 0.7890 | Forecast Acc: 0.7234 | LR: 0.000010
2024-01-15 12:30:00 - INFO - Training complete!
2024-01-15 12:30:00 - INFO - Best validation loss: 0.7890
```

---

## ⚙️ Configuration Examples

### Training Config (`config_train.yaml`)
```yaml
model:
  d_model: 256
  nhead: 8
  num_layers: 6
  seq_len: 96

training:
  batch_size: 32
  num_epochs: 100
  optimizer:
    lr: 0.0001
```

### Realtime Config (`realtime_config.yaml`)
```yaml
trading:
  min_confidence: 0.65
  min_context_strength: 0.5
  stop_loss_atr_multiple: 2.0
  take_profit_atr_multiple: 3.0
```

---

## 🗺️ Roadmap: G9–G13 Enhancements

### G9: Forecast Fusion
- Combine multiple forecast horizons
- Ensemble predictions from different models
- Confidence-weighted aggregation

### G10: Order Flow Analysis
- Level 2 order book integration
- Market microstructure features
- Flow imbalance indicators

### G11: Sentiment Integration
- News sentiment analysis
- Social media signals
- Economic calendar events

### G12: Strategy Selector
- Multi-strategy ensemble
- Regime-aware strategy switching
- Performance-based allocation

### G13: Advanced Risk Management
- Portfolio-level risk controls
- Dynamic position sizing
- Correlation-aware hedging

---

## 📝 License

This project is provided as-is for educational and research purposes.

---

## 🤝 Contributing

Contributions welcome! Please follow PEP8 style guidelines and include docstrings.

---

## 📧 Support

For issues and questions, please open an issue on the project repository.

---

**Natron V1.0 – Galaxy-class**  
*Advanced AI Trading System*
