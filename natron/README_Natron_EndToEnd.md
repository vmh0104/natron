# Natron End-to-End AI Trading System

**Version 1.0 - Galaxy-class**

An advanced AI trading system inspired by QuantConnect, DeepMind AlphaTrade, and MetaTrader MQL5 integration. Natron combines multi-head transformer models with realtime execution and adaptive learning capabilities.

---

## 🏗️ Architecture Overview

Natron consists of 6 integrated phases:

1. **Data Pipeline** - Preprocesses OHLCV data and computes 50-70 technical features
2. **Model Training** - Trains multi-head transformer with regime, context, and forecast heads
3. **Evaluation** - Evaluates model performance and generates reports
4. **Realtime Execution** - Executes trades via MQL5/MT5 integration
5. **Feedback Learning** - Adapts configuration based on trading performance
6. **Deployment** - Dockerized deployment with monitoring

---

## 📁 Project Structure

```
natron/
├── configs/                    # Configuration files
│   ├── config_train.yaml       # Training configuration
│   ├── realtime_config.yaml    # Realtime trading configuration
│   └── adaptive_config.yaml    # Adaptive learning configuration
│
├── data/                       # Data directory
│   ├── raw_data.csv           # Input OHLCV data
│   └── processed_data.csv     # Processed features
│
├── models/                     # Trained models
│   ├── natron_v6.pt           # Model checkpoint
│   └── scaler.pkl             # Feature scaler
│
├── logs/                       # Logs and outputs
│   ├── training_history.json
│   ├── natron_predictions.csv
│   ├── realtime_events.csv
│   └── learning_curve.json
│
├── mql5/                       # MetaTrader 5 Expert Advisor
│   └── natron_ea.mq5
│
├── scripts/                    # Utility scripts
│   ├── start_natron.sh        # Startup script
│   └── monitor_natron.py      # Monitoring script
│
├── src/                        # Source code
│   ├── data_pipeline/         # Phase 1: Data preprocessing
│   │   ├── data_pipeline.py
│   │   ├── feature_engineering.py
│   │   └── regime_labeler.py
│   │
│   ├── training/              # Phase 2: Model training
│   │   ├── model_natron.py
│   │   ├── dataset_loader.py
│   │   └── train_model.py
│   │
│   ├── evaluation/            # Phase 3: Evaluation
│   │   ├── evaluate_model.py
│   │   └── plot_regime_distribution.py
│   │
│   ├── realtime/              # Phase 4: Realtime execution
│   │   ├── natron_server_v5.py
│   │   └── realtime_socket.py
│   │
│   ├── feedback/              # Phase 5: Feedback learning
│   │   ├── log_analyzer.py
│   │   └── auto_feedback_engine.py
│   │
│   └── deployment/            # Phase 6: Deployment utilities
│
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker container definition
└── README_Natron_EndToEnd.md # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- MetaTrader 5 (for live trading)
- Docker (optional, for containerized deployment)

### Installation

1. **Clone and setup:**
```bash
cd natron
pip install -r requirements.txt
```

2. **Prepare data:**
   - Place your OHLCV CSV file in `data/raw_data.csv`
   - Format: `time,open,high,low,close,volume`

3. **Run Phase 1 - Data Pipeline:**
```bash
python -m src.data_pipeline.data_pipeline \
    --input data/raw_data.csv \
    --output data/processed_data.csv \
    --config configs/config_train.yaml
```

4. **Run Phase 2 - Model Training:**
```bash
python -m src.training.train_model --config configs/config_train.yaml
```

5. **Run Phase 3 - Evaluation:**
```bash
python -m src.evaluation.evaluate_model \
    --model models/natron_v6.pt \
    --data data/processed_data.csv \
    --output logs/natron_predictions.csv \
    --report logs/metrics_report.json
```

6. **Run Phase 4 - Realtime Server:**
```bash
python -m src.realtime.natron_server_v5 --config configs/realtime_config.yaml
```

7. **Run Phase 5 - Feedback Learning:**
```bash
python -m src.feedback.auto_feedback_engine \
    --config configs/adaptive_config.yaml \
    --predictions logs/natron_predictions.csv \
    --events logs/realtime_events.csv
```

---

## 📊 Running on Google Colab / Vertex AI Studio

### Google Colab Setup

1. **Upload project:**
```python
from google.colab import files
import zipfile
# Upload natron.zip, then:
with zipfile.ZipFile('natron.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
```

2. **Install dependencies:**
```bash
!pip install -r natron/requirements.txt
```

3. **Enable GPU:**
   - Runtime → Change runtime type → GPU (T4 or A100)

4. **Run training:**
```python
import sys
sys.path.append('natron')

from src.training.train_model import NatronTrainer

trainer = NatronTrainer('natron/configs/config_train.yaml')
trainer.train()
```

### Vertex AI Studio Setup

1. **Create Vertex AI Workbench instance:**
   - Use GPU-enabled instance (T4 or A100)
   - Python 3.10 environment

2. **Clone repository:**
```bash
git clone <your-repo-url>
cd natron
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run training with Vertex AI:**
```bash
python -m src.training.train_model --config configs/config_train.yaml
```

---

## ⚙️ Configuration

### Training Configuration (`configs/config_train.yaml`)

Key parameters:
- `sequence_length`: 96 (candles per sequence)
- `d_model`: 256 (transformer embedding dimension)
- `nhead`: 8 (attention heads)
- `num_layers`: 6 (transformer layers)
- `epochs`: 100
- `batch_size`: 32

### Realtime Configuration (`configs/realtime_config.yaml`)

Key parameters:
- `entry_confidence_threshold`: 0.65
- `risk_per_trade`: 0.02 (2%)
- `stop_loss_atr_multiplier`: 2.0
- `take_profit_atr_multiplier`: 3.0

---

## 📈 Example Training Run Log

```
2024-01-15 10:00:00 - INFO - ============================================================
2024-01-15 10:00:00 - INFO - NATRON DATA PIPELINE - Starting
2024-01-15 10:00:00 - INFO - ============================================================
2024-01-15 10:00:01 - INFO - Loading data from data/raw_data.csv
2024-01-15 10:00:02 - INFO - Loaded 100000 candles
2024-01-15 10:00:02 - INFO - Starting feature engineering...
2024-01-15 10:00:15 - INFO - Engineered 68 features
2024-01-15 10:00:15 - INFO - Labeling regimes...
2024-01-15 10:00:18 - INFO - Regime distribution:
2024-01-15 10:00:18 - INFO -   BULL_STRONG: 15234 (15.23%)
2024-01-15 10:00:18 - INFO -   BULL_WEAK: 18456 (18.46%)
2024-01-15 10:00:18 - INFO -   BEAR_STRONG: 14234 (14.23%)
2024-01-15 10:00:18 - INFO -   BEAR_WEAK: 17543 (17.54%)
2024-01-15 10:00:18 - INFO -   RANGE: 22345 (22.35%)
2024-01-15 10:00:18 - INFO -   VOLATILE: 12188 (12.19%)
2024-01-15 10:00:19 - INFO - Saved 100000 rows to data/processed_data.csv
2024-01-15 10:00:19 - INFO - ============================================================
2024-01-15 10:00:19 - INFO - NATRON DATA PIPELINE - Complete
2024-01-15 10:00:19 - INFO - ============================================================

2024-01-15 10:00:20 - INFO - ============================================================
2024-01-15 10:00:20 - INFO - NATRON MODEL TRAINING - Starting
2024-01-15 10:00:20 - INFO - ============================================================
2024-01-15 10:00:21 - INFO - Using device: cuda
2024-01-15 10:00:21 - INFO - Loading processed data from data/processed_data.csv
2024-01-15 10:00:23 - INFO - Train: 70000, Val: 15000, Test: 15000
2024-01-15 10:00:25 - INFO - Input dimension: 68
2024-01-15 10:00:26 - INFO - Model parameters: 2,456,789
2024-01-15 10:00:26 - INFO - 
Epoch 1/100
Epoch 1: 100%|████████| 2188/2188 [02:15<00:00, 16.15it/s, loss=1.234]
Validating: 100%|████████| 469/469 [00:28<00:00, 16.45it/s]
2024-01-15 10:02:49 - INFO - Train Loss: 1.2345 | Val Loss: 1.1892
2024-01-15 10:02:49 - INFO -   Regime: 0.4523 / 0.4389
2024-01-15 10:02:49 - INFO -   Context: 0.1234 / 0.1198
2024-01-15 10:02:49 - INFO -   Forecast: 0.6588 / 0.6305
2024-01-15 10:02:49 - INFO - Saved best model to models/natron_v6.pt
...
```

---

## 🐳 Docker Deployment

### Build Docker Image

```bash
docker build -t natron:1.0 .
```

### Run Container

```bash
docker run -d \
    --name natron-server \
    --gpus all \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/logs:/app/logs \
    -v $(pwd)/configs:/app/configs \
    -p 8888:8888 \
    natron:1.0
```

### View Logs

```bash
docker logs -f natron-server
```

---

## 📊 Monitoring

### Start Monitor

```bash
python scripts/monitor_natron.py \
    --log-dir logs \
    --telegram-token YOUR_TOKEN \
    --telegram-chat-id YOUR_CHAT_ID \
    --interval 60
```

### Daily Summary

The monitor automatically sends daily summaries via Telegram (if configured) or logs them to console.

---

## 🔮 Roadmap: G9-G13 Features

### G9 - Forecast Fusion
- Ensemble multiple forecast models
- Weighted voting based on regime
- Confidence calibration

### G10 - Order Flow Analysis
- Level 2 order book integration
- Volume profile analysis
- Market microstructure features

### G11 - Sentiment Analysis
- News sentiment integration
- Social media sentiment
- Economic calendar events

### G12 - Strategy Selector
- Multi-strategy ensemble
- Dynamic strategy allocation
- Regime-based strategy switching

### G13 - Advanced Risk Management
- Portfolio-level risk controls
- Correlation analysis
- Dynamic position sizing

---

## 📝 Example YAML Configuration

### `configs/config_train.yaml` (excerpt)

```yaml
data:
  sequence_length: 96
  batch_size: 32
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

model:
  d_model: 256
  nhead: 8
  num_layers: 6
  dim_feedforward: 1024
  dropout: 0.1
  regime_classes: 6
  forecast_classes: 2

training:
  epochs: 100
  learning_rate: 0.0001
  loss_weights:
    regime: 1.0
    context: 0.5
    forecast: 1.5
```

---

## 🛠️ Troubleshooting

### Common Issues

1. **CUDA out of memory:**
   - Reduce `batch_size` in config
   - Use gradient accumulation
   - Enable mixed precision training

2. **MT5 connection failed:**
   - Check MT5 is running
   - Verify login credentials in `realtime_config.yaml`
   - Check firewall settings

3. **Feature engineering errors:**
   - Ensure input CSV has required columns
   - Check for NaN values in data
   - Verify data types

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

## 🙏 Acknowledgments

Inspired by:
- QuantConnect - Algorithmic trading platform
- DeepMind AlphaTrade - Reinforcement learning for trading
- MetaTrader MQL5 - Trading platform integration

---

**Natron V1.0 - Galaxy-class**  
*"Trading at the speed of light"*
