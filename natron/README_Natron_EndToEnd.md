# Natron End-to-End AI Trading System

**Version: Natron V1.0 – Galaxy-class**

An advanced AI trading system inspired by QuantConnect, DeepMind AlphaTrade, and MetaTrader MQL5 integration. Natron combines multi-head transformer models with realtime execution and adaptive learning capabilities.

---

## 🏗️ Architecture Overview

Natron consists of 6 integrated phases:

1. **Data Pipeline** - Preprocesses OHLCV data and computes 50-70 technical features
2. **Model Training** - Trains multi-head transformer with regime, context, and forecast heads
3. **Evaluation** - Evaluates model performance and generates reports
4. **Realtime Engine** - Executes trades via MQL5/MT5 integration
5. **Feedback Learning** - Adapts configuration based on trading outcomes
6. **Deployment** - Docker containerization and monitoring

---

## 📁 Project Structure

```
natron/
├── data_pipeline.py              # Main data preprocessing pipeline
├── feature_engineering.py         # Technical indicator computation
├── regime_labeler.py             # Market regime labeling (6 classes)
├── model_natron.py               # Multi-head transformer model
├── dataset_loader.py             # PyTorch dataset and dataloader
├── train_model.py                # Model training script
├── evaluate_model.py             # Model evaluation script
├── plot_regime_distribution.py   # Visualization tools
├── natron_server_v5.py           # Realtime trading server
├── realtime_socket.py            # Socket communication module
├── natron_ea.mq5                 # MetaTrader 5 Expert Advisor
├── log_analyzer.py               # Trading log analysis
├── auto_feedback_engine.py       # Adaptive learning engine
├── monitor_natron.py             # System monitoring
├── start_natron.sh               # Startup script with auto-restart
├── Dockerfile                    # Docker container definition
├── config_train.yaml             # Training configuration
├── realtime_config.yaml          # Realtime trading configuration
├── adaptive_config.yaml          # Adaptive configuration (auto-generated)
├── requirements.txt              # Python dependencies
└── README_Natron_EndToEnd.md     # This file

data/
├── processed/
│   └── processed_data.csv        # Preprocessed data with features

models/
└── natron_v6/
    ├── natron_v6.pt              # Trained model checkpoint
    ├── checkpoint_latest.pt      # Latest training checkpoint
    ├── training_history.json     # Training metrics history
    └── norm_stats.npy            # Feature normalization statistics

logs/
├── natron.log                    # Server log
└── natron.pid                    # Process ID file

plots/
├── regime_distribution.png       # Regime distribution visualization
└── regime_transitions.png        # Regime transition matrix

feedback_output/
├── adaptive_config.yaml          # Adapted configuration
├── learning_history.json         # Learning history
└── feedback_analysis_report.json # Analysis report
```

---

## 🚀 Quick Start Guide

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: T4, A100, or similar)
- MetaTrader 5 (for live trading)
- Docker (optional, for containerized deployment)

### Installation

1. **Clone and setup environment:**
```bash
cd natron
pip install -r requirements.txt
```

2. **Prepare data:**
   - Place your OHLCV CSV file in the project root
   - CSV should have columns: `time`, `open`, `high`, `low`, `close`, `volume`

3. **Run Phase 1 - Data Pipeline:**
```bash
python data_pipeline.py data/raw/market_data.csv data/processed/processed_data.csv
```

4. **Run Phase 2 - Model Training:**
```bash
python train_model.py --config config_train.yaml --data data/processed/processed_data.csv
```

5. **Run Phase 3 - Evaluation:**
```bash
python evaluate_model.py --model models/natron_v6/natron_v6.pt --config config_train.yaml --data data/processed/processed_data.csv
python plot_regime_distribution.py --data data/processed/processed_data.csv --output plots
```

---

## 📖 Detailed Phase Instructions

### Phase 1: Data Pipeline

**Goal:** Preprocess raw OHLCV data into ML-ready format.

**Input:** CSV file with columns: `time`, `open`, `high`, `low`, `close`, `volume`

**Output:** `processed_data.csv` with 50-70 features + regime labels

**Usage:**
```bash
python data_pipeline.py <input_csv> [output_csv]
```

**Features computed:**
- ATR, EMA(20,50,200), RSI, MACD, Bollinger Bands
- Candle body/wick ratios, volatility metrics
- Trend slopes, momentum indicators, volume features
- Regime labels: BULL_STRONG, BULL_WEAK, BEAR_STRONG, BEAR_WEAK, RANGE, VOLATILE

---

### Phase 2: Model Training

**Goal:** Train multi-head transformer model.

**Architecture:**
- Input: 96 candles × features
- Transformer Encoder: 6 layers, 8 heads, 128 dim
- Outputs:
  - Regime classification (6 classes)
  - Context strength (0-1)
  - Forecast direction (up/down)

**Usage:**
```bash
python train_model.py --config config_train.yaml --data data/processed/processed_data.csv
```

**Configuration (`config_train.yaml`):**
- Model architecture parameters
- Optimizer settings (AdamW, lr=1e-4)
- Loss weights (regime, context, forecast)
- Training epochs, batch size, mixed precision

**Output:**
- `models/natron_v6/natron_v6.pt` - Best model checkpoint
- `models/natron_v6/training_history.json` - Training metrics

---

### Phase 3: Evaluation & Analysis

**Goal:** Evaluate model performance and visualize results.

**Usage:**
```bash
# Evaluate on test set
python evaluate_model.py \
    --model models/natron_v6/natron_v6.pt \
    --config config_train.yaml \
    --data data/processed/processed_data.csv \
    --split test \
    --output evaluation_results

# Plot regime distributions
python plot_regime_distribution.py \
    --data data/processed/processed_data.csv \
    --output plots
```

**Outputs:**
- `evaluation_results/metrics_report.json` - Accuracy, F1, precision/recall
- `evaluation_results/natron_predictions.csv` - Predictions vs targets
- `plots/regime_distribution.png` - Regime visualization over time
- `plots/regime_transitions.png` - Transition matrix heatmap

---

### Phase 4: Realtime Execution Engine

**Goal:** Execute trades in realtime via MQL5 integration.

**Components:**
1. **Python Server** (`natron_server_v5.py`) - Processes predictions and sends signals
2. **MQL5 EA** (`natron_ea.mq5`) - Receives signals and executes trades in MT5

**Setup:**

1. **Start Python server:**
```bash
python natron_server_v5.py --config realtime_config.yaml
```

2. **Install MQL5 EA:**
   - Copy `natron_ea.mq5` to MetaTrader 5 `MQL5/Experts/` directory
   - Compile in MetaEditor
   - Attach to chart in MT5

3. **Configure:**
   - Edit `realtime_config.yaml` with model path and trading parameters
   - Set socket port (default: 8888)
   - Adjust entry/cancel thresholds

**Trading Logic:**
- Entry: Forecast confidence > threshold + context strength > 0.3
- Entry zone: Limit orders at ATR-based offsets
- Cancel: ATR spike, confidence drop, or context weakness
- Trailing stop: 2x ATR trailing stop

**Logging:**
- All events saved to `realtime_events.csv`
- Entry signals, trade executions, cancellations

---

### Phase 5: Feedback Learning & Adaptation

**Goal:** Learn from trading outcomes and adapt configuration.

**Usage:**
```bash
python auto_feedback_engine.py \
    --config realtime_config.yaml \
    --predictions natron_predictions.csv \
    --events realtime_events.csv \
    --output feedback_output
```

**Process:**
1. Analyzes prediction accuracy vs actual outcomes
2. Computes trade performance metrics (win rate, profit factor)
3. Adjusts entry/cancel thresholds dynamically
4. Updates `adaptive_config.yaml`

**Adaptation Rules:**
- Low win rate → Increase entry threshold (more selective)
- Low profit factor → Increase cancel threshold (exit faster)
- High accuracy → Scale confidence thresholds up
- Low accuracy → Scale confidence thresholds down

**Outputs:**
- `feedback_output/adaptive_config.yaml` - Updated configuration
- `feedback_output/learning_history.json` - Adaptation history
- `feedback_output/feedback_analysis_report.json` - Performance analysis

---

### Phase 6: Deployment & Monitoring

**Goal:** Deploy as autonomous system with monitoring.

**Docker Deployment:**
```bash
# Build image
docker build -t natron:v1.0 .

# Run container
docker run -d \
    --name natron \
    --restart unless-stopped \
    -p 8888:8888 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/logs:/app/logs \
    natron:v1.0
```

**Standalone Deployment:**
```bash
# Make startup script executable
chmod +x start_natron.sh

# Run with auto-restart
./start_natron.sh
```

**Monitoring:**
```bash
python monitor_natron.py --config monitoring_config.yaml
```

**Features:**
- Auto-restart on failure (max 10 restarts)
- System health monitoring (CPU, memory, disk)
- Daily summary reports
- Optional Telegram alerts

---

## 🔧 Configuration Files

### `config_train.yaml`
Training configuration: model architecture, optimizer, loss weights, data splits.

### `realtime_config.yaml`
Realtime trading configuration: model path, socket settings, trading parameters, risk management.

### `adaptive_config.yaml`
Auto-generated adaptive configuration (updated by feedback engine).

---

## 📊 Example Training Run Log

```
============================================================
NATRON DATA PIPELINE - Processing Started
============================================================

[1/4] Loading raw data...
Loaded 10000 candles from data/raw/market_data.csv
[2/4] Standardizing data...
   Cleaned dataset: 9985 candles
[3/4] Computing features (50-70 features)...
   Computed 65 features
[4/4] Labeling regimes...

=== Regime Distribution ===
BULL_STRONG    :   1245 (12.47%)
BULL_WEAK      :   1856 (18.60%)
BEAR_STRONG    :   1123 (11.25%)
BEAR_WEAK      :   1654 (16.57%)
RANGE          :   2890 (28.97%)
VOLATILE       :   1217 (12.19%)
Total          :   9985 (100.00%)

[SAVE] Saving processed data to data/processed/processed_data.csv...
   Saved 9985 rows × 71 columns

============================================================
DATA PIPELINE - Processing Complete!
============================================================

============================================================
NATRON MODEL TRAINING - Starting
============================================================

Model architecture:
  Input features: 65
  Model dimension: 128
  Attention heads: 8
  Encoder layers: 6
  Total parameters: 2,345,678

Created 9890 sequences:
  X shape: (9890, 96, 65)
  y_regime shape: (9890,)
  y_context shape: (9890,)
  y_forecast shape: (9890,)

Dataset splits:
  Train: 6923 samples
  Val:   1483 samples
  Test:  1484 samples

Epoch 1/50:
  Train Loss: 1.234567 (R:0.8234 C:0.1234 F:0.2879)
  Val Loss:   1.198765 (R:0.8123 C:0.1156 F:0.2709)
  LR: 1.00e-04

...

Epoch 50/50:
  Train Loss: 0.456789 (R:0.3456 C:0.0456 F:0.0657)
  Val Loss:   0.478901 (R:0.3567 C:0.0489 F:0.0732)
  LR: 1.00e-06

✓ Saved best model to models/natron_v6/natron_v6.pt

============================================================
TRAINING COMPLETE!
Best validation loss: 0.456789
============================================================
```

---

## 🌐 Running on Google Colab / Vertex AI Studio

### Google Colab Setup

1. **Upload project:**
```python
from google.colab import files
# Upload natron folder and data files
```

2. **Install dependencies:**
```python
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -r natron/requirements.txt
```

3. **Run training:**
```python
import sys
sys.path.append('natron')

from train_model import Trainer
import pandas as pd

# Load data
processed_df = pd.read_csv('data/processed/processed_data.csv', index_col=0)

# Create sequences (use data_pipeline)
from data_pipeline import DataPipeline
pipeline = DataPipeline()
feature_cols = [col for col in processed_df.columns 
               if col not in ['open', 'high', 'low', 'close', 'volume', 'regime']]
feature_df = processed_df[feature_cols]
labels = processed_df['regime'].astype(int)

X, y_regime, y_context, y_forecast = pipeline.create_sequences(
    feature_df, labels, sequence_length=96, forecast_horizon=5
)

# Train
trainer = Trainer('natron/config_train.yaml')
trainer.build_model(n_features=X.shape[-1])
trainer.build_optimizer_and_scheduler()
trainer.build_criterion()

from dataset_loader import create_dataloaders
train_loader, val_loader, test_loader, _ = create_dataloaders(
    X, y_regime, y_context, y_forecast, batch_size=32
)

trainer.train(train_loader, val_loader)
```

### Vertex AI Studio Setup

1. **Create notebook with GPU:**
   - Use T4 or A100 GPU instance
   - Python 3.10 environment

2. **Clone repository:**
```bash
git clone <repo_url>
cd natron
```

3. **Run training:**
```bash
python train_model.py --config config_train.yaml --data data/processed/processed_data.csv
```

---

## 🗺️ Roadmap: G9–G13 Enhancements

### G9: Forecast Fusion
- Combine multiple forecast horizons (5, 10, 20 candles)
- Ensemble predictions from multiple models
- Weighted fusion based on regime

### G10: Order Flow Analysis
- Level 2 order book integration
- Volume profile analysis
- Market microstructure features

### G11: Sentiment Integration
- News sentiment analysis (NLP)
- Social media sentiment (Twitter, Reddit)
- Economic calendar integration

### G12: Strategy Selector
- Multiple trading strategies (scalping, swing, trend)
- Regime-based strategy selection
- Dynamic strategy switching

### G13: Multi-Asset & Portfolio
- Cross-asset correlation
- Portfolio optimization
- Risk parity allocation

---

## 📝 Notes

- **GPU Requirements:** CUDA-capable GPU recommended for training (T4 minimum, A100 preferred)
- **Data Format:** Input CSV must have `time`, `open`, `high`, `low`, `close`, `volume` columns
- **Model Size:** ~2.3M parameters, ~10MB checkpoint file
- **Inference Speed:** ~10ms per prediction on GPU, ~50ms on CPU
- **Memory:** ~2GB RAM for training, ~500MB for inference

---

## 📄 License

This project is provided as-is for educational and research purposes.

---

## 🤝 Contributing

Contributions welcome! Please follow PEP8 style guidelines and include docstrings.

---

**Natron V1.0 – Galaxy-class**  
*Advanced AI Trading System*
