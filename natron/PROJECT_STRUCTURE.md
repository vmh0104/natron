# Natron Project Structure

```
natron/
│
├── 📁 configs/                          # Configuration Files
│   ├── config_train.yaml                # Training configuration
│   ├── realtime_config.yaml             # Realtime trading config
│   └── adaptive_config.yaml             # Adaptive learning config
│
├── 📁 data/                             # Data Directory
│   ├── raw_data.csv                     # Input OHLCV data
│   └── processed_data.csv               # Processed features (generated)
│
├── 📁 models/                           # Model Checkpoints
│   ├── natron_v6.pt                     # Trained model (generated)
│   └── scaler.pkl                       # Feature scaler (generated)
│
├── 📁 logs/                             # Logs and Outputs
│   ├── training_history.json            # Training metrics (generated)
│   ├── natron_predictions.csv           # Model predictions (generated)
│   ├── realtime_events.csv              # Trading events (generated)
│   ├── metrics_report.json              # Evaluation metrics (generated)
│   └── learning_curve.json             # Learning curve (generated)
│
├── 📁 mql5/                            # MetaTrader 5 Expert Advisor
│   └── natron_ea.mq5                   # MQL5 EA for MT5 integration
│
├── 📁 scripts/                         # Utility Scripts
│   ├── start_natron.sh                 # Server startup script
│   ├── monitor_natron.py               # Monitoring script
│   └── generate_sample_data.py         # Sample data generator
│
├── 📁 src/                             # Source Code
│   │
│   ├── 📁 data_pipeline/               # Phase 1: Data Pipeline
│   │   ├── __init__.py
│   │   ├── data_pipeline.py            # Main pipeline orchestrator
│   │   ├── feature_engineering.py     # Technical indicators (50-70 features)
│   │   └── regime_labeler.py          # Regime classification (6 classes)
│   │
│   ├── 📁 training/                    # Phase 2: Model Training
│   │   ├── __init__.py
│   │   ├── model_natron.py             # Multi-head transformer model
│   │   ├── dataset_loader.py           # PyTorch dataset & data loaders
│   │   └── train_model.py              # Training script
│   │
│   ├── 📁 evaluation/                  # Phase 3: Evaluation
│   │   ├── __init__.py
│   │   ├── evaluate_model.py          # Model evaluation & metrics
│   │   └── plot_regime_distribution.py # Visualization scripts
│   │
│   ├── 📁 realtime/                    # Phase 4: Realtime Execution
│   │   ├── __init__.py
│   │   ├── natron_server_v5.py         # Main trading server
│   │   └── realtime_socket.py          # Socket communication
│   │
│   ├── 📁 feedback/                    # Phase 5: Feedback Learning
│   │   ├── __init__.py
│   │   ├── log_analyzer.py             # Performance analysis
│   │   └── auto_feedback_engine.py     # Adaptive learning engine
│   │
│   └── __init__.py                     # Package init
│
├── 📄 requirements.txt                 # Python dependencies
├── 📄 Dockerfile                       # Docker container definition
├── 📄 README_Natron_EndToEnd.md        # Full documentation
├── 📄 QUICKSTART.md                    # Quick start guide
├── 📄 PROJECT_STRUCTURE.md             # This file
└── 📄 .gitignore                       # Git ignore rules

```

## File Descriptions

### Configuration Files (`configs/`)

- **config_train.yaml**: Model architecture, training hyperparameters, data splits
- **realtime_config.yaml**: MT5 connection, trading parameters, risk management
- **adaptive_config.yaml**: Feedback learning parameters, threshold adaptation

### Source Code (`src/`)

#### Data Pipeline (`src/data_pipeline/`)
- **feature_engineering.py**: Computes 50-70 technical indicators (ATR, EMA, RSI, MACD, Bollinger Bands, etc.)
- **regime_labeler.py**: Labels market regimes (BULL_STRONG, BULL_WEAK, BEAR_STRONG, BEAR_WEAK, RANGE, VOLATILE)
- **data_pipeline.py**: Orchestrates preprocessing pipeline

#### Training (`src/training/`)
- **model_natron.py**: Multi-head transformer with 3 outputs (regime, context, forecast)
- **dataset_loader.py**: PyTorch Dataset for sequence data
- **train_model.py**: Training loop with early stopping, mixed precision, etc.

#### Evaluation (`src/evaluation/`)
- **evaluate_model.py**: Computes accuracy, F1, precision/recall, exports predictions
- **plot_regime_distribution.py**: Generates visualizations

#### Realtime (`src/realtime/`)
- **natron_server_v5.py**: Main server coordinating model inference and trade execution
- **realtime_socket.py**: Socket-based data streaming

#### Feedback (`src/feedback/`)
- **log_analyzer.py**: Analyzes trading performance metrics
- **auto_feedback_engine.py**: Adapts configuration based on performance

### Scripts (`scripts/`)

- **start_natron.sh**: Startup script with auto-restart
- **monitor_natron.py**: Health monitoring and alerts
- **generate_sample_data.py**: Generates synthetic OHLCV data for testing

### MQL5 (`mql5/`)

- **natron_ea.mq5**: MetaTrader 5 Expert Advisor for live trading integration

---

## Data Flow

```
Raw OHLCV CSV
    ↓
[Phase 1] Data Pipeline
    ├── Feature Engineering (50-70 features)
    └── Regime Labeling (6 classes)
    ↓
Processed Data CSV
    ↓
[Phase 2] Model Training
    ├── Create Sequences (96 candles)
    ├── Train Transformer
    └── Save Model Checkpoint
    ↓
Trained Model (.pt)
    ↓
[Phase 3] Evaluation
    ├── Generate Predictions
    ├── Compute Metrics
    └── Create Visualizations
    ↓
Predictions CSV + Reports
    ↓
[Phase 4] Realtime Server
    ├── Load Model
    ├── Receive Live Data
    ├── Make Predictions
    └── Execute Trades (MT5)
    ↓
Trading Events Log
    ↓
[Phase 5] Feedback Learning
    ├── Analyze Performance
    ├── Adapt Thresholds
    └── Update Configuration
    ↓
Adapted Config
```

---

## Key Components

### Model Architecture
- **Input**: 96 candles × 50-70 features
- **Architecture**: Transformer Encoder (6 layers, 8 heads, 256 dim)
- **Outputs**:
  1. Regime classification (6 classes)
  2. Context strength (1 value, 0-1)
  3. Forecast direction (2 classes: UP/DOWN)

### Features (50-70 total)
- Technical Indicators: ATR, EMA(20,50,200), RSI, MACD, Bollinger Bands
- Candle Features: body_pct, wick_ratio
- Price Features: returns, momentum, volatility
- Volume Features: volume_ratio, volume_price_trend

### Regimes (6 classes)
1. BULL_STRONG - Strong uptrend with breakouts
2. BULL_WEAK - Weak uptrend
3. BEAR_STRONG - Strong downtrend with breakouts
4. BEAR_WEAK - Weak downtrend
5. RANGE - Sideways market
6. VOLATILE - High volatility regime

---

## Dependencies

See `requirements.txt` for full list. Key packages:
- PyTorch 2.0+ (with CUDA support)
- Pandas, NumPy (data processing)
- scikit-learn (metrics, preprocessing)
- MetaTrader5 (live trading)
- Matplotlib, Seaborn (visualization)
- PyYAML (configuration)

---

## Version

**Natron V1.0 - Galaxy-class**
