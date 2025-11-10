# Natron Project Structure

```
natron/
│
├── __init__.py                          # Package initialization
├── requirements.txt                     # Python dependencies
├── README_Natron_EndToEnd.md            # Main documentation
├── PROJECT_STRUCTURE.md                 # This file
│
├── Phase 1 - Data Pipeline/
│   ├── data_pipeline.py                 # Main pipeline orchestrator
│   ├── feature_engineering.py           # 50-70 technical indicators
│   └── regime_labeler.py                # 6-class regime labeling
│
├── Phase 2 - Model Training/
│   ├── model_natron.py                  # Multi-head transformer model
│   ├── dataset_loader.py                # PyTorch Dataset & DataLoader
│   ├── train_model.py                   # Training script
│   └── config_train.yaml                # Training configuration
│
├── Phase 3 - Evaluation/
│   ├── evaluate_model.py                # Model evaluation & metrics
│   └── plot_regime_distribution.py      # Visualization tools
│
├── Phase 4 - Realtime Engine/
│   ├── natron_server_v5.py              # Python trading server
│   ├── realtime_socket.py               # Socket client for data feed
│   ├── natron_ea.mq5                    # MetaTrader 5 Expert Advisor
│   └── realtime_config.yaml             # Realtime configuration
│
├── Phase 5 - Feedback Learning/
│   ├── auto_feedback_engine.py          # Adaptive weight adjustment
│   ├── log_analyzer.py                  # Trade outcome analysis
│   └── adaptive_config.yaml             # Adaptive configuration (auto-updated)
│
├── Phase 6 - Deployment/
│   ├── Dockerfile                       # Docker container definition
│   ├── start_natron.sh                  # Auto-start script with restart
│   ├── monitor_natron.py                # System monitoring & alerts
│   └── monitor_config.yaml              # Monitor configuration
│
├── data/                                # Data directory (created at runtime)
│   ├── raw_data.csv                     # Input OHLCV data
│   └── processed_data.csv               # Processed features
│
├── checkpoints/                         # Model checkpoints (created at runtime)
│   ├── natron_v6.pt                     # Best model
│   ├── latest.pt                         # Latest checkpoint
│   └── training_history.json            # Training history
│
├── logs/                                 # Logs directory (created at runtime)
│   └── natron_*.log                     # Application logs
│
├── outputs/                              # Outputs (created at runtime)
│   ├── natron_predictions.csv            # Model predictions
│   ├── metrics_report.json               # Evaluation metrics
│   ├── regime_distribution.png           # Visualization plots
│   ├── realtime_events.csv               # Trading events log
│   └── learning_curve.json              # Learning curve data
│
└── .gitignore                           # Git ignore file
```

## File Descriptions

### Phase 1 - Data Pipeline
- **data_pipeline.py**: Orchestrates data loading, feature engineering, and regime labeling
- **feature_engineering.py**: Computes 50-70 technical indicators (ATR, EMA, RSI, MACD, Bollinger Bands, etc.)
- **regime_labeler.py**: Labels market regimes into 6 classes based on trend, volatility, and breakouts

### Phase 2 - Model Training
- **model_natron.py**: Multi-head transformer architecture with 3 output heads
- **dataset_loader.py**: PyTorch Dataset for sequence-based training
- **train_model.py**: Training script with mixed precision, LR scheduling, checkpointing
- **config_train.yaml**: Training hyperparameters and model configuration

### Phase 3 - Evaluation
- **evaluate_model.py**: Computes accuracy, F1, precision/recall, generates reports
- **plot_regime_distribution.py**: Visualizes regime distributions and forecast accuracy over time

### Phase 4 - Realtime Engine
- **natron_server_v5.py**: Python server that processes live data and generates trade signals
- **realtime_socket.py**: Socket client for receiving market data
- **natron_ea.mq5**: MetaTrader 5 Expert Advisor for trade execution
- **realtime_config.yaml**: Trading parameters, socket settings, risk management

### Phase 5 - Feedback Learning
- **auto_feedback_engine.py**: Analyzes trade outcomes and adapts configuration weights
- **log_analyzer.py**: Parses trading logs and computes performance metrics
- **adaptive_config.yaml**: Auto-updated configuration with confidence scaling and regime weights

### Phase 6 - Deployment
- **Dockerfile**: Container definition for production deployment
- **start_natron.sh**: Startup script with auto-restart on failure
- **monitor_natron.py**: System health monitoring and alerting
- **monitor_config.yaml**: Monitor settings and Telegram alert configuration

## Data Flow

```
Raw OHLCV CSV
    ↓
[Phase 1] Data Pipeline
    ↓
Processed Data (features + regimes)
    ↓
[Phase 2] Model Training
    ↓
Trained Model (natron_v6.pt)
    ↓
[Phase 3] Evaluation
    ↓
Predictions + Metrics
    ↓
[Phase 4] Realtime Server
    ↓
Trade Signals → MQL5 EA
    ↓
[Phase 5] Feedback Learning
    ↓
Updated Config → Back to Phase 4
```

## Execution Order

1. **Data Preparation**: `python data_pipeline.py input.csv processed_data.csv`
2. **Model Training**: `python train_model.py config_train.yaml`
3. **Evaluation**: `python evaluate_model.py config_train.yaml checkpoints/natron_v6.pt`
4. **Realtime Trading**: `python natron_server_v5.py realtime_config.yaml` (with MT5 EA running)
5. **Feedback Learning**: `python auto_feedback_engine.py adapt` (periodic)
6. **Monitoring**: `python monitor_natron.py monitor_config.yaml` (continuous)
