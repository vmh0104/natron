# Natron Project Structure

Complete folder structure tree of the Natron End-to-End AI Trading System.

```
natron/
├── __init__.py                      # Package initialization
├── data_pipeline.py                 # Phase 1: Main data preprocessing pipeline
├── feature_engineering.py          # Phase 1: Technical indicator computation (50-70 features)
├── regime_labeler.py                # Phase 1: Market regime labeling (6 classes)
├── model_natron.py                  # Phase 2: Multi-head transformer model architecture
├── dataset_loader.py                # Phase 2: PyTorch dataset and dataloader
├── train_model.py                   # Phase 2: Model training script
├── evaluate_model.py                # Phase 3: Model evaluation script
├── plot_regime_distribution.py      # Phase 3: Visualization tools
├── natron_server_v5.py              # Phase 4: Realtime trading server
├── realtime_socket.py               # Phase 4: Socket communication module
├── natron_ea.mq5                   # Phase 4: MetaTrader 5 Expert Advisor (MQL5)
├── log_analyzer.py                  # Phase 5: Trading log analysis
├── auto_feedback_engine.py          # Phase 5: Adaptive learning engine
├── monitor_natron.py                # Phase 6: System monitoring
├── start_natron.sh                  # Phase 6: Startup script with auto-restart
├── Dockerfile                       # Phase 6: Docker container definition
├── config_train.yaml                # Training configuration
├── realtime_config.yaml             # Realtime trading configuration
├── adaptive_config.yaml             # Adaptive configuration (auto-generated)
├── requirements.txt                 # Python dependencies
├── README_Natron_EndToEnd.md        # Main documentation
└── PROJECT_STRUCTURE.md             # This file

data/
├── raw/                             # Raw market data (user-provided)
│   └── market_data.csv              # Input CSV: time, open, high, low, close, volume
└── processed/                       # Processed data
    └── processed_data.csv           # Output: features + regime labels

models/
└── natron_v6/                       # Model checkpoints
    ├── natron_v6.pt                 # Best model checkpoint
    ├── checkpoint_latest.pt         # Latest training checkpoint
    ├── training_history.json        # Training metrics history
    └── norm_stats.npy               # Feature normalization statistics

logs/                                # Log files
├── natron.log                       # Server log
└── natron.pid                       # Process ID file

plots/                               # Visualization outputs
├── regime_distribution.png          # Regime distribution over time
└── regime_transitions.png           # Regime transition matrix

feedback_output/                     # Feedback learning outputs
├── adaptive_config.yaml             # Adapted configuration
├── learning_history.json            # Learning history
└── feedback_analysis_report.json    # Analysis report

evaluation_results/                  # Evaluation outputs
├── metrics_report.json              # Evaluation metrics
└── natron_predictions.csv           # Predictions vs targets

realtime_events.csv                  # Realtime trading events log
```

## File Descriptions

### Phase 1: Data Pipeline
- **data_pipeline.py**: Main orchestration script for data preprocessing
- **feature_engineering.py**: Computes 50-70 technical indicators (ATR, EMA, RSI, MACD, Bollinger Bands, etc.)
- **regime_labeler.py**: Labels market regimes into 6 classes based on trend, volatility, and breakouts

### Phase 2: Model Training
- **model_natron.py**: Multi-head transformer model with 3 output heads (regime, context, forecast)
- **dataset_loader.py**: PyTorch Dataset and DataLoader implementations
- **train_model.py**: Training script with mixed precision, GPU support, and checkpointing
- **config_train.yaml**: Training configuration (model architecture, optimizer, loss weights)

### Phase 3: Evaluation
- **evaluate_model.py**: Evaluates model performance (accuracy, F1, precision/recall)
- **plot_regime_distribution.py**: Visualizes regime distributions and transitions

### Phase 4: Realtime Engine
- **natron_server_v5.py**: Main trading server that processes predictions and sends signals
- **realtime_socket.py**: Socket communication for live candle/tick stream
- **natron_ea.mq5**: MetaTrader 5 Expert Advisor that receives signals and executes trades
- **realtime_config.yaml**: Realtime trading configuration

### Phase 5: Feedback Learning
- **log_analyzer.py**: Analyzes trading logs and predictions
- **auto_feedback_engine.py**: Adapts configuration based on performance feedback
- **adaptive_config.yaml**: Auto-generated adaptive configuration

### Phase 6: Deployment
- **Dockerfile**: Docker container definition
- **start_natron.sh**: Startup script with auto-restart on failure
- **monitor_natron.py**: System health monitoring and alerts

## Data Flow

```
Raw CSV → data_pipeline.py → processed_data.csv
                                    ↓
                            train_model.py → natron_v6.pt
                                    ↓
                            evaluate_model.py → metrics_report.json
                                    ↓
                            natron_server_v5.py ← realtime_socket.py
                                    ↓
                            natron_ea.mq5 (MT5) → realtime_events.csv
                                    ↓
                            auto_feedback_engine.py → adaptive_config.yaml
```
