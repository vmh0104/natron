# Natron Project Structure

```
natron/
│
├── data/                              # Data directory
│   ├── raw/                           # Raw market data (CSV files)
│   │   └── market_data.csv            # Input: OHLCV candles
│   └── processed/                     # Processed data ready for ML
│       └── processed_data.csv         # Output: Features + targets
│
├── models/                            # Trained model checkpoints
│   ├── natron_v6.pt                  # Best model checkpoint
│   └── checkpoint_latest.pt           # Latest checkpoint
│
├── configs/                           # Configuration files (YAML)
│   ├── config_train.yaml             # Training configuration
│   ├── realtime_config.yaml           # Realtime trading config
│   └── adaptive_config.yaml           # Auto-adapted config
│
├── logs/                              # Logs and outputs
│   ├── training/                      # Training logs
│   │   └── training_history.json     # Training metrics history
│   ├── evaluation/                    # Evaluation results
│   │   ├── metrics_report.json        # Evaluation metrics
│   │   ├── natron_predictions.csv     # Model predictions
│   │   ├── regime_distribution.png    # Visualization
│   │   └── regime_statistics.png      # Statistics plot
│   ├── realtime_events.csv            # Trading events log
│   └── monitor_status.json            # Monitor status log
│
├── src/                               # Source code
│   ├── __init__.py
│   │
│   ├── data_pipeline/                 # Phase 1: Data Pipeline
│   │   ├── __init__.py
│   │   ├── data_pipeline.py           # Main pipeline orchestrator
│   │   ├── feature_engineering.py     # Feature computation (50-70 features)
│   │   └── regime_labeler.py          # Regime classification (6 classes)
│   │
│   ├── training/                      # Phase 2: Model Training
│   │   ├── __init__.py
│   │   ├── train_model.py             # Training script
│   │   ├── model_natron.py            # Multi-head transformer model
│   │   └── dataset_loader.py          # PyTorch Dataset/DataLoader
│   │
│   ├── evaluation/                    # Phase 3: Evaluation
│   │   ├── __init__.py
│   │   ├── evaluate_model.py          # Model evaluation script
│   │   └── plot_regime_distribution.py # Visualization scripts
│   │
│   ├── realtime/                      # Phase 4: Realtime Engine
│   │   ├── __init__.py
│   │   ├── natron_server_v5.py        # Python trading server
│   │   ├── realtime_socket.py         # Socket communication
│   │   └── natron_ea.mq5              # MQL5 Expert Advisor
│   │
│   ├── feedback/                       # Phase 5: Feedback Learning
│   │   ├── __init__.py
│   │   ├── auto_feedback_engine.py    # Adaptive learning engine
│   │   └── log_analyzer.py            # Log analysis and metrics
│   │
│   └── deployment/                    # Phase 6: Deployment
│       ├── __init__.py
│       └── monitor_natron.py          # Monitoring script
│
├── scripts/                           # Utility scripts
│   └── start_natron.sh                # Startup script with auto-restart
│
├── tests/                             # Unit tests (optional)
│
├── docs/                               # Additional documentation
│
├── Dockerfile                          # Docker container definition
├── requirements.txt                    # Python dependencies
├── README_Natron_EndToEnd.md          # Main documentation
├── PROJECT_STRUCTURE.md               # This file
├── example_training_log.txt           # Example training output
└── example_config.yaml                # Example configuration

```

## Key Files Description

### Data Pipeline (Phase 1)
- `data_pipeline.py`: Main orchestrator for preprocessing
- `feature_engineering.py`: Computes 50-70 technical indicators
- `regime_labeler.py`: Labels 6 market regimes

### Model Training (Phase 2)
- `model_natron.py`: Multi-head transformer architecture
- `train_model.py`: Training loop with mixed precision
- `dataset_loader.py`: Sequence data loading

### Evaluation (Phase 3)
- `evaluate_model.py`: Computes accuracy, F1, precision/recall
- `plot_regime_distribution.py`: Creates visualizations

### Realtime (Phase 4)
- `natron_server_v5.py`: Main trading server
- `natron_ea.mq5`: MetaTrader 5 Expert Advisor

### Feedback (Phase 5)
- `auto_feedback_engine.py`: Self-learning from trades
- `log_analyzer.py`: Analyzes prediction vs outcome

### Deployment (Phase 6)
- `monitor_natron.py`: Health monitoring
- `Dockerfile`: Containerization
- `start_natron.sh`: Auto-start script
