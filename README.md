# 🧠 Natron Transformer – Multi-Task Financial Trading Model

End-to-End GPU Pipeline for AI-Powered Trading

## 📋 Overview

Natron is a comprehensive deep learning system for financial trading that combines:
- **Transformer architecture** for sequence modeling
- **Multi-task learning** (Buy/Sell, Direction, Regime classification)
- **Three-phase training** (Pretraining → Supervised → Reinforcement Learning)
- **Real-time inference** via REST API and Socket server
- **MetaTrader 5 integration** for automated execution

## 🏗️ Architecture

### Model Components

1. **NatronTransformerEncoder**: Transformer encoder for market representation learning
2. **NatronTransformer**: Multi-task model with task-specific heads
3. **FeatureEngine**: Generates ~100 technical features from OHLCV data
4. **LabelGenerator**: Creates buy/sell/direction/regime labels

### Training Phases

1. **Phase 1 - Pretraining**: Unsupervised learning via masked modeling and contrastive learning
2. **Phase 2 - Supervised Fine-tuning**: Multi-task learning with labeled data
3. **Phase 3 - Reinforcement Learning**: PPO-based optimization for trading performance

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.x with CUDA support
- Ubuntu/Debian Linux (or Docker)
- MetaTrader 5 (for EA integration)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd natron

# Install dependencies
pip install -r requirements.txt

# Prepare data
# Place your OHLCV data as data_export.csv with columns:
# time, open, high, low, close, volume
```

### Training

```bash
# Phase 1: Pretraining
python train_pretrain.py

# Phase 2: Supervised Fine-tuning
python train_supervised.py

# Phase 3: Reinforcement Learning (optional)
python train_rl.py

# Or train all phases:
bash start_natron.sh train-all
```

### Inference

#### Flask API Server

```bash
python api_server.py --host 0.0.0.0 --port 5000
```

**API Endpoint**: `POST /predict`

Request:
```json
{
  "candles": [
    {"time": "2024-01-01 00:00:00", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 1000},
    ...
  ]
}
```

Response:
```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_up": 0.69,
  "regime": "BULL_WEAK",
  "regime_id": 1,
  "confidence": 0.82
}
```

#### Socket Server (for MQL5)

```bash
python socket_server.py --host 127.0.0.1 --port 8888
```

### MetaTrader 5 Integration

1. Copy `natron_ea.mq5` to `MetaTrader 5/MQL5/Experts/`
2. Compile in MetaEditor
3. Attach to chart with parameters:
   - Server Host: `127.0.0.1`
   - Server Port: `8888`
   - Lot Size, Stop Loss, Take Profit, etc.

## 📁 Project Structure

```
natron/
├── config.yaml              # Configuration file
├── requirements.txt        # Python dependencies
├── feature_engine.py       # Feature engineering (~100 features)
├── label_generator.py      # Label generation (buy/sell/regime)
├── dataset_loader.py       # PyTorch dataset and data loaders
├── model_natron.py         # Transformer model architecture
├── losses.py               # Loss functions (multi-task, contrastive, etc.)
├── train_pretrain.py       # Phase 1: Pretraining
├── train_supervised.py     # Phase 2: Supervised fine-tuning
├── train_rl.py             # Phase 3: Reinforcement learning
├── api_server.py           # Flask REST API server
├── socket_server.py        # TCP socket server for MQL5
├── natron_ea.mq5           # MetaTrader 5 Expert Advisor
├── monitor_natron.py       # System monitoring script
├── start_natron.sh         # Startup script
├── Dockerfile              # Docker container definition
└── README.md               # This file
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

- **Model architecture**: d_model, nhead, num_layers, etc.
- **Training parameters**: batch_size, learning_rate, epochs
- **Data splits**: train/val/test ratios
- **Labeling rules**: thresholds for buy/sell signals
- **API/Socket settings**: ports, hosts

## 📊 Features Generated

The system automatically generates ~100 technical features:

- **Moving Averages** (13): MA5-100, EMA12/26, crossovers, slopes
- **Momentum** (13): RSI, ROC, CCI, Stochastic, MACD
- **Volatility** (15): ATR, Bollinger Bands, Keltner Channels, StdDev
- **Volume** (9): OBV, VWAP, MFI, volume ratios
- **Price Patterns** (8): Doji, gaps, shadows, body percentage
- **Returns** (8): Log returns, intraday returns, cumulative
- **Trend Strength** (6): ADX, +DI, -DI, Aroon
- **Statistical** (6): Z-score, skewness, kurtosis, Hurst exponent
- **Support/Resistance** (4): Distance to highs/lows
- **SMC** (6): Swing highs/lows, BOS, CHOCH
- **Market Profile** (10): POC, VAH, VAL, entropy

## 🎯 Labeling Logic

### Buy Signal (≥2 conditions):
- close > MA20 > MA50
- RSI > 50 or recently left oversold
- close > BB midband and MA20_slope > 0
- volume > 1.5× rolling average
- close near high (≥70%)
- MACD_hist > 0 and increasing

### Sell Signal (≥2 conditions):
- close < MA20 < MA50
- RSI < 50 or turning down from overbought
- close < BB midband and MA20_slope < 0
- volume spike and close near low
- MACD_hist < 0 and decreasing

### Regime Classes (6):
0. **BULL_STRONG**: trend > +2%, ADX > 25
1. **BULL_WEAK**: 0 < trend ≤ 2%, ADX ≤ 25
2. **RANGE**: lateral market
3. **BEAR_WEAK**: -2% ≤ trend < 0, ADX ≤ 25
4. **BEAR_STRONG**: trend < -2%, ADX > 25
5. **VOLATILE**: ATR > 90th percentile or volume spike

## 🔧 Monitoring

```bash
# Monitor system health and performance
python monitor_natron.py --interval 60
```

Tracks:
- API health
- Socket server connectivity
- CPU/GPU usage
- Memory consumption
- Model status

## 🐳 Docker Deployment

```bash
# Build image
docker build -t natron:latest .

# Run API server
docker run -d -p 5000:5000 --gpus all natron:latest python api_server.py

# Run socket server
docker run -d -p 8888:8888 --gpus all natron:latest python socket_server.py
```

## 📈 Performance Optimization

- **GPU Acceleration**: Automatic CUDA detection
- **Batch Processing**: Configurable batch sizes
- **Mixed Precision**: Can be enabled for faster training
- **Data Loading**: Multi-worker DataLoader with pin_memory

## 🔒 Production Considerations

- **Error Handling**: Comprehensive try-catch blocks
- **Logging**: TensorBoard integration for training metrics
- **Model Checkpointing**: Automatic best model saving
- **Scalability**: Threaded socket server for multiple clients
- **Security**: Input validation and sanitization

## 📝 License

[Your License Here]

## 🤝 Contributing

[Contributing Guidelines]

## 📧 Support

[Support Information]

---

**Built with PyTorch 2.x | Optimized for CUDA | Production-Ready**
