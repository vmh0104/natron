# 🧠 Natron Transformer – Multi-Task Financial Trading Model

End-to-End GPU Pipeline for AI-Powered Financial Trading

## 📋 Overview

Natron Transformer is a multi-task deep learning model that learns multiple market representations and trading decisions from sequences of 96 consecutive OHLCV candles. The system operates through three layers of intelligence:

1. **Structure Understanding (Pretrain)**: Learn hidden dynamics of market sequences
2. **Signal Recognition (Supervised)**: Detect patterns leading to directional outcomes
3. **Behavioral Adaptation (Reinforcement)**: Continuously optimize decisions (optional)

## 🎯 Model Outputs

- **Buy/Sell Classification**: Binary signals for entry/exit
- **Directional Prediction**: Up/Down movement forecast
- **Market Regime Classification**: 6 market states (BULL_STRONG, BULL_WEAK, RANGE, BEAR_WEAK, BEAR_STRONG, VOLATILE)

## 🏗️ Architecture

### Input
- **Sequence**: 96 consecutive OHLCV candles
- **Features**: ~100 engineered technical features

### Model
- **Architecture**: Multi-head Transformer Encoder
- **Tasks**: 4 parallel prediction heads
  - Buy head (sigmoid)
  - Sell head (sigmoid)
  - Direction head (softmax, 2 classes)
  - Regime head (softmax, 6 classes)

## 📦 Installation

### Requirements
- Python 3.10+
- PyTorch 2.x with CUDA support
- Ubuntu/Debian Linux (or compatible)

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Ensure CUDA is available
python -c "import torch; print(torch.cuda.is_available())"
```

## 🚀 Quick Start

### 1. Prepare Data

Place your OHLCV data in `data_export.csv` with columns:
- `time`: Timestamp
- `open`: Open price
- `high`: High price
- `low`: Low price
- `close`: Close price
- `volume`: Volume

### 2. Train Model

```bash
python train_natron.py --data data_export.csv --config config.yaml
```

Training consists of two phases:
- **Phase 1**: Unsupervised pretraining (masked reconstruction + contrastive learning)
- **Phase 2**: Supervised fine-tuning (multi-task prediction)

### 3. Run Inference Server

```bash
python server_natron.py --model model/natron_v2.pt --flask-port 5000 --socket-port 8888
```

### 4. Deploy MQL5 EA

1. Copy `natron_ea.mq5` to MetaTrader 5 `Experts` folder
2. Configure EA parameters:
   - `ServerHost`: Python server IP (use `localhost` if same machine)
   - `ServerPort`: Socket port (default: 8888)
   - `BuyThreshold`: Minimum buy probability (default: 0.65)
   - `SellThreshold`: Minimum sell probability (default: 0.65)
3. Attach EA to chart

## 📡 API Usage

### Flask REST API

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "candles": [
      {"time": 1234567890, "open": 1.1000, "high": 1.1050, "low": 1.0990, "close": 1.1030, "volume": 1000},
      ...
    ]
  }'
```

**Response:**
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

### Socket API (MQL5)

The MQL5 EA communicates via TCP socket on port 8888 (default). Messages are JSON-encoded:

**Request:**
```json
{
  "action": "predict",
  "candles": [...]
}
```

**Response:**
```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_up": 0.69,
  "regime": "BULL_WEAK",
  "confidence": 0.82
}
```

## 📁 Project Structure

```
.
├── feature_engine.py      # Technical feature generation (~100 features)
├── label_generator.py     # Buy/Sell/Direction/Regime labeling
├── dataset_loader.py      # Sequence creation and PyTorch Dataset
├── model_natron.py        # Transformer architecture
├── losses.py              # Multi-task loss functions
├── train_natron.py        # Training pipeline
├── server_natron.py       # Inference server (Flask + Socket)
├── natron_ea.mq5          # MetaTrader 5 Expert Advisor
├── config.yaml            # Configuration file
├── requirements.txt       # Python dependencies
├── start_natron.sh        # Startup script
└── README.md              # This file
```

## ⚙️ Configuration

Edit `config.yaml` to customize:

- **Model architecture**: Dimensions, layers, heads
- **Training parameters**: Learning rates, batch sizes, epochs
- **Loss weights**: Balance between tasks
- **Server ports**: Flask and socket ports

## 🔧 Feature Engineering

The system automatically generates ~100 technical features across 11 categories:

1. **Moving Averages** (13): MA, EMA, slopes, crossovers
2. **Momentum** (13): RSI, ROC, CCI, Stochastic, MACD
3. **Volatility** (15): ATR, Bollinger Bands, Keltner Channels
4. **Volume** (9): OBV, VWAP, MFI, volume ratios
5. **Price Patterns** (8): Doji, gaps, shadows, body%
6. **Returns** (8): Log returns, intraday, cumulative
7. **Trend Strength** (6): ADX, +DI, -DI, Aroon
8. **Statistical** (6): Skewness, Kurtosis, Z-score, Hurst
9. **Support/Resistance** (4): Distance to highs/lows
10. **SMC** (6): Swing High/Low, BOS/CHOCH
11. **Market Profile** (10): POC, VAH, VAL, Entropy

## 🎓 Training Philosophy

### Phase 1: Pretraining
- **Objective**: Learn latent market representations
- **Methods**: 
  - Masked modeling (reconstruct masked tokens)
  - Contrastive learning (InfoNCE)

### Phase 2: Supervised Fine-Tuning
- **Objective**: Predict buy/sell, direction, regime simultaneously
- **Loss**: Weighted combination of binary cross-entropy (buy/sell) and cross-entropy (direction/regime)

### Phase 3: Reinforcement Learning (Optional)
- **Objective**: Maximize real-world trading performance
- **Algorithm**: PPO or SAC
- **Reward**: `profit - α * turnover - β * drawdown`

## 📊 Model Performance

Monitor training progress:
- Buy/Sell accuracy
- Direction accuracy
- Regime classification accuracy
- Validation loss trends

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in `config.yaml`
- Reduce `sequence_length` or `d_model`

### Connection Issues (MQL5)
- Ensure Python server is running: `python server_natron.py`
- Check firewall settings
- Verify `ServerHost` and `ServerPort` in EA settings

### Low Prediction Accuracy
- Increase training epochs
- Adjust loss weights
- Check data quality and feature engineering
- Enable focal loss for imbalanced classes

## 📝 License

This project is provided as-is for research and educational purposes.

## 🤝 Contributing

Contributions welcome! Please ensure:
- Code follows PEP 8 style
- All tests pass
- Documentation is updated

## 📧 Support

For issues and questions, please open an issue on the project repository.

---

**Built with PyTorch 2.x | Optimized for CUDA | Production-Ready**
