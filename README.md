# 🧠 Natron V2 – Multi-Task Financial Trading AI System

**End-to-End GPU-Optimized Transformer Model for Automated Trading**

Natron V2 is a sophisticated AI trading system that uses a multi-task Transformer architecture to predict market movements, classify regimes, and generate trading signals. The system integrates seamlessly with MetaTrader 5 for real-time automated trading.

---

## 🎯 System Overview

Natron learns market dynamics through three progressive training phases:

1. **Phase 1: Pretraining (Unsupervised)** – Learns latent market representations through masked modeling and contrastive learning
2. **Phase 2: Supervised Fine-Tuning** – Multi-task learning for buy/sell signals, direction prediction, and regime classification
3. **Phase 3: Reinforcement Learning (Optional)** – Optimizes trading decisions for real-world profitability

### Architecture Highlights

- **Input**: 96 consecutive OHLCV candles
- **Features**: 100+ technical indicators (MA, RSI, MACD, Bollinger Bands, ADX, etc.)
- **Model**: Transformer encoder with positional encoding
- **Outputs**:
  - Buy/Sell probabilities (sigmoid)
  - Direction prediction (2-class softmax)
  - Market regime classification (6 classes)

---

## 📊 Market Regime Classification

Natron classifies market conditions into 6 distinct regimes:

| Regime ID | Name | Condition |
|-----------|------|-----------|
| 0 | BULL_STRONG | Trend > +2%, ADX > 25 |
| 1 | BULL_WEAK | 0 < Trend ≤ 2%, ADX ≤ 25 |
| 2 | RANGE | Lateral market |
| 3 | BEAR_WEAK | -2% ≤ Trend < 0, ADX ≤ 25 |
| 4 | BEAR_STRONG | Trend < -2%, ADX > 25 |
| 5 | VOLATILE | ATR > 90th percentile |

---

## 🚀 Quick Start

### Prerequisites

- **Python**: 3.10+
- **GPU**: NVIDIA GPU with CUDA 11.8+ (recommended)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ for models and logs

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd natron-v2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Data Preparation

Place your OHLCV data in `data/data_export.csv` with the following format:

```csv
time,open,high,low,close,volume
1234567890,1.1234,1.1250,1.1220,1.1245,1000
...
```

**Requirements**:
- Columns: `time`, `open`, `high`, `low`, `close`, `volume`
- Minimum: 1000+ candles (more data = better performance)
- Timeframe: M15 or H1 recommended

---

## 🎓 Training

### Full Training Pipeline (All Phases)

```bash
python train_natron.py --data data/data_export.csv
```

This runs:
1. Data preprocessing + feature engineering
2. Phase 1: Pretraining (50 epochs)
3. Phase 2: Supervised training (100 epochs)
4. Phase 3: Reinforcement learning (50 episodes)

### Training Individual Phases

**Pretraining only:**
```bash
python train_natron.py --phase pretrain --data data/data_export.csv
```

**Supervised training only:**
```bash
python train_natron.py --phase supervised \
    --pretrain-checkpoint model/pretrain_best.pt \
    --data data/data_export.csv
```

**Reinforcement learning only:**
```bash
python train_natron.py --phase rl \
    --supervised-checkpoint model/supervised_best.pt \
    --data data/data_export.csv
```

### Training Options

```bash
python train_natron.py --help

Options:
  --config CONFIG              Path to config file (default: config/config.yaml)
  --data DATA                  Path to OHLCV CSV
  --phase {all,pretrain,supervised,rl}
  --device {cuda,cpu}          Device for training
  --skip-pretrain             Skip pretraining phase
  --skip-rl                   Skip RL phase
  --pretrain-checkpoint PATH   Load pretrained weights
  --supervised-checkpoint PATH Load supervised weights
```

### Monitoring Training

Use Tensorboard to monitor training progress:

```bash
tensorboard --logdir logs
```

Open browser to `http://localhost:6006`

---

## 🔮 Inference

### Option 1: Socket Server (Recommended for MQL5)

Start the real-time socket server:

```bash
python src/inference/socket_server.py --model model/natron_v2.pt
```

This starts a TCP server on port 9090 for MQL5 communication.

### Option 2: Flask REST API

Start the Flask API server:

```bash
python src/inference/flask_api.py --model model/natron_v2.pt --port 5000
```

**API Endpoints:**

- `GET /health` – Health check
- `POST /predict` – Single prediction
- `POST /batch_predict` – Batch predictions
- `GET /model_info` – Model information

**Example Request:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      [1234567890, 1.1234, 1.1250, 1.1220, 1.1245, 1000],
      ...  # 96 candles total
    ]
  }'
```

**Example Response:**
```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_up": 0.69,
  "direction_class": 1,
  "regime": "BULL_WEAK",
  "regime_class": 1,
  "confidence": 0.82
}
```

---

## 📈 MetaTrader 5 Integration

### Setup

1. **Start the Socket Server:**
   ```bash
   python src/inference/socket_server.py
   ```

2. **Install MQL5 EA:**
   - Copy `mql5/natron_ea.mq5` to `MT5_DATA_FOLDER/MQL5/Experts/`
   - Compile in MetaEditor (F7)

3. **Attach EA to Chart:**
   - Drag "NatronEA" onto a chart
   - Configure parameters:
     - `ServerIP`: IP address of Python server (default: 127.0.0.1)
     - `ServerPort`: Socket port (default: 9090)
     - `MinConfidence`: Minimum confidence threshold (default: 0.6)
     - `LotSize`: Position size
     - `StopLoss`, `TakeProfit`: Risk management

### EA Parameters

```
ServerIP = "127.0.0.1"       // Python server IP
ServerPort = 9090            // Socket port
SequenceLength = 96          // Number of candles
MinConfidence = 0.6          // Min confidence (0-1)
MinBuyProb = 0.6            // Min buy probability
MinSellProb = 0.6           // Min sell probability
LotSize = 0.1               // Position size
StopLoss = 100              // SL in points
TakeProfit = 200            // TP in points
UseTrailingStop = true      // Enable trailing stop
UpdateIntervalSeconds = 60  // Update frequency
```

### How It Works

```
MT5 EA → Socket (Port 9090) → Python Server → Natron Model → Prediction → EA → Order Execution
```

1. EA collects 96 OHLCV candles
2. Sends data to Python server via TCP socket
3. Server runs inference and returns prediction
4. EA executes trades based on signals
5. Manages positions with trailing stops

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build Docker image
cd deployment
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f natron-server

# Stop services
docker-compose down
```

### Services

- `natron-server`: Main inference server (ports 5000, 9090)
- `tensorboard`: Training monitoring (port 6006)

### GPU Support

Requires `nvidia-docker` runtime:

```bash
# Install nvidia-docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

---

## 📊 System Monitoring

Monitor system health, GPU usage, and model files:

```bash
# Continuous monitoring (every 60 seconds)
python deployment/monitor_natron.py

# Single check
python deployment/monitor_natron.py --once

# Custom interval
python deployment/monitor_natron.py --interval 30
```

Output:
```
[2025-11-10 12:00:00]
----------------------------------------
CPU: 45.2% | RAM: 67.3% (8.45 GB free) | Disk: 34.5%
GPU: NVIDIA GeForce RTX 3090 | Memory: 2.34 GB / 4.12 GB

Model Files:
  ✓ model/natron_v2.pt (487.23 MB)
  ✓ model/scaler.pkl (2.45 MB)
  ✗ model/rl_best.pt (not found)
----------------------------------------
```

---

## 📁 Project Structure

```
natron-v2/
├── config/
│   └── config.yaml              # Configuration file
├── data/
│   └── data_export.csv          # OHLCV data (user provides)
├── src/
│   ├── features/
│   │   └── feature_engine.py    # 100+ technical indicators
│   ├── labels/
│   │   └── label_generator.py   # Signal + regime labeling
│   ├── data/
│   │   └── dataset_loader.py    # Data preprocessing
│   ├── models/
│   │   ├── natron_transformer.py # Model architecture
│   │   └── losses.py            # Multi-task losses
│   ├── training/
│   │   ├── pretrain.py          # Phase 1 training
│   │   ├── train_supervised.py  # Phase 2 training
│   │   └── train_rl.py          # Phase 3 training
│   └── inference/
│       ├── flask_api.py         # REST API server
│       └── socket_server.py     # Socket server (MQL5)
├── mql5/
│   └── natron_ea.mq5            # MetaTrader 5 EA
├── deployment/
│   ├── Dockerfile               # Docker image
│   ├── docker-compose.yml       # Docker services
│   ├── start_natron.sh          # Startup script
│   └── monitor_natron.py        # System monitor
├── model/                        # Saved models (auto-created)
├── logs/                         # Training logs (auto-created)
├── train_natron.py              # Main training script
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## ⚙️ Configuration

Edit `config/config.yaml` to customize:

```yaml
data:
  sequence_length: 96          # Input sequence length
  train_split: 0.7            # Training data ratio
  val_split: 0.15             # Validation data ratio

model:
  d_model: 256                # Model dimension
  nhead: 8                    # Number of attention heads
  num_encoder_layers: 6       # Transformer layers
  dropout: 0.1                # Dropout rate

training:
  batch_size: 32              # Batch size
  learning_rate: 0.0001       # Learning rate
  epochs_pretrain: 50         # Pretraining epochs
  epochs_supervised: 100      # Supervised epochs
  epochs_rl: 50               # RL episodes

loss_weights:
  buy: 1.0                    # Buy signal weight
  sell: 1.0                   # Sell signal weight
  direction: 1.0              # Direction weight
  regime: 1.5                 # Regime weight (higher priority)
```

---

## 📈 Performance Tips

### Training

1. **More data is better**: Use 10,000+ candles for best results
2. **GPU acceleration**: Training on GPU is 10-50x faster
3. **Batch size**: Increase if you have more GPU memory
4. **Pretraining**: Always use Phase 1 for better convergence

### Inference

1. **GPU vs CPU**: GPU inference is 5-10x faster
2. **Batch predictions**: Use batch API for multiple sequences
3. **Socket server**: Lower latency than HTTP for real-time trading
4. **Model caching**: Model loads once at startup

### Trading

1. **Confidence threshold**: Start with 0.7+ for high-quality signals
2. **Regime filtering**: Avoid VOLATILE regime for safer trades
3. **Risk management**: Always use stop loss and take profit
4. **Backtesting**: Test thoroughly before live trading

---

## 🧪 Testing

Run system health check:

```bash
# Check all components
python deployment/monitor_natron.py --once

# Test Flask API
curl http://localhost:5000/health

# Test Socket server
python -c "import socket; s = socket.socket(); s.connect(('localhost', 9090)); print('Connected!')"
```

---

## 🛠️ Troubleshooting

### Training Issues

**Out of Memory (OOM)**:
- Reduce batch size in `config/config.yaml`
- Reduce sequence length (e.g., 48 instead of 96)
- Use gradient accumulation

**Slow Training**:
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Use smaller model (reduce `d_model`, `num_encoder_layers`)

### Inference Issues

**Socket Connection Failed**:
- Check if server is running: `netstat -an | grep 9090`
- Check firewall settings
- Verify IP address and port in EA settings

**Model Not Found**:
- Train model first: `python train_natron.py`
- Check model path in config

**Low Prediction Quality**:
- Train with more data
- Use pretraining (Phase 1)
- Adjust confidence thresholds

---

## 📚 Technical Details

### Feature Engineering (100+ Features)

1. **Moving Averages (13)**: MA, EMA, slopes, crossovers
2. **Momentum (13)**: RSI, ROC, CCI, Stochastic, MACD
3. **Volatility (15)**: ATR, Bollinger Bands, Keltner, StdDev
4. **Volume (9)**: OBV, VWAP, MFI, Force Index
5. **Price Patterns (8)**: Doji, gaps, shadows, position
6. **Returns (8)**: Log returns, intraday, cumulative
7. **Trend Strength (6)**: ADX, DI+/-, Aroon
8. **Statistical (6)**: Skewness, kurtosis, z-score, Hurst
9. **Support/Resistance (4)**: Distance to highs/lows
10. **Smart Money (6)**: Swing points, BOS, CHOCH, FVG
11. **Market Profile (10)**: POC, VAH, VAL, entropy

### Label Generation

**Buy Signals** (≥2 conditions):
- Close > MA20 > MA50
- RSI > 50 or exiting oversold
- Close > BB mid, MA slope positive
- Volume surge
- Close near high
- MACD histogram positive and rising

**Sell Signals** (≥2 conditions):
- Close < MA20 < MA50
- RSI < 50 or exiting overbought
- Close < BB mid, MA slope negative
- Volume surge, close near low
- MACD histogram negative and falling

---

## 🎯 Roadmap

- [ ] Multi-symbol support
- [ ] Ensemble models (multiple timeframes)
- [ ] Sentiment analysis integration
- [ ] Advanced order types (limit, stop-limit)
- [ ] Web dashboard for monitoring
- [ ] Backtesting framework
- [ ] Walk-forward optimization
- [ ] Risk management module

---

## 📄 License

This project is proprietary software. All rights reserved.

---

## 🤝 Support

For issues, questions, or feature requests, please open an issue or contact the development team.

---

## ⚠️ Disclaimer

**IMPORTANT**: This software is for educational and research purposes only. Trading financial instruments carries significant risk. Past performance does not guarantee future results. Always perform thorough backtesting and use proper risk management. The developers are not responsible for any financial losses incurred from using this software.

---

## 🎉 Acknowledgments

Built with:
- PyTorch 2.x
- Transformers architecture
- MetaTrader 5
- Technical analysis libraries

---

**Natron V2** – Where AI meets Trading 🚀

---
