# 🧠 Natron Transformer – End-to-End AI Trading System

> **Multi-Task Deep Learning for Financial Markets**  
> A state-of-the-art Transformer-based AI system for automated trading with real-time MetaTrader 5 integration.

---

## 🎯 Overview

Natron Transformer is a production-ready AI trading system that combines:

- **Multi-Task Learning**: Simultaneously predicts buy/sell signals, price direction, and market regime
- **Three-Phase Training**: Unsupervised pretraining → Supervised fine-tuning → Reinforcement learning
- **100+ Technical Features**: Comprehensive market analysis across momentum, volatility, volume, and smart money concepts
- **Real-Time Execution**: Direct integration with MetaTrader 5 via MQL5 Expert Advisor
- **GPU-Optimized**: Built for PyTorch 2.x with CUDA acceleration

---

## 🏗️ Architecture

```
OHLCV Data (96 candles)
    ↓
Feature Engine (100+ indicators)
    ↓
Natron Transformer (Multi-Task)
    ├── Buy/Sell Heads (Binary)
    ├── Direction Head (Up/Down)
    └── Regime Head (6 Classes)
    ↓
Trading Signals
    ↓
MQL5 EA → MetaTrader 5 → Live Orders
```

### Model Specifications

- **Input**: 96 consecutive OHLCV candles
- **Features**: 100 technical indicators
- **Architecture**: 6-layer Transformer Encoder (256d, 8 heads)
- **Parameters**: ~5M trainable parameters
- **Outputs**: 
  - Buy probability (sigmoid)
  - Sell probability (sigmoid)
  - Direction (2-class softmax)
  - Market regime (6-class softmax)

---

## 📊 Market Regimes

| ID | Regime | Description |
|----|--------|-------------|
| 0 | BULL_STRONG | Uptrend >2%, ADX >25 |
| 1 | BULL_WEAK | Uptrend 0-2%, ADX ≤25 |
| 2 | RANGE | Sideways, low trend |
| 3 | BEAR_WEAK | Downtrend -2-0%, ADX ≤25 |
| 4 | BEAR_STRONG | Downtrend <-2%, ADX >25 |
| 5 | VOLATILE | High ATR/volume spike |

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone https://github.com/yourusername/natron-transformer.git
cd natron-transformer

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Place your OHLCV data in `data/data_export.csv` with columns:
```
time, open, high, low, close, volume
```

Example:
```csv
time,open,high,low,close,volume
2023-01-01 00:00:00,1.0500,1.0520,1.0495,1.0510,15000
2023-01-01 00:15:00,1.0510,1.0530,1.0505,1.0525,18000
...
```

### 3. Train the Model

**Full Pipeline (All 3 Phases)**:
```bash
python scripts/train_full_pipeline.py --phases 1,2,3
```

**Individual Phases**:
```bash
# Phase 1: Unsupervised Pretraining
python src/training/pretrain.py

# Phase 2: Supervised Training
python src/training/train_supervised.py

# Phase 3: Reinforcement Learning (Optional)
python src/training/train_rl.py
```

### 4. Start Inference Server

```bash
./scripts/start_server.sh
```

Or manually:
```bash
python src/server/api_server.py
```

**Endpoints**:
- REST API: `http://localhost:5000`
- Socket Server: `localhost:9090`

### 5. Deploy MQL5 Expert Advisor

1. Copy `mql5/natron_ea.mq5` to MetaTrader 5 `Experts` folder
2. Compile the EA in MetaEditor
3. Attach to chart with settings:
   - Server IP: `127.0.0.1` (or remote server)
   - Server Port: `9090`
   - Enable Trading: `true`
4. Monitor the information panel on chart

---

## 📁 Project Structure

```
natron-transformer/
├── config/
│   └── config.yaml              # Configuration file
├── data/
│   └── data_export.csv          # OHLCV data (user provides)
├── src/
│   ├── features/
│   │   └── feature_engine.py    # 100+ technical indicators
│   ├── labels/
│   │   └── label_generator.py   # Buy/sell/direction/regime labels
│   ├── data/
│   │   └── dataset_loader.py    # PyTorch datasets & dataloaders
│   ├── models/
│   │   ├── natron_transformer.py  # Main model architecture
│   │   └── losses.py            # Multi-task & contrastive losses
│   ├── training/
│   │   ├── pretrain.py          # Phase 1: Unsupervised
│   │   ├── train_supervised.py  # Phase 2: Supervised
│   │   └── train_rl.py          # Phase 3: Reinforcement
│   └── server/
│       └── api_server.py        # Flask API + Socket server
├── mql5/
│   └── natron_ea.mq5            # MetaTrader 5 Expert Advisor
├── scripts/
│   ├── train_full_pipeline.py   # Complete training pipeline
│   └── start_server.sh          # Server startup script
├── model/                        # Saved model checkpoints
├── logs/                         # Training logs & TensorBoard
├── requirements.txt
└── README.md
```

---

## 🎓 Training Phases

### Phase 1: Unsupervised Pretraining

**Goal**: Learn latent market representations without labels

**Methods**:
- **Masked Modeling**: Reconstruct randomly masked features
- **Contrastive Learning**: NT-Xent loss on augmented pairs

**Duration**: ~50 epochs (~2-4 hours on GPU)

**Output**: `model/natron_pretrained_best.pt`

---

### Phase 2: Supervised Multi-Task Training

**Goal**: Fine-tune for trading signals

**Tasks**:
1. Buy signal classification
2. Sell signal classification
3. Direction prediction (up/down)
4. Market regime classification (6 classes)

**Loss**: Weighted multi-task loss
```python
L = w_buy * L_buy + w_sell * L_sell + w_dir * L_dir + w_regime * L_regime
```

**Duration**: ~100 epochs (~3-5 hours on GPU)

**Output**: `model/natron_supervised_best.pt`

---

### Phase 3: Reinforcement Learning (Optional)

**Goal**: Optimize for real-world trading performance

**Algorithm**: Proximal Policy Optimization (PPO)

**Reward Function**:
```
R = profit - α * turnover - β * drawdown
```

**Environment**: Custom trading simulator with balance tracking

**Duration**: ~1000 episodes (~2-4 hours)

**Output**: `model/natron_rl_best.pt`

---

## 🔧 Configuration

Edit `config/config.yaml` to customize:

### Model Architecture
```yaml
model:
  d_model: 256           # Embedding dimension
  nhead: 8               # Number of attention heads
  num_encoder_layers: 6  # Transformer depth
  dim_feedforward: 1024  # FFN hidden size
  dropout: 0.1
```

### Training
```yaml
supervised:
  epochs: 100
  batch_size: 32
  learning_rate: 0.00005
  task_weights:
    buy: 1.0
    sell: 1.0
    direction: 1.5
    regime: 1.2
```

### Server
```yaml
server:
  host: "0.0.0.0"
  port: 5000
  socket_port: 9090
```

---

## 🌐 API Usage

### REST API

**Endpoint**: `POST /predict`

**Request**:
```json
{
  "data": [
    {
      "time": "2023-01-01 00:00:00",
      "open": 1.0500,
      "high": 1.0520,
      "low": 1.0495,
      "close": 1.0510,
      "volume": 15000
    },
    // ... 96 candles total
  ]
}
```

**Response**:
```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_up": 0.69,
  "direction_down": 0.31,
  "regime": "BULL_WEAK",
  "regime_id": 1,
  "regime_confidence": 0.82,
  "confidence": 0.75,
  "signal": "BUY"
}
```

### Socket Interface (MQL5)

Send JSON via TCP socket to port 9090:
```json
{"data": [{"time": "...", "open": 1.05, ...}, ...]}
```

Receive JSON response with prediction.

---

## 📈 Performance Monitoring

### TensorBoard

```bash
tensorboard --logdir logs/tensorboard
```

View training metrics:
- Loss curves (train/val)
- Task-specific losses
- Accuracy metrics
- Learning rate schedule

### Trading Metrics

The MQL5 EA tracks:
- Total trades
- Win rate
- Total profit/loss
- Maximum drawdown
- Current positions

---

## 🔬 Feature Categories

| Category | Count | Examples |
|----------|-------|----------|
| Moving Averages | 13 | MA, EMA, slopes, ratios |
| Momentum | 13 | RSI, MACD, CCI, Stochastic |
| Volatility | 15 | ATR, Bollinger Bands, Keltner |
| Volume | 9 | OBV, VWAP, MFI, volume ratio |
| Price Patterns | 8 | Doji, gaps, candle positions |
| Returns | 8 | Log returns, intraday, cumulative |
| Trend Strength | 6 | ADX, +DI, -DI, Aroon |
| Statistical | 6 | Skew, kurtosis, Hurst, autocorr |
| Support/Resistance | 4 | Distance to high/low 20/50 |
| Smart Money | 6 | Swing points, BOS, CHOCH |
| Market Profile | 10 | POC, VAH, VAL, entropy |

**Total**: ~100 features

---

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t natron-transformer .
```

### Run Container
```bash
docker run -d \
  --name natron \
  --gpus all \
  -p 5000:5000 \
  -p 9090:9090 \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/data:/app/data \
  natron-transformer
```

---

## 🛠️ Development

### Run Tests
```bash
# Test feature engine
python src/features/feature_engine.py

# Test label generator
python src/labels/label_generator.py

# Test model
python src/models/natron_transformer.py
```

### Add Custom Features

Edit `src/features/feature_engine.py`:

```python
def _my_custom_features(self, df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat['my_indicator'] = calculate_indicator(df['close'])
    return feat

# Add to generate_all_features()
features = pd.concat([features, self._my_custom_features(df)], axis=1)
```

---

## 📚 Citation

If you use Natron Transformer in your research, please cite:

```bibtex
@software{natron_transformer_2024,
  title={Natron Transformer: End-to-End AI Trading System},
  author={Natron AI Team},
  year={2024},
  url={https://github.com/yourusername/natron-transformer}
}
```

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

## ⚖️ License

This project is licensed under the MIT License - see LICENSE file for details.

---

## ⚠️ Disclaimer

**Trading involves substantial risk of loss.**

This software is provided for educational and research purposes only. Past performance does not guarantee future results. Use at your own risk. The authors are not responsible for any financial losses incurred through the use of this software.

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/natron-transformer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/natron-transformer/discussions)
- **Email**: support@natron-ai.com

---

## 🌟 Acknowledgments

Built with:
- PyTorch 2.x
- Flask
- MetaTrader 5
- Technical Analysis Libraries (TA-Lib, ta)

Inspired by state-of-the-art research in:
- Transformer architectures (Vaswani et al., 2017)
- Contrastive learning (Chen et al., 2020)
- Multi-task learning (Caruana, 1997)

---

**Made with ❤️ by the Natron AI Team**
