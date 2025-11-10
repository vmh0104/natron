# 🧠 Natron Transformer V2

**Multi-Task Financial Trading Model with End-to-End GPU Pipeline**

Natron is a state-of-the-art Transformer-based AI system for automated financial trading. It learns market structure through unsupervised pretraining and fine-tunes on multi-task objectives to predict buy/sell signals, price direction, and market regimes.

---

## 🎯 Features

- **Multi-Task Learning**: Simultaneous prediction of buy, sell, direction, and market regime
- **Three-Phase Training**: Pretraining → Supervised → Reinforcement (optional)
- **100+ Technical Features**: Comprehensive feature engineering with TA indicators
- **Real-Time API**: Flask REST API + TCP socket server for MT5 integration
- **GPU Optimized**: Mixed precision training on CUDA-enabled GPUs
- **Production Ready**: Complete MQL5 Expert Advisor for MetaTrader 5

---

## 📊 Architecture

```
Input: 96 OHLCV Candles (M15/H1)
    ↓
Feature Engineering (100+ indicators)
    ↓
Transformer Encoder (6 layers, 8 heads)
    ↓
Multi-Task Heads:
    ├── Buy Signal (Binary)
    ├── Sell Signal (Binary)
    ├── Direction (Up/Down)
    └── Regime (6 classes: Bull Strong/Weak, Bear Strong/Weak, Range, Volatile)
```

---

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
cd /workspace

# Install dependencies
pip install -r requirements.txt

# Verify CUDA (optional but recommended)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2. Prepare Data

Place your OHLCV data in `data/data_export.csv`:

```csv
time,open,high,low,close,volume
2023-01-01 00:00,1.0850,1.0865,1.0845,1.0860,1500
2023-01-01 00:15,1.0860,1.0875,1.0855,1.0870,1600
...
```

Required: Minimum 2000+ rows for proper training.

### 3. Configure

Edit `config.yaml` to adjust:
- Model hyperparameters
- Training settings
- API configuration

### 4. Train Model

**Option A: Full Pipeline (Recommended)**
```bash
python train_natron.py
```

**Option B: Phase-by-Phase**
```bash
# Phase 1: Pretraining (Unsupervised)
python src/train_pretrain.py

# Phase 2: Supervised Fine-Tuning
python src/train_supervised.py
```

Training time: ~2-4 hours on GPU (depends on data size and epochs)

### 5. Test Inference

```bash
python inference.py --csv data/data_export.csv
```

### 6. Start API Server

```bash
python server_natron.py
```

Server will start on:
- Flask API: `http://localhost:8888`
- Socket Server: `tcp://localhost:9999` (for MQL5)

---

## 🔌 API Usage

### REST API

**Endpoint**: `POST /predict`

**Request**:
```json
{
  "data": [
    {"time": "2023-01-01 00:00", "open": 1.0850, "high": 1.0865, "low": 1.0845, "close": 1.0860, "volume": 1500},
    ...
    (96 candles total)
  ]
}
```

**Response**:
```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_up": 0.69,
  "regime": "BULL_WEAK",
  "confidence": 0.82,
  "timestamp": 1234567890.123
}
```

### Test with cURL

```bash
curl -X POST http://localhost:8888/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

---

## 🤖 MetaTrader 5 Integration

### 1. Copy EA to MT5

Copy `natron_ea.mq5` to:
```
C:\Users\[YourUser]\AppData\Roaming\MetaQuotes\Terminal\[ID]\MQL5\Experts\
```

### 2. Compile EA

Open MetaEditor → Open `natron_ea.mq5` → Press F7 to compile

### 3. Configure EA

Attach to chart and set parameters:
- **ServerHost**: IP address of Python server (127.0.0.1 for local)
- **ServerPort**: 9999
- **BuyThreshold**: 0.65 (signal threshold for buy)
- **SellThreshold**: 0.65 (signal threshold for sell)
- **LotSize**: 0.1
- **StopLoss**: 100 points
- **TakeProfit**: 200 points

### 4. Start Trading

EA will:
1. Collect last 96 candles every minute
2. Send to Python server via socket
3. Receive AI prediction
4. Execute trades based on signals
5. Display signals on chart

---

## 📈 Training Details

### Phase 1: Pretraining (Unsupervised)

**Objective**: Learn market structure representations

**Methods**:
- Masked token reconstruction (15% masking)
- Contrastive learning (InfoNCE)

**Duration**: 50 epochs (~1-2 hours on GPU)

**Output**: Pretrained encoder → `models/pretrain/best_pretrain.pt`

### Phase 2: Supervised Fine-Tuning

**Objective**: Multi-task prediction

**Tasks**:
- Buy signal classification
- Sell signal classification  
- Direction prediction (up/down)
- Regime classification (6 classes)

**Loss**: Weighted multi-task loss
```
L_total = L_buy + L_sell + 0.8*L_direction + 0.6*L_regime
```

**Duration**: 100 epochs with early stopping (~2-3 hours on GPU)

**Output**: Final model → `models/natron_v2.pt`

### Phase 3: Reinforcement Learning (Optional)

Not implemented in this version. Future enhancement for live trading optimization.

---

## 📊 Model Performance

Expected metrics on validation set:
- Buy Signal Accuracy: 60-70%
- Sell Signal Accuracy: 60-70%
- Direction Accuracy: 55-65%
- Regime Classification: 65-75%

**Note**: Financial prediction is inherently noisy. The model provides probabilistic signals that should be combined with risk management and portfolio strategies.

---

## 🏗️ Project Structure

```
/workspace/
├── config.yaml                 # Configuration
├── requirements.txt            # Dependencies
├── train_natron.py            # Main training orchestrator
├── server_natron.py           # API + Socket server
├── inference.py               # Standalone inference
├── natron_ea.mq5              # MQL5 Expert Advisor
├── README.md                  # This file
│
├── src/                       # Core modules
│   ├── feature_engine.py      # 100+ technical features
│   ├── label_generator.py     # Multi-task labels
│   ├── dataset_loader.py      # PyTorch datasets
│   ├── model_natron.py        # Transformer architecture
│   ├── losses.py              # Loss functions
│   ├── train_pretrain.py      # Phase 1 training
│   └── train_supervised.py    # Phase 2 training
│
├── data/                      # Data directory
│   └── data_export.csv        # Your OHLCV data (not included)
│
├── models/                    # Trained models
│   ├── natron_v2.pt          # Final model
│   ├── scaler.pkl            # Feature scaler
│   ├── pretrain/             # Pretraining checkpoints
│   └── supervised/           # Supervised checkpoints
│
└── logs/                      # Training logs
```

---

## ⚙️ Configuration

Key parameters in `config.yaml`:

### Model Architecture
```yaml
model:
  d_model: 256              # Transformer dimension
  nhead: 8                  # Attention heads
  num_encoder_layers: 6     # Transformer layers
  dim_feedforward: 1024     # FFN dimension
```

### Training
```yaml
supervised:
  epochs: 100
  batch_size: 32
  learning_rate: 0.00005
  loss_weights:
    buy: 1.0
    sell: 1.0
    direction: 0.8
    regime: 0.6
```

### API
```yaml
api:
  host: "0.0.0.0"
  port: 8888
  model_path: "models/natron_v2.pt"
```

---

## 🐛 Troubleshooting

### Issue: CUDA Out of Memory

**Solution**: Reduce batch size in `config.yaml`:
```yaml
supervised:
  batch_size: 16  # or 8
```

### Issue: Model not converging

**Solutions**:
1. Increase training data (need 2000+ samples)
2. Adjust learning rate
3. Run pretraining first
4. Check data quality (no NaN, proper format)

### Issue: Socket connection failed (MQL5)

**Solutions**:
1. Check if server is running: `curl http://localhost:8888/health`
2. Verify port 9999 is not blocked by firewall
3. Ensure correct ServerHost in EA parameters

### Issue: Poor predictions

**Solutions**:
1. Train on more data (>5000 samples recommended)
2. Adjust buy/sell thresholds in EA
3. Fine-tune label generation rules in `config.yaml`
4. Use pretrained model (Phase 1) before supervised training

---

## 📚 Technical Details

### Feature Groups (100+ indicators)

1. **Moving Averages** (13): MA, EMA, slopes, crossovers
2. **Momentum** (13): RSI, ROC, Stochastic, MACD, CCI
3. **Volatility** (15): ATR, Bollinger Bands, Keltner, HV
4. **Volume** (9): OBV, VWAP, MFI, Volume ratios
5. **Price Patterns** (8): Doji, gaps, shadows, position
6. **Returns** (8): Simple, log, intraday, cumulative
7. **Trend Strength** (6): ADX, DI, Aroon
8. **Statistical** (6): Skewness, kurtosis, z-score, Hurst
9. **Support/Resistance** (4): Distance to highs/lows
10. **Smart Money Concepts** (6): Swing points, BOS
11. **Market Profile** (10): POC, VAH, VAL, entropy

### Market Regimes

| ID | Regime | Condition |
|----|--------|-----------|
| 0 | BULL_STRONG | Trend > +2%, ADX > 25 |
| 1 | BULL_WEAK | 0 < Trend ≤ 2%, ADX ≤ 25 |
| 2 | RANGE | Lateral market |
| 3 | BEAR_WEAK | −2% ≤ Trend < 0, ADX ≤ 25 |
| 4 | BEAR_STRONG | Trend < −2%, ADX > 25 |
| 5 | VOLATILE | ATR > 90th percentile |

---

## 🔬 Research Background

Natron is inspired by:
- **Transformers**: "Attention is All You Need" (Vaswani et al., 2017)
- **Multi-Task Learning**: Improved generalization through auxiliary tasks
- **Financial ML**: "Advances in Financial Machine Learning" (López de Prado, 2018)
- **Contrastive Learning**: SimCLR, MoCo for representation learning

---

## 🤝 Contributing

This is a production-ready research project. Contributions welcome:
- Feature engineering improvements
- New model architectures
- Backtesting framework
- Risk management modules

---

## ⚠️ Disclaimer

**This software is for research and educational purposes only.**

- No financial advice is provided
- Past performance ≠ future results
- Trading involves substantial risk of loss
- Always use proper risk management
- Test thoroughly on demo accounts first
- The authors are not responsible for any financial losses

**USE AT YOUR OWN RISK**

---

## 📝 License

MIT License - See LICENSE file for details

---

## 📧 Contact

For questions or collaboration:
- GitHub Issues: [Project Issues]
- Email: [Your Email]

---

## 🎉 Acknowledgments

Built with:
- PyTorch 2.x
- Pandas, NumPy, Scikit-learn
- Flask
- MetaTrader 5

---

**🧠 Natron Transformer V2** - *End-to-End AI Trading System*

*"The AI doesn't just predict prices—it learns the grammar of the market."*
