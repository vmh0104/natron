# 🧠 Natron Transformer – Multi-Task Financial Trading Model

**End-to-End GPU Pipeline for AI-Powered Trading**

Natron is a sophisticated deep learning system that learns multiple market representations and trading decisions through a Transformer architecture. It combines unsupervised pretraining, supervised fine-tuning, and optional reinforcement learning to create a comprehensive trading AI.

## 🎯 System Overview

Natron learns from sequences of 96 consecutive OHLCV candles and predicts:
- **Buy/Sell signals** (binary classification)
- **Directional prediction** (up/down)
- **Market regime classification** (6 classes: BULL_STRONG, BULL_WEAK, RANGE, BEAR_WEAK, BEAR_STRONG, VOLATILE)

## ⚙️ Architecture

### Three-Layer Intelligence

1. **Structure Understanding (Pretrain)** - Learns hidden dynamics of market sequences through masked modeling and contrastive learning
2. **Signal Recognition (Supervised)** - Detects patterns leading to directional outcomes via multi-task learning
3. **Behavioral Adaptation (Reinforcement)** - Continuously optimizes decisions for profit and risk balance using PPO

### Model Architecture

- **Transformer Encoder**: 6 layers, 8 attention heads, 256-dimensional embeddings
- **Multi-Task Heads**: Separate heads for buy, sell, direction, and regime prediction
- **Feature Engineering**: ~100 technical indicators automatically generated
- **Sequence Length**: 96 consecutive candles per sample

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.x with CUDA support (for GPU training)
- Ubuntu/Debian Linux (or compatible)
- MetaTrader 5 (for MQL5 EA integration)

### Installation

```bash
# Clone repository
cd /workspace

# Install dependencies
pip install -r requirements.txt

# Ensure CUDA is available (for GPU training)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Data Preparation

Place your OHLCV data in `data_export.csv` with columns:
- `time`: Timestamp
- `open`: Open price
- `high`: High price
- `low`: Low price
- `close`: Close price
- `volume`: Volume

Example:
```csv
time,open,high,low,close,volume
2024-01-01 00:00:00,1.1000,1.1050,1.0990,1.1030,1000
2024-01-01 00:15:00,1.1030,1.1070,1.1020,1.1060,1200
...
```

## 📊 Training Pipeline

### Phase 1: Pretraining (Unsupervised)

Learn latent market representations through masked modeling and contrastive learning.

```bash
python train_pretrain.py \
    --data data_export.csv \
    --batch_size 32 \
    --epochs 50 \
    --lr 1e-4 \
    --save_dir ./models
```

**Output**: `models/natron_pretrain_best.pt`

### Phase 2: Supervised Fine-Tuning

Train multi-task heads for buy/sell, direction, and regime prediction.

```bash
python train_natron.py \
    --data data_export.csv \
    --pretrain_model ./models/natron_pretrain_best.pt \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-4 \
    --save_dir ./models
```

**Output**: `models/natron_v2.pt`

### Phase 3: Reinforcement Learning (Optional)

Optimize trading decisions using PPO algorithm.

```bash
python train_rl.py \
    --data data_export.csv \
    --model ./models/natron_v2.pt \
    --episodes 100 \
    --lr 3e-5
```

**Output**: `models/natron_rl_final.pt`

## 🔌 Deployment

### Flask API Server

Start REST API server for predictions:

```bash
python api_server.py \
    --model ./models/natron_v2.pt \
    --port 5000 \
    --host 0.0.0.0
```

**API Endpoints**:
- `GET /health` - Health check
- `POST /predict` - Single prediction
- `POST /predict_batch` - Batch predictions

**Example Request**:
```bash
curl -X POST http://localhost:5000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "candles": [
            {"time": "2024-01-01 00:00:00", "open": 1.1000, "high": 1.1050, "low": 1.0990, "close": 1.1030, "volume": 1000},
            ...
        ]
    }'
```

**Example Response**:
```json
{
    "buy_prob": 0.71,
    "sell_prob": 0.24,
    "direction_up": 0.69,
    "regime": "BULL_WEAK",
    "confidence": 0.82,
    "regime_probs": [0.1, 0.4, 0.2, 0.1, 0.1, 0.1],
    "timestamp": "2024-01-01T12:00:00"
}
```

### Socket Server (MQL5 Integration)

Start TCP socket server for MetaTrader 5 EA:

```bash
python socket_server.py \
    --host localhost \
    --port 8888 \
    --model ./models/natron_v2.pt
```

### MetaTrader 5 Expert Advisor

1. Copy `natron_ea.mq5` to `MetaTrader 5/MQL5/Experts/`
2. Compile in MetaEditor
3. Attach to chart with settings:
   - **Server Host**: `localhost`
   - **Server Port**: `8888`
   - **Lot Size**: `0.01`
   - **Buy Threshold**: `0.6`
   - **Sell Threshold**: `0.6`

### Automated Startup

Use the startup script to launch all services:

```bash
./start_natron.sh
```

This starts:
- API server (port 5000)
- Socket server (port 8888)
- Monitoring system

Stop with:
```bash
./stop_natron.sh
```

## 📈 Monitoring

Check system health and performance:

```bash
# One-time check
python monitor_natron.py --once

# Continuous monitoring (60s interval)
python monitor_natron.py --interval 60
```

## 🧮 Feature Engineering

The system automatically generates ~100 technical features:

- **Moving Averages** (13): MA, EMA, slopes, crossovers
- **Momentum** (13): RSI, ROC, CCI, Stochastic, MACD
- **Volatility** (15): ATR, Bollinger Bands, Keltner Channels
- **Volume** (9): OBV, VWAP, MFI, volume ratios
- **Price Patterns** (8): Doji, gaps, shadows, body%
- **Returns** (8): Log returns, intraday, cumulative
- **Trend Strength** (6): ADX, +DI, -DI, Aroon
- **Statistical** (6): Skewness, kurtosis, z-score, Hurst
- **Support/Resistance** (4): Distance to highs/lows
- **SMC** (6): Swing highs/lows, BOS/CHOCH
- **Market Profile** (10): POC, VAH, VAL, entropy

## 🏷️ Labeling Strategy

### Buy Signal (≥2 conditions):
1. `close > MA20 > MA50`
2. `RSI > 50` or recently left oversold
3. `close > BB midband` and `MA20_slope > 0`
4. `volume > 1.5 × rolling20`
5. `close near high (≥70%)`
6. `MACD_hist > 0` and increasing

### Sell Signal (≥2 conditions):
1. `close < MA20 < MA50`
2. `RSI < 50` or turning down from overbought
3. `close < BB midband` and `MA20_slope < 0`
4. `volume > 1.5 × rolling20`, `close near low (≤30%)`
5. `MACD_hist < 0` and decreasing

### Regime Classification:
- **BULL_STRONG**: trend > +2%, ADX > 25
- **BULL_WEAK**: 0 < trend ≤ 2%, ADX ≤ 25
- **RANGE**: lateral market
- **BEAR_WEAK**: −2% ≤ trend < 0, ADX ≤ 25
- **BEAR_STRONG**: trend < −2%, ADX > 25
- **VOLATILE**: ATR > 90th percentile or volume spike

## 📁 Project Structure

```
/workspace/
├── feature_engine.py          # Feature engineering module
├── label_generator.py         # Label generation
├── dataset_loader.py           # PyTorch dataset and loaders
├── model_natron.py            # Transformer model architecture
├── losses.py                  # Loss functions
├── train_pretrain.py          # Phase 1: Pretraining
├── train_natron.py            # Phase 2: Supervised fine-tuning
├── train_rl.py                # Phase 3: Reinforcement learning
├── api_server.py              # Flask REST API
├── socket_server.py           # TCP socket server (MQL5)
├── natron_ea.mq5              # MetaTrader 5 Expert Advisor
├── monitor_natron.py          # System monitoring
├── config.yaml                # Configuration file
├── requirements.txt           # Python dependencies
├── start_natron.sh            # Startup script
├── stop_natron.sh             # Stop script
└── README.md                  # This file
```

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model architecture (dimensions, layers, heads)
- Training parameters (learning rates, batch sizes)
- API and socket server settings
- MQL5 EA parameters

## 🧪 Testing

### Test Feature Engineering

```python
from feature_engine import FeatureEngine
import pandas as pd

df = pd.read_csv('data_export.csv')
engine = FeatureEngine()
features_df = engine.fit_transform(df)
print(f"Features shape: {features_df.shape}")
```

### Test Model Inference

```python
import torch
from model_natron import NatronTransformer

model = NatronTransformer(num_features=100, sequence_length=96)
x = torch.randn(1, 96, 100)  # (batch, seq_len, features)
outputs = model.predict(x)
print(outputs)
```

## 📊 Performance Metrics

The model tracks:
- **Buy/Sell Accuracy**: Classification accuracy for signals
- **Direction Accuracy**: Up/down prediction accuracy
- **Regime Accuracy**: Market regime classification accuracy
- **Multi-task Loss**: Combined loss across all tasks

## 🚨 Troubleshooting

### Model Not Loading
- Check model path in config
- Ensure model file exists and is compatible version
- Verify feature columns match training data

### Socket Connection Failed
- Ensure socket server is running: `python socket_server.py`
- Check firewall settings
- Verify host/port in MQL5 EA settings

### CUDA Out of Memory
- Reduce batch size in training scripts
- Use gradient accumulation
- Train on CPU: `--device cpu`

### MQL5 EA Not Connecting
- Verify Python server is running
- Check network connectivity
- Review socket server logs: `logs/socket_server.log`

## 📝 License

This project is provided as-is for research and educational purposes.

## 🤝 Contributing

This is a complete end-to-end system. To extend:
1. Add new features in `feature_engine.py`
2. Modify labeling rules in `label_generator.py`
3. Adjust model architecture in `model_natron.py`
4. Customize training in `train_*.py` scripts

## 📚 References

- Transformer Architecture: "Attention Is All You Need" (Vaswani et al., 2017)
- Multi-Task Learning: Caruana (1997)
- PPO Algorithm: Schulman et al. (2017)

---

**Built for GPU-accelerated trading on Linux servers**

For questions or issues, check logs in `./logs/` directory.
