# 🧠 Natron Transformer – Multi-Task Financial Trading Model

End-to-End GPU Pipeline for Financial Trading with Transformer Architecture

## Overview

Natron is a multi-task Transformer model that learns multiple market representations and trading decisions:
- **Buy/Sell Classification**: Binary signals for entry/exit
- **Directional Prediction**: Up/down movement forecasting
- **Market Regime Classification**: 6-class market state identification

The system processes sequences of 96 consecutive OHLCV candles and generates ~100 technical features automatically.

## Architecture

### 🧩 Phase 1: Pretraining (Unsupervised)
- **Masked Modeling**: Reconstruct masked tokens in sequences
- **Contrastive Learning**: Learn latent market representations
- **Goal**: Understand market structure and dynamics

### 🧩 Phase 2: Supervised Fine-Tuning
- **Multi-Head Transformer**: Simultaneous prediction of all tasks
- **Multi-Task Loss**: Weighted combination of buy/sell/direction/regime losses
- **Goal**: Detect patterns leading to trading outcomes

### 🧩 Phase 3: Reinforcement Learning (Optional)
- **Algorithm**: PPO or SAC
- **Reward Function**: Profit - α × turnover - β × drawdown
- **Goal**: Optimize real-world trading performance

## Installation

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

## Data Format

Input CSV file (`data_export.csv`) should have columns:
- `time`: Timestamp (YYYY-MM-DD HH:MM:SS)
- `open`: Open price
- `high`: High price
- `low`: Low price
- `close`: Close price
- `volume`: Volume

Example:
```csv
time,open,high,low,close,volume
2024-01-01 00:00:00,1.1000,1.1050,1.0990,1.1040,1000
2024-01-01 00:15:00,1.1040,1.1060,1.1030,1.1050,1200
...
```

## Training

### Full Pipeline (Pretraining + Supervised)

```bash
python natron/train_pipeline.py \
    --data data_export.csv \
    --config config.yaml \
    --phase both \
    --pretrain_epochs 10 \
    --supervised_epochs 50 \
    --batch_size 32 \
    --device cuda
```

### Pretraining Only

```bash
python natron/train_pipeline.py \
    --data data_export.csv \
    --phase pretrain \
    --pretrain_epochs 10
```

### Supervised Only

```bash
python natron/train_pipeline.py \
    --data data_export.csv \
    --phase supervised \
    --supervised_epochs 50
```

## Feature Engineering

The system automatically generates ~100 technical features:

| Group | Count | Examples |
|-------|-------|----------|
| Moving Averages | 13 | MA, EMA, slopes, crossovers |
| Momentum | 13 | RSI, ROC, CCI, Stochastic, MACD |
| Volatility | 15 | ATR, Bollinger Bands, Keltner |
| Volume | 9 | OBV, VWAP, MFI, ratios |
| Price Patterns | 8 | Doji, gaps, shadows, body% |
| Returns | 8 | Log return, intraday, cumulative |
| Trend Strength | 6 | ADX, +DI, -DI, Aroon |
| Statistical | 6 | Skewness, Kurtosis, Z-score, Hurst |
| Support/Resistance | 4 | Distance to High/Low 20-50 |
| SMC | 6 | Swing High/Low, BOS/CHOCH |
| Market Profile | 10 | POC, VAH, VAL, Entropy |

## Label Generation

### Buy Signals (≥2 conditions):
- `close > MA20 > MA50`
- `RSI > 50` or recently left oversold
- `close > BB midband` and `MA20_slope > 0`
- `volume > 1.5 × rolling20`
- `close near high (≥70%)`
- `MACD_hist > 0` and increasing

### Sell Signals (≥2 conditions):
- `close < MA20 < MA50`
- `RSI < 50` or turning down from overbought
- `close < BB midband` and `MA20_slope < 0`
- `volume > 1.5 × rolling20`, `close near low (≤30%)`
- `MACD_hist < 0` and decreasing

### Regime Classes (6 states):
- **0: BULL_STRONG** - trend > +2%, ADX > 25
- **1: BULL_WEAK** - 0 < trend ≤ 2%, ADX ≤ 25
- **2: RANGE** - lateral market
- **3: BEAR_WEAK** - -2% ≤ trend < 0, ADX ≤ 25
- **4: BEAR_STRONG** - trend < -2%, ADX > 25
- **5: VOLATILE** - ATR > 90th percentile or volume spike

## API Server

### Start Server

```bash
export MODEL_PATH=model/natron_v2.pt
export CONFIG_PATH=config.yaml
export HOST=0.0.0.0
export PORT=5000

python natron/api_server.py
```

### Predict Endpoint

**POST** `/predict`

Request:
```json
{
  "candles": [
    {
      "time": "2024-01-01 00:00:00",
      "open": 1.1000,
      "high": 1.1050,
      "low": 1.0990,
      "close": 1.1040,
      "volume": 1000
    },
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
  "direction_down": 0.31,
  "regime": "BULL_WEAK",
  "regime_id": 1,
  "regime_probs": {
    "BULL_STRONG": 0.05,
    "BULL_WEAK": 0.45,
    "RANGE": 0.20,
    "BEAR_WEAK": 0.15,
    "BEAR_STRONG": 0.10,
    "VOLATILE": 0.05
  },
  "confidence": 0.82
}
```

## MetaTrader 5 Integration

### Setup

1. **Copy EA to MT5**:
   - Copy `natron/mql5_ea.py` → `MetaTrader 5/MQL5/Experts/NatronEA.mq5`
   - Note: The file is Python-formatted; convert to proper MQL5 syntax (see below)

2. **Configure EA Parameters**:
   - `ServerIP`: Python server IP (e.g., `127.0.0.1` for localhost)
   - `ServerPort`: Python server port (default: `5000`)
   - `MagicNumber`: Unique identifier for trades
   - `LotSize`: Position size
   - `StopLoss`: Stop loss in points
   - `TakeProfit`: Take profit in points
   - `Timeframe`: Chart timeframe (M15, H1, etc.)
   - `BuyThreshold`: Minimum buy probability (default: 0.6)
   - `SellThreshold`: Minimum sell probability (default: 0.6)

3. **Start Python API Server**:
   ```bash
   python natron/api_server.py
   ```

4. **Attach EA to Chart**:
   - Drag `NatronEA` to your chart
   - Configure parameters
   - Enable AutoTrading

### MQL5 EA Code

The provided `mql5_ea.py` is a template. For production use, convert it to proper MQL5 syntax. Key components:
- Socket connection to Python server
- HTTP POST requests with candle data
- JSON parsing of predictions
- Trade execution based on signals

## Project Structure

```
workspace/
├── natron/
│   ├── feature_engine.py      # Feature generation (~100 features)
│   ├── label_generator.py     # Buy/sell/direction/regime labels
│   ├── sequence_creator.py    # 96-candle sequence builder
│   ├── model.py               # Transformer architecture
│   ├── training.py            # Pretraining & supervised trainers
│   ├── train_pipeline.py      # End-to-end training script
│   ├── api_server.py          # Flask API server
│   └── mql5_ea.py            # MQL5 Expert Advisor template
├── config.yaml                # Configuration file
├── requirements.txt           # Python dependencies
├── data_export.csv           # Input OHLCV data
└── model/
    └── natron_v2.pt          # Trained model (after training)
```

## Usage Examples

### Training on GPU

```bash
# Full training pipeline
python natron/train_pipeline.py \
    --data data_export.csv \
    --phase both \
    --pretrain_epochs 10 \
    --supervised_epochs 50 \
    --batch_size 32 \
    --device cuda
```

### Inference via API

```bash
# Start server
python natron/api_server.py

# Test prediction (in another terminal)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "candles": [
      {"time": "2024-01-01 00:00:00", "open": 1.1000, "high": 1.1050, 
       "low": 1.0990, "close": 1.1040, "volume": 1000},
      ...
    ]
  }'
```

## Performance Tips

1. **GPU Memory**: Adjust `batch_size` if running out of memory
2. **Sequence Length**: Default 96 candles; adjust in config if needed
3. **Feature Count**: Currently ~100 features; can be extended
4. **Model Size**: Adjust `d_model`, `num_layers` in config for speed/accuracy tradeoff

## License

MIT License

## Support

For issues and questions, please open an issue on GitHub.
