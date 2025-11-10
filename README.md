# 🧠 Natron Transformer – Multi-Task Financial Trading Model

End-to-End GPU Pipeline for Financial Trading with Transformer Architecture

## 🎯 Overview

Natron Transformer is a deep learning system that learns multiple market representations and trading decisions from OHLCV candle sequences:

- **Buy/Sell Classification**: Binary signals for entry/exit
- **Directional Prediction**: Up/down movement forecasting
- **Market Regime Classification**: 6-class market state detection

## 🏗️ Architecture

### 3-Layer Intelligence

1. **Structure Understanding (Pretraining)**: Masked modeling learns hidden market dynamics
2. **Signal Recognition (Supervised)**: Multi-task fine-tuning detects trading patterns
3. **Behavioral Adaptation (RL)**: Optional reinforcement learning for profit optimization

### Model Components

- **Transformer Encoder**: 6 layers, 8 heads, 256 dimensions
- **Multi-Task Heads**: Buy/Sell (sigmoid), Direction (softmax-2), Regime (softmax-6)
- **Feature Engineering**: ~100 technical indicators automatically generated

## 📁 Project Structure

```
/workspace/
├── config.py                 # Configuration parameters
├── feature_engine.py         # ~100 technical features
├── label_generator.py        # Buy/sell/direction/regime labels
├── sequence_creator.py       # Sequence construction (96 candles)
├── model.py                  # Transformer architecture
├── train.py                  # Training pipeline
├── api_server.py             # Flask REST API
├── mql5_socket_server.py     # Socket server for MQL5
├── NatronEA.mq5              # MetaTrader 5 Expert Advisor
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your OHLCV data in `data_export.csv` with columns:
- `time`: Timestamp
- `open`, `high`, `low`, `close`: Price data
- `volume`: Volume data

### 3. Train Model

```bash
python train.py
```

This will:
1. Generate ~100 technical features
2. Create buy/sell/direction/regime labels
3. Build sequences of 96 candles
4. Pretrain with masked modeling
5. Fine-tune with supervised learning
6. Save model to `model/natron_v2.pt`

### 4. Run API Server

```bash
python api_server.py
```

API endpoints:
- `GET /health`: Health check
- `POST /predict`: Single prediction
- `POST /predict_batch`: Batch predictions

### 5. Run MQL5 Socket Server

```bash
python mql5_socket_server.py
```

This server communicates with the MetaTrader 5 Expert Advisor.

### 6. Deploy MQL5 EA

1. Copy `NatronEA.mq5` to MetaTrader 5 `Experts` folder
2. Compile in MetaEditor
3. Attach to chart with appropriate settings:
   - Server IP: Your Python server IP
   - Server Port: 8888
   - Timeframe: M15 or H1

## 📊 Feature Groups

| Group | Count | Description |
|-------|-------|-------------|
| Moving Average | 13 | MA, EMA, slopes, crossovers |
| Momentum | 13 | RSI, ROC, CCI, Stochastic, MACD |
| Volatility | 15 | ATR, Bollinger Bands, Keltner, StdDev |
| Volume | 9 | OBV, VWAP, MFI, Volume ratios |
| Price Pattern | 8 | Doji, Gaps, Shadows, Body% |
| Returns | 8 | Log return, intraday, cumulative |
| Trend Strength | 6 | ADX, +DI, -DI, Aroon |
| Statistical | 6 | Skewness, Kurtosis, Z-score, Hurst |
| Support/Resistance | 4 | Distance to High/Low |
| SMC | 6 | Swing High/Low, BOS/CHOCH |
| Market Profile | 10 | POC, VAH, VAL, Entropy |

## 🎯 Labeling Rules

### Buy Signal (≥2 conditions):
- close > MA20 > MA50
- RSI > 50 or recently left oversold (<30)
- close > BB midband and MA20_slope > 0
- volume > 1.5 × rolling20
- close near high (≥70%)
- MACD_hist > 0 and increasing

### Sell Signal (≥2 conditions):
- close < MA20 < MA50
- RSI < 50 or turning down from overbought (>70)
- close < BB midband and MA20_slope < 0
- volume > 1.5 × rolling20, close near low (≤30%)
- MACD_hist < 0 and decreasing

### Regime Classes:
- **BULL_STRONG**: trend > +2%, ADX > 25
- **BULL_WEAK**: 0 < trend ≤ 2%, ADX ≤ 25
- **RANGE**: lateral market
- **BEAR_WEAK**: -2% ≤ trend < 0, ADX ≤ 25
- **BEAR_STRONG**: trend < -2%, ADX > 25
- **VOLATILE**: ATR > 90th percentile or volume spike

## 🔧 Configuration

Edit `config.py` to adjust:
- Model architecture (d_model, n_heads, n_layers)
- Training parameters (batch_size, learning_rate, epochs)
- Multi-task weights
- API/Socket server ports

## 📡 API Usage

### Single Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "candles": [
      {"time": "2024-01-01 00:00", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 1000},
      ...
    ]
  }'
```

Response:
```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_up": 0.69,
  "regime": "BULL_WEAK",
  "confidence": 0.82
}
```

## 🖥️ System Requirements

- **OS**: Ubuntu/Debian (Linux)
- **Python**: 3.10+
- **GPU**: CUDA-capable (recommended)
- **RAM**: 8GB+ (16GB+ recommended)
- **Storage**: 10GB+ for model and data

## 📝 Notes

- Model requires sequences of exactly 96 candles
- Ensure sufficient historical data (at least 200+ candles recommended)
- GPU acceleration significantly speeds up training
- MQL5 EA requires MetaTrader 5 terminal

## 🔒 License

Proprietary - Natron AI

## 📧 Support

For issues or questions, refer to the code documentation or contact support.
