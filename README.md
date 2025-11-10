# 🧠 Natron Transformer – Multi-Task Financial Trading Model

End-to-end GPU pipeline for training and deploying a multi-task Transformer model for financial trading.

## 🎯 Features

- **~100 Technical Features**: Comprehensive feature engineering
- **Multi-Task Learning**: Buy/Sell classification, Direction prediction, Regime classification
- **3-Phase Training**: Pretraining → Supervised Fine-tuning → RL (optional)
- **Real-time Inference**: Flask API + Socket server for MQL5 integration
- **GPU Optimized**: PyTorch 2.x with CUDA support

## 📋 Requirements

- Python 3.10+
- PyTorch 2.x with CUDA
- Ubuntu/Debian Linux (or compatible)
- MetaTrader 5 (for MQL5 EA)

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
# Train both phases (pretrain + supervised)
python train.py --data data_export.csv --phase both

# Or train separately
python train.py --data data_export.csv --phase 1  # Pretraining only
python train.py --data data_export.csv --phase 2  # Supervised only
```

### 4. Start API Server

```bash
python api_server.py --model models/natron_v2.pt --port 5000
```

### 5. Start Socket Server (for MQL5)

```bash
python socket_server.py --host localhost --port 8888 --model models/natron_v2.pt
```

### 6. Deploy MQL5 EA

1. Copy `mql5_ea.mq5` to MetaTrader 5 `Experts` folder
2. Compile in MetaEditor
3. Attach to chart with appropriate settings

## 📁 Project Structure

```
.
├── feature_engine.py      # Feature extraction (~100 features)
├── label_generator.py      # Label generation (buy/sell/direction/regime)
├── sequence_creator.py     # Sequence construction
├── model.py               # Transformer model architecture
├── trainer.py             # Training pipeline (Phase 1, 2, 3)
├── train.py               # Main training script
├── api_server.py          # Flask API for inference
├── socket_server.py       # Socket server for MQL5
├── mql5_ea.mq5           # MetaTrader 5 Expert Advisor
├── config.yaml           # Configuration file
└── requirements.txt       # Python dependencies
```

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model architecture (dimensions, layers, etc.)
- Training parameters (batch size, learning rate, epochs)
- API and socket server settings

## 📊 Model Output

The model returns:
```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_up": 0.69,
  "regime": "BULL_WEAK",
  "confidence": 0.82
}
```

## 🎓 Training Phases

### Phase 1: Pretraining
- Masked modeling: Reconstruct masked tokens
- Contrastive learning: Learn market representations

### Phase 2: Supervised Fine-tuning
- Multi-task heads: Buy/Sell/Direction/Regime
- Weighted loss function

### Phase 3: Reinforcement Learning (Optional)
- PPO/SAC algorithm
- Reward: profit - turnover - drawdown

## 📝 API Usage

### Flask API

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "candles": [
      {"time": "...", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 1000},
      ...
    ]
  }'
```

### Socket Server (MQL5)

The MQL5 EA automatically connects to the socket server and sends candle data for predictions.

## ⚠️ Notes

- Ensure you have sufficient GPU memory for training
- The model requires sequences of 96 consecutive candles
- Adjust `MinConfidence` in MQL5 EA based on your risk tolerance
- Always backtest before live trading

## 📄 License

MIT License
