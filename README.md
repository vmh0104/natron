# 🧠 Natron Transformer – Multi-Task Financial Trading Model

End-to-end GPU pipeline for training a multi-task Transformer model for financial trading using sequences of 96 consecutive OHLCV candles.

## 🎯 Features

- **100 Technical Features**: Comprehensive feature engineering (MA, RSI, MACD, Bollinger Bands, SMC, Market Profile, etc.)
- **Multi-Task Learning**: Simultaneous prediction of Buy/Sell signals, Direction, and Market Regime (6 classes)
- **3-Phase Training**:
  1. **Pretraining**: Masked modeling + Contrastive learning
  2. **Supervised Fine-Tuning**: Multi-task classification
  3. **Reinforcement Learning**: PPO for trading optimization
- **Production-Ready API**: Flask REST API for inference
- **MQL5 Integration**: Real-time socket server for MetaTrader 5 Expert Advisor

## 📋 Requirements

- Python 3.10+
- PyTorch 2.x with CUDA support
- Ubuntu/Debian Linux (or GCP/Vertex AI)

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

### 3. Configure Training

Edit `config.yaml` to adjust model architecture, training parameters, etc.

### 4. Train Model

```bash
python train.py --data data_export.csv --config config.yaml
```

This will run all enabled phases:
- Phase 1: Pretraining (if enabled)
- Phase 2: Supervised fine-tuning
- Phase 3: RL training (if enabled)

### 5. Run API Server

```bash
python api_server.py --model_path models/natron_v2.pt --port 5000
```

### 6. Run MQL5 Socket Server

```bash
python mql5_socket_server.py --host localhost --port 8888 --model_path models/natron_v2.pt
```

## 📁 Project Structure

```
.
├── feature_engine.py          # Technical feature generation (~100 features)
├── label_generator.py          # Buy/Sell/Direction/Regime labeling
├── sequence_creator.py         # Sequence construction (96 candles)
├── model.py                   # Transformer architecture
├── train_phase1_pretrain.py   # Pretraining (masked/contrastive)
├── train_phase2_supervised.py # Supervised fine-tuning
├── train_phase3_rl.py        # Reinforcement learning (PPO)
├── train.py                   # Main training pipeline
├── api_server.py              # Flask REST API
├── mql5_socket_server.py      # Socket server for MQL5
├── mql5_ea.mq5                # MetaTrader 5 Expert Advisor
├── config.yaml                # Configuration file
├── requirements.txt            # Python dependencies
└── README.md                  # This file
```

## 🔧 API Usage

### Predict Endpoint

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
  "direction_down": 0.31,
  "regime": "BULL_WEAK",
  "regime_id": 1,
  "regime_probs": [0.1, 0.4, 0.2, 0.1, 0.1, 0.1],
  "confidence": 0.82
}
```

## 📊 Model Output

The model predicts:

1. **Buy Signal** (0-1): Probability of buy opportunity
2. **Sell Signal** (0-1): Probability of sell opportunity
3. **Direction** (Up/Down): Price direction prediction
4. **Regime** (6 classes):
   - `BULL_STRONG`: Strong uptrend
   - `BULL_WEAK`: Weak uptrend
   - `RANGE`: Lateral/range-bound market
   - `BEAR_WEAK`: Weak downtrend
   - `BEAR_STRONG`: Strong downtrend
   - `VOLATILE`: High volatility period

## 🔌 MQL5 Integration

1. Copy `mql5_ea.mq5` to your MetaTrader 5 `Experts` folder
2. Compile the EA in MetaEditor
3. Start the Python socket server: `python mql5_socket_server.py`
4. Attach the EA to a chart in MT5
5. Configure EA parameters (server host, port, thresholds)

The EA will:
- Send candle data to the Python server
- Receive predictions
- Execute trades based on buy/sell signals

## ⚙️ Configuration

Key parameters in `config.yaml`:

- **Model**: Architecture dimensions, layers, dropout
- **Training**: Batch sizes, learning rates, epochs per phase
- **Data**: Sequence length, train/val/test splits
- **API/Socket**: Server ports and model paths

## 📈 Training Phases

### Phase 1: Pretraining
- **Masked Modeling**: Reconstruct masked tokens in sequences
- **Contrastive Learning**: Learn similar/dissimilar market patterns

### Phase 2: Supervised Fine-Tuning
- Multi-task classification with weighted losses
- Optional encoder freezing

### Phase 3: Reinforcement Learning (Optional)
- PPO algorithm
- Reward: Profit - α×turnover - β×drawdown
- Optimizes real trading performance

## 🎓 Model Philosophy

Natron learns market grammar through 3 layers:

1. **Structure Understanding** (Pretrain): Hidden dynamics of market sequences
2. **Signal Recognition** (Supervised): Patterns leading to directional outcomes
3. **Behavioral Adaptation** (RL): Continuous optimization for profit/risk balance

## 📝 Notes

- Ensure GPU availability for training (CUDA)
- Model requires exactly 96 candles for prediction
- Feature normalization is applied automatically
- All NaN/inf values are handled gracefully

## 🔒 License

This project is provided as-is for educational and research purposes.

## 🤝 Contributing

Contributions welcome! Please ensure code follows the existing style and includes tests.
