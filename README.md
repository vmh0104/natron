# 🧠 Natron Transformer – Multi-Task Financial Trading Model

End-to-End GPU Pipeline for AI-Powered Trading

## Overview

Natron is a sophisticated Transformer-based deep learning system designed for financial market analysis and automated trading. It learns multiple market representations simultaneously:

- **Buy/Sell Classification**: Detects optimal entry/exit points
- **Directional Prediction**: Forecasts price movement direction
- **Market Regime Classification**: Identifies 6 distinct market states

## Architecture

### Three-Layer Intelligence

1. **Structure Understanding (Pretraining)**
   - Masked token modeling
   - Contrastive learning
   - Learns latent market dynamics

2. **Signal Recognition (Supervised)**
   - Multi-task Transformer heads
   - Simultaneous prediction of buy/sell, direction, and regime

3. **Behavioral Adaptation (Reinforcement Learning)**
   - PPO algorithm
   - Optimizes for profit while managing risk

## System Components

### Core Modules

- `feature_engine.py`: Generates ~100 technical features from OHLCV data
- `label_generator.py`: Creates trading labels using institutional rules
- `dataset_loader.py`: Constructs sequences of 96 consecutive candles
- `model_natron.py`: Transformer model architecture
- `losses.py`: Multi-task, contrastive, and RL loss functions

### Training Scripts

- `train_pretrain.py`: Phase 1 - Pretraining (masked modeling/contrastive)
- `train_supervised.py`: Phase 2 - Supervised fine-tuning
- `rl_trading.py`: Phase 3 - Reinforcement learning (PPO)

### Inference Servers

- `api_server.py`: Flask REST API server (`/predict` endpoint)
- `socket_server.py`: TCP/JSON server for MQL5 integration

### Trading Integration

- `natron_ea.mq5`: MetaTrader 5 Expert Advisor

### Deployment

- `Dockerfile`: Containerized deployment
- `start_natron.sh`: Startup script
- `monitor_natron.py`: System monitoring

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (if using GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Prepare Data

Place your OHLCV data in `data_export.csv` with columns:
- `time`: Timestamp
- `open`, `high`, `low`, `close`: Price data
- `volume`: Trading volume

### 3. Training

#### Phase 1: Pretraining (Optional)

```bash
python train_pretrain.py \
    --data_path data_export.csv \
    --method masked \
    --epochs 50 \
    --batch_size 32 \
    --save_dir ./checkpoints
```

#### Phase 2: Supervised Fine-Tuning

```bash
python train_supervised.py \
    --data_path data_export.csv \
    --pretrained_path ./checkpoints/pretrain_best.pt \
    --epochs 100 \
    --batch_size 32 \
    --save_dir ./checkpoints
```

### 4. Inference

#### Flask API Server

```bash
python api_server.py \
    --model_path ./checkpoints/natron_v2.pt \
    --host 0.0.0.0 \
    --port 5000
```

#### Socket Server (for MQL5)

```bash
python socket_server.py \
    --model_path ./checkpoints/natron_v2.pt \
    --host localhost \
    --port 8888
```

### 5. MetaTrader 5 Integration

1. Copy `natron_ea.mq5` to MetaTrader 5's `Experts` folder
2. Compile the EA in MetaEditor
3. Configure server host/port in EA inputs
4. Attach EA to chart

### 6. Start Complete System

```bash
chmod +x start_natron.sh
./start_natron.sh
```

## API Usage

### Prediction Endpoint

**POST** `/predict`

Request:
```json
{
  "candles": [
    {
      "time": 1234567890,
      "open": 1.0,
      "high": 1.1,
      "low": 0.9,
      "close": 1.05,
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
  "regime": "BULL_WEAK",
  "regime_id": 1,
  "confidence": 0.82
}
```

### Health Check

**GET** `/health`

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## Configuration

Edit `config.yaml` to customize:

- Model architecture (dimensions, layers, etc.)
- Training hyperparameters
- Loss weights
- API/socket server settings

## Monitoring

```bash
# Run once
python monitor_natron.py --once

# Continuous monitoring (every 60 seconds)
python monitor_natron.py --interval 60
```

## Docker Deployment

```bash
# Build image
docker build -t natron:latest .

# Run container
docker run -d \
    --gpus all \
    -p 5000:5000 \
    -p 8888:8888 \
    -v $(pwd)/checkpoints:/app/checkpoints \
    natron:latest
```

## Model Output Interpretation

### Buy/Sell Probabilities
- `buy_prob > 0.6`: Strong buy signal
- `sell_prob > 0.6`: Strong sell signal
- Both low: Hold/wait

### Direction
- `direction_up > 0.5`: Upward movement expected
- `direction_up < 0.5`: Downward movement expected

### Regime Classes
- `BULL_STRONG`: Strong uptrend (trend > +2%, ADX > 25)
- `BULL_WEAK`: Weak uptrend (0 < trend ≤ 2%, ADX ≤ 25)
- `RANGE`: Lateral/consolidation market
- `BEAR_WEAK`: Weak downtrend (-2% ≤ trend < 0, ADX ≤ 25)
- `BEAR_STRONG`: Strong downtrend (trend < -2%, ADX > 25)
- `VOLATILE`: High volatility (ATR > 90th percentile)

## Performance Optimization

- **GPU**: Ensure CUDA is available for training and inference
- **Batch Size**: Adjust based on GPU memory (32-128 typical)
- **Sequence Length**: Fixed at 96 candles (can be modified in config)
- **Model Size**: Default 256-dim, 6 layers (adjustable)

## Troubleshooting

### Model Not Found
Train a model first using `train_supervised.py`

### CUDA Out of Memory
- Reduce batch size
- Use gradient accumulation
- Reduce model dimensions

### Socket Connection Failed
- Check firewall settings
- Verify server is running
- Check host/port configuration

## License

This project is provided as-is for research and educational purposes.

## Disclaimer

Trading involves substantial risk. This system is for educational purposes only. Past performance does not guarantee future results. Always test thoroughly before live trading.

## Support

For issues and questions, refer to the code documentation and inline comments.

---

**Built with PyTorch 2.x | CUDA Optimized | Production Ready**
