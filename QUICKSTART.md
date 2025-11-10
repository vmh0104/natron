# Natron Quick Start Guide

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- MetaTrader 5 (for trading integration)

## Step-by-Step Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. Prepare Your Data

Ensure `data_export.csv` exists with columns:
- `time`, `open`, `high`, `low`, `close`, `volume`

### 3. Train the Model

```bash
# Phase 1: Pretraining (optional but recommended)
python train_pretrain.py --data_path data_export.csv --epochs 50

# Phase 2: Supervised Fine-Tuning (required)
python train_supervised.py --data_path data_export.csv --epochs 100
```

The trained model will be saved to `./checkpoints/natron_v2.pt`

### 4. Start Inference Servers

**Option A: Start Both Servers**
```bash
./start_natron.sh
```

**Option B: Start Individually**

API Server:
```bash
python api_server.py --model_path ./checkpoints/natron_v2.pt
```

Socket Server (for MQL5):
```bash
python socket_server.py --model_path ./checkpoints/natron_v2.pt
```

### 5. Test the API

```bash
# Health check
curl http://localhost:5000/health

# Test prediction (requires 96 candles)
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

### 6. Connect MetaTrader 5

1. Copy `natron_ea.mq5` to `MetaTrader 5/MQL5/Experts/`
2. Open MetaEditor and compile the EA
3. Configure EA inputs:
   - `ServerHost`: `localhost` (or your server IP)
   - `ServerPort`: `8888`
   - `BuyThreshold`: `0.6`
   - `SellThreshold`: `0.6`
4. Attach EA to chart

### 7. Monitor System

```bash
python monitor_natron.py --once
```

## File Structure

```
/workspace/
├── feature_engine.py          # Feature generation (~100 features)
├── label_generator.py         # Label creation (buy/sell/regime)
├── dataset_loader.py          # Sequence construction
├── model_natron.py            # Transformer model
├── losses.py                  # Loss functions
├── train_pretrain.py          # Phase 1 training
├── train_supervised.py        # Phase 2 training
├── rl_trading.py             # Phase 3 RL (optional)
├── api_server.py             # Flask API
├── socket_server.py          # MQL5 socket server
├── natron_ea.mq5            # MetaTrader 5 EA
├── config.yaml               # Configuration
├── requirements.txt          # Dependencies
├── Dockerfile               # Docker deployment
├── start_natron.sh          # Startup script
├── monitor_natron.py        # Monitoring tool
├── README.md                # Full documentation
└── QUICKSTART.md           # This file
```

## Common Issues

**Issue**: Model not found
- **Solution**: Train a model first using `train_supervised.py`

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size in training scripts (e.g., `--batch_size 16`)

**Issue**: Socket connection failed (MQL5)
- **Solution**: Check firewall, verify server is running, check host/port

**Issue**: Import errors
- **Solution**: Ensure all dependencies are installed: `pip install -r requirements.txt`

## Next Steps

1. Train on your historical data
2. Backtest the model predictions
3. Fine-tune hyperparameters in `config.yaml`
4. Deploy to production server
5. Monitor performance with `monitor_natron.py`

## Support

Refer to `README.md` for detailed documentation.
