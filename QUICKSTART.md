# 🚀 Natron Transformer - Quick Start Guide

## Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- Linux/Ubuntu/Debian
- MetaTrader 5 (for MQL5 integration, optional)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify PyTorch CUDA (if using GPU)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step 1: Prepare Data

Option A: Use your own data
```bash
# Place your OHLCV CSV at data_export.csv
# Required columns: time, open, high, low, close, volume
```

Option B: Generate sample data
```bash
python create_sample_data.py
# Creates data_export.csv with 500 sample candles
```

## Step 2: Train Model

```bash
python train.py
```

This will:
- Generate ~100 technical features
- Create buy/sell/direction/regime labels
- Build sequences of 96 candles
- **Phase 1**: Pretrain with masked modeling (50 epochs)
- **Phase 2**: Supervised fine-tuning (100 epochs)
- Save model to `model/natron_v2.pt`

**Expected time**: 
- CPU: ~2-4 hours for 1000 samples
- GPU: ~10-30 minutes for 1000 samples

## Step 3: Test Inference

```bash
python test_inference.py
```

This verifies the model loads and makes predictions correctly.

## Step 4: Deploy API Server

### Option A: Flask REST API

```bash
python api_server.py
```

Test with:
```bash
curl http://localhost:5000/health
```

### Option B: MQL5 Socket Server

```bash
python mql5_socket_server.py
```

Then deploy `NatronEA.mq5` in MetaTrader 5.

## Step 5: Use MQL5 EA (Optional)

1. Copy `NatronEA.mq5` to `MetaTrader 5/MQL5/Experts/`
2. Open MetaEditor and compile
3. Attach to chart:
   - Server IP: `127.0.0.1` (or your server IP)
   - Server Port: `8888`
   - Timeframe: M15 or H1
   - Enable Trading: true

## Troubleshooting

### "CUDA out of memory"
- Reduce `BATCH_SIZE` in `config.py`
- Use CPU: Set `DEVICE = "cpu"` in `config.py`

### "Not enough bars"
- Ensure you have at least 200+ candles in your data
- Model needs 96 candles for each prediction

### "Model not found"
- Train the model first: `python train.py`
- Check `model/natron_v2.pt` exists

### MQL5 connection issues
- Verify Python server is running: `python mql5_socket_server.py`
- Check firewall allows port 8888
- Test with: `telnet <server_ip> 8888`

## File Structure

```
/workspace/
├── config.py              # Configuration
├── feature_engine.py      # Feature generation
├── label_generator.py     # Label creation
├── sequence_creator.py    # Sequence building
├── model.py               # Transformer model
├── train.py               # Training pipeline
├── api_server.py          # Flask API
├── mql5_socket_server.py  # Socket server
├── NatronEA.mq5           # MQL5 EA
├── test_inference.py      # Test script
├── create_sample_data.py  # Sample data generator
└── requirements.txt       # Dependencies
```

## Next Steps

1. **Fine-tune hyperparameters** in `config.py`
2. **Add more features** in `feature_engine.py`
3. **Adjust labeling rules** in `label_generator.py`
4. **Implement full RL** in `rl_trainer.py` (currently placeholder)
5. **Backtest** your model on historical data
6. **Deploy** to production GPU server

## Performance Tips

- Use GPU for training (10-50x faster)
- Increase `BATCH_SIZE` if you have GPU memory
- Use `NUM_WORKERS=4` in DataLoader for faster data loading
- Monitor training with TensorBoard (add logging)

## Support

Check `README.md` for detailed documentation.
