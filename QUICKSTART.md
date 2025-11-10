# 🚀 Natron V2 Quick Start Guide

Get up and running with Natron V2 in under 10 minutes!

---

## Step 1: Installation (2 minutes)

```bash
# Clone repository
cd /path/to/natron-v2

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or without GPU (CPU only)
pip install torch torchvision torchaudio
```

---

## Step 2: Prepare Data (1 minute)

Place your OHLCV data in `data/data_export.csv`:

```csv
time,open,high,low,close,volume
1699000000,1.0850,1.0865,1.0840,1.0858,15000
1699000900,1.0858,1.0870,1.0850,1.0862,12500
...
```

**Note**: You need at least 1,000 candles, but 10,000+ is recommended for best results.

---

## Step 3: Train the Model (5+ minutes)

```bash
# Full training pipeline (all 3 phases)
python train_natron.py --data data/data_export.csv

# Or skip pretraining and RL for faster results
python train_natron.py --data data/data_export.csv --skip-pretrain --skip-rl
```

**Training Time Estimates:**
- With GPU: 30-60 minutes (full pipeline)
- With CPU: 2-4 hours (full pipeline)
- Fast mode (supervised only): 10-20 minutes on GPU

**Output**: Model saved to `model/natron_v2.pt`

---

## Step 4: Start Inference Server (30 seconds)

```bash
# Start socket server (for MetaTrader 5)
python src/inference/socket_server.py

# Or start Flask API (for HTTP requests)
python src/inference/flask_api.py
```

Server will start on:
- Socket: `localhost:9090`
- Flask: `localhost:5000`

---

## Step 5: Connect MetaTrader 5 (2 minutes)

1. **Copy EA to MT5:**
   ```bash
   cp mql5/natron_ea.mq5 /path/to/MT5/MQL5/Experts/
   ```

2. **Compile in MetaEditor:**
   - Open `natron_ea.mq5` in MetaEditor
   - Press F7 to compile

3. **Attach to Chart:**
   - Drag "NatronEA" from Navigator → Expert Advisors
   - Configure settings (defaults work fine)
   - Enable "AutoTrading" button

**Done!** Natron is now analyzing the market and generating signals.

---

## Quick Test: Make a Prediction

### Using Python (Socket)

```python
import socket
import json

# Prepare dummy data (96 candles)
data = {
    "symbol": "EURUSD",
    "timeframe": "M15",
    "data": [[1699000000 + i*900, 1.0850, 1.0865, 1.0840, 1.0858, 1000] for i in range(96)]
}

# Send request
s = socket.socket()
s.connect(('localhost', 9090))
s.sendall((json.dumps(data) + '\n').encode())

# Receive response
response = s.recv(4096).decode()
print(json.dumps(json.loads(response), indent=2))
s.close()
```

### Using cURL (Flask)

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      [1699000000, 1.0850, 1.0865, 1.0840, 1.0858, 1000],
      # ... 95 more candles ...
    ]
  }'
```

**Expected Output:**
```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_up": 0.69,
  "regime": "BULL_WEAK",
  "confidence": 0.82
}
```

---

## Monitoring Training Progress

Open Tensorboard to visualize training:

```bash
tensorboard --logdir logs
```

Navigate to `http://localhost:6006` in your browser.

---

## Common Commands Cheat Sheet

```bash
# Training
python train_natron.py --data data/data_export.csv                    # Full pipeline
python train_natron.py --phase supervised --skip-pretrain             # Supervised only
python train_natron.py --device cpu                                   # Force CPU

# Inference
python src/inference/socket_server.py                                 # Socket server
python src/inference/flask_api.py --port 5000                        # Flask API

# Monitoring
python deployment/monitor_natron.py                                   # System monitor
tensorboard --logdir logs                                             # Training viz

# Docker
docker-compose -f deployment/docker-compose.yml up -d                 # Start services
docker-compose -f deployment/docker-compose.yml logs -f natron-server # View logs
docker-compose -f deployment/docker-compose.yml down                  # Stop services
```

---

## Troubleshooting Quick Fixes

### "CUDA out of memory"
```yaml
# Edit config/config.yaml
training:
  batch_size: 16  # Reduce from 32
```

### "Model not found"
```bash
# Make sure training completed successfully
ls -lh model/natron_v2.pt

# If missing, retrain:
python train_natron.py --data data/data_export.csv --skip-pretrain --skip-rl
```

### "Socket connection refused"
```bash
# Check if server is running
netstat -an | grep 9090

# If not running, start it:
python src/inference/socket_server.py
```

### "Not enough data"
```
Minimum required: 96 + train_split candles
Example: 96 + (1000 * 0.7) = ~800 candles minimum
Recommended: 10,000+ candles
```

---

## Next Steps

1. **Backtest thoroughly** before live trading
2. **Adjust confidence thresholds** based on your risk tolerance
3. **Monitor performance** using the system monitor
4. **Optimize hyperparameters** in `config/config.yaml`
5. **Add more data** for better predictions

---

## Need Help?

- 📖 Full documentation: `README.md`
- 🐛 Found a bug? Open an issue
- 💡 Feature request? Let us know!

---

**Happy Trading! 🎉**
