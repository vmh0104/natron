# 🚀 Natron Transformer - Quick Start Guide

Get up and running with Natron in 5 minutes!

---

## 📋 Prerequisites

- **OS**: Ubuntu 20.04+ or Debian 11+
- **GPU**: NVIDIA GPU with CUDA 12.1+ (optional but recommended)
- **Python**: 3.10+
- **RAM**: 16GB+ recommended
- **Disk**: 10GB+ free space

---

## ⚡ Quick Installation

### 1. Clone or navigate to workspace
```bash
cd /workspace
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare your data
Place your OHLCV data in `data/data_export.csv` with columns:
```
time, open, high, low, close, volume
```

---

## 🎯 Usage

### Option A: Train Model (All Phases)
```bash
python src/train_natron.py --config config.yaml --phase all
```

This will run:
1. **Pretraining** (unsupervised learning)
2. **Supervised training** (multi-task learning)
3. **Reinforcement learning** (trading optimization)

### Option B: Train Individual Phases
```bash
# Phase 1: Pretraining only
python src/train_natron.py --phase pretrain

# Phase 2: Supervised only
python src/train_natron.py --phase supervised

# Phase 3: RL only
python src/train_natron.py --phase reinforcement
```

### Option C: Use Startup Script
```bash
# Train model
./deployment/start_natron.sh --mode train

# Start API server
./deployment/start_natron.sh --mode api

# Start socket server (for MQL5)
./deployment/start_natron.sh --mode socket

# Start all services
./deployment/start_natron.sh --mode all

# Use Docker
./deployment/start_natron.sh --mode docker
```

---

## 🌐 Start Services

### 1. Flask API Server
```bash
python api/api_server.py --config config.yaml
```

**Endpoints:**
- `GET /health` - Health check
- `POST /predict` - Get trading prediction
- `GET /model_info` - Model information

**Test API:**
```bash
curl http://localhost:5000/health
```

### 2. Socket Server (for MQL5)
```bash
python api/socket_server.py --config config.yaml
```

Listens on `localhost:9090` for MQL5 connections.

---

## 🎮 MetaTrader 5 Integration

### 1. Copy EA to MT5
```bash
cp mql5/natron_ea.mq5 ~/MetaTrader5/MQL5/Experts/
```

### 2. Compile in MetaEditor

### 3. Configure EA
- **Server Host**: `127.0.0.1`
- **Server Port**: `9090`
- **Sequence Length**: `96`
- **Timeframe**: `M15` or `H1`
- **Lot Size**: `0.1`

### 4. Attach to Chart
The EA will automatically connect to the Python socket server and start trading!

---

## 📊 Monitor System

```bash
python deployment/monitor_natron.py
```

This displays:
- CPU/GPU usage
- Memory usage
- Service health
- Active processes

---

## 🐳 Docker Deployment

### Build and run
```bash
# Start all services
docker-compose -f deployment/docker-compose.yml up -d

# View logs
docker-compose -f deployment/docker-compose.yml logs -f

# Stop services
docker-compose -f deployment/docker-compose.yml down
```

---

## 📈 View Training Progress

```bash
# Start TensorBoard
tensorboard --logdir=runs --port=6006

# Open browser
http://localhost:6006
```

---

## 🧪 Test Inference

```python
from api.inference import NatronInference
import pandas as pd

# Load model
engine = NatronInference('models/natron_v2.pt')

# Prepare data (96 candles)
df = pd.read_csv('data/data_export.csv').tail(96)

# Predict
result = engine.predict(df)

print(result)
# {
#   'buy_prob': 0.71,
#   'sell_prob': 0.24,
#   'direction_up': 0.69,
#   'regime': 'BULL_WEAK',
#   'confidence': 0.82,
#   'signal': 'BUY'
# }
```

---

## 🔧 Configuration

Edit `config.yaml` to customize:
- Model architecture (layers, heads, dimensions)
- Training hyperparameters
- Feature engineering settings
- API/socket ports
- Loss weights

---

## 📝 Common Issues

### Issue: CUDA out of memory
**Solution**: Reduce batch size in `config.yaml`:
```yaml
training:
  pretrain:
    batch_size: 128  # Reduce this
  supervised:
    batch_size: 64   # Reduce this
```

### Issue: Model not found
**Solution**: Train model first:
```bash
python src/train_natron.py --phase all
```

### Issue: API connection refused
**Solution**: Make sure server is running:
```bash
python api/api_server.py --config config.yaml
```

### Issue: Socket timeout in MT5
**Solution**: Check firewall and socket server:
```bash
# Test socket connection
python -c "import socket; s=socket.socket(); s.connect(('localhost', 9090)); print('Connected!')"
```

---

## 📚 Next Steps

1. **Customize Features**: Edit `src/feature_engine.py` to add custom indicators
2. **Tune Hyperparameters**: Modify `config.yaml` for your data
3. **Deploy to Production**: Use Docker for production deployment
4. **Monitor Performance**: Track metrics via TensorBoard
5. **Backtest**: Evaluate on historical data before live trading

---

## 🆘 Support

- **Documentation**: See `README.md` for detailed docs
- **Issues**: Check logs in `logs/natron.log`
- **Configuration**: Review `config.yaml` for all settings

---

## ⚠️ Important Notes

- **Never trade with real money without thorough testing**
- **Monitor GPU temperature during training**
- **Backup models regularly**
- **Keep your data secure**
- **Review all trades before going live**

---

**Ready to trade with AI? Let's go! 🚀**
