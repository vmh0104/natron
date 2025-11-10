# 🚀 Natron Transformer - Quick Start Guide

**Get up and running in 15 minutes!**

---

## ⚡ TL;DR - 5 Commands to Start

```bash
# 1. Setup
git clone https://github.com/yourusername/natron-transformer.git && cd natron-transformer
python3.10 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# 2. Add your data
cp /your/data/path.csv data/data_export.csv

# 3. Train (takes ~9 hours on GPU)
python scripts/train_full_pipeline.py

# 4. Start server
./scripts/start_server.sh

# 5. Test
curl -X POST http://localhost:5000/health
```

---

## 📦 What You Need

1. **Python 3.10+**
2. **NVIDIA GPU** (for training)
3. **OHLCV Data** (CSV format)
4. **30 minutes** of your time

---

## 🎯 Step-by-Step

### 1️⃣ Installation (5 min)

```bash
# Clone
git clone https://github.com/yourusername/natron-transformer.git
cd natron-transformer

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -r requirements.txt

# Verify GPU
python -c "import torch; print(torch.cuda.is_available())"
```

✅ **Expected**: `True` (if GPU available)

---

### 2️⃣ Prepare Data (2 min)

Your CSV should look like this:

```csv
time,open,high,low,close,volume
2023-01-01 00:00:00,1.0500,1.0520,1.0495,1.0510,15000
2023-01-01 00:15:00,1.0510,1.0530,1.0505,1.0525,18000
2023-01-01 00:30:00,1.0525,1.0540,1.0520,1.0535,20000
...
```

**Requirements**:
- Minimum 5,000 rows (more is better!)
- Chronologically ordered
- No missing values
- Consistent timeframe (M15, H1, etc.)

```bash
# Copy your data
cp /path/to/your/data.csv data/data_export.csv

# Quick check
head -n 5 data/data_export.csv
wc -l data/data_export.csv
```

---

### 3️⃣ Train Model (9 hours)

**Option A: Full Pipeline (Recommended)**

```bash
python scripts/train_full_pipeline.py --phases 1,2,3
```

This runs:
- ✨ **Phase 1**: Unsupervised pretraining (~3h)
- 🎯 **Phase 2**: Supervised training (~4h)
- 🤖 **Phase 3**: Reinforcement learning (~2h)

**Option B: Quick Test (Skip Phase 3)**

```bash
python scripts/train_full_pipeline.py --phases 1,2
```

**Option C: Individual Phases**

```bash
# Phase 1 only
python src/training/pretrain.py

# Phase 2 only (after Phase 1)
python src/training/train_supervised.py
```

**Monitor Training**:

```bash
# In another terminal
tensorboard --logdir logs/tensorboard
# Open: http://localhost:6006
```

---

### 4️⃣ Start Server (1 min)

```bash
./scripts/start_server.sh
```

Or manually:
```bash
python src/server/api_server.py
```

You should see:
```
🖥️ Using device: cuda
✅ Model loaded successfully
🌐 Starting Flask API server...
   REST API: http://0.0.0.0:5000
   Socket: 0.0.0.0:9090
```

---

### 5️⃣ Test Inference (2 min)

**Health Check**:
```bash
curl http://localhost:5000/health
```

Expected:
```json
{"status": "healthy", "model_loaded": true}
```

**Test Prediction**:

```bash
# Prepare test data (96 candles)
cat > test_data.json << 'EOF'
{
  "data": [
    {"time": "2023-01-01 00:00:00", "open": 1.05, "high": 1.052, "low": 1.049, "close": 1.051, "volume": 15000},
    ... (94 more candles)
    {"time": "2023-01-01 23:45:00", "open": 1.06, "high": 1.062, "low": 1.059, "close": 1.061, "volume": 18000}
  ]
}
EOF

# Make prediction
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @test_data.json
```

Expected response:
```json
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_up": 0.69,
  "regime": "BULL_WEAK",
  "confidence": 0.82,
  "signal": "BUY"
}
```

✅ **It works!**

---

## 🎮 MetaTrader 5 Integration (5 min)

### Step 1: Copy EA

```bash
# Find your MT5 Experts folder
# Windows: C:\Users\[You]\AppData\Roaming\MetaQuotes\Terminal\[ID]\MQL5\Experts
# macOS: ~/Library/Application Support/MetaTrader 5/Experts

# Copy the EA
cp mql5/natron_ea.mq5 [YOUR_MT5_PATH]/Experts/
```

### Step 2: Compile

1. Open **MetaEditor** (press F4 in MT5)
2. Open `natron_ea.mq5`
3. Click **Compile** (F7)
4. ✅ Should show: "0 errors, 0 warnings"

### Step 3: Attach to Chart

1. Open MT5
2. Open any chart (e.g., EURUSD M15)
3. Drag `NatronEA` from Navigator → Expert Advisors
4. Configure:
   - Server IP: `127.0.0.1` (or your server IP)
   - Server Port: `9090`
   - Lot Size: `0.01`
   - Enable Trading: `true`
5. Click **OK**

### Step 4: Verify

You should see:
- ✅ Panel on chart showing "NATRON AI TRADER"
- ✅ Signals updating every bar
- ✅ Terminal log: "Connected to Natron server"

---

## 🐛 Troubleshooting

### "CUDA out of memory"

**Solution**: Reduce batch size in `config/config.yaml`:

```yaml
supervised:
  batch_size: 16  # Was 32
```

### "Model file not found"

**Solution**: Verify training completed:

```bash
ls -lh model/
# Should see: natron_supervised_best.pt
```

### "Cannot connect to server" (MQL5)

**Solution 1**: Check server is running:
```bash
netstat -an | grep 9090
```

**Solution 2**: Check firewall:
```bash
# Linux
sudo ufw allow 9090

# Windows: Add firewall rule for port 9090
```

### "Insufficient data" error

**Solution**: You need at least 96 candles + more for training:
```bash
# Check your data
wc -l data/data_export.csv
# Should be > 5000
```

---

## 📚 Next Steps

### 1. Customize Configuration

Edit `config/config.yaml`:

```yaml
# Try different model sizes
model:
  d_model: 512  # Larger model (was 256)
  num_encoder_layers: 8  # Deeper (was 6)

# Adjust trading thresholds
supervised:
  task_weights:
    direction: 2.0  # Focus more on direction
```

### 2. Add Custom Features

Edit `src/features/feature_engine.py`:

```python
def _my_indicator(self, df):
    # Add your custom indicator
    feat = pd.DataFrame(index=df.index)
    feat['my_custom'] = your_calculation(df)
    return feat
```

### 3. Optimize Parameters

```bash
# Try different hyperparameters
python scripts/train_full_pipeline.py \
  --config config/experimental.yaml
```

### 4. Backtest & Optimize

- Run on demo account first
- Monitor for 1-2 weeks
- Analyze win rate, profit factor
- Adjust EA parameters

---

## 🎓 Learning Resources

- **Full Documentation**: See `README.md`
- **Deployment Guide**: See `DEPLOYMENT.md`
- **Architecture Details**: See code docstrings
- **Model Outputs**: Check `logs/tensorboard`

---

## 📞 Need Help?

- 🐛 **Bug Reports**: [GitHub Issues](https://github.com/yourusername/natron-transformer/issues)
- 💬 **Questions**: [GitHub Discussions](https://github.com/yourusername/natron-transformer/discussions)
- 📧 **Email**: support@natron-ai.com

---

## ⚠️ Important Reminders

1. **ALWAYS TEST IN DEMO FIRST** ⚠️
2. Start with small lot sizes (0.01)
3. Monitor closely for first week
4. Markets can be unpredictable
5. **Never risk more than you can afford to lose**

---

## ✅ Checklist

- [ ] Python 3.10+ installed
- [ ] GPU drivers + CUDA installed
- [ ] Data prepared (>5000 rows)
- [ ] Models trained (all 3 phases)
- [ ] Server running and healthy
- [ ] MQL5 EA compiled
- [ ] EA attached to demo chart
- [ ] Signals appearing correctly
- [ ] Understand risk management
- [ ] Read full documentation

---

**Congratulations! 🎉**

You're now running an AI-powered trading system. 

**Remember**: Past performance ≠ future results. Trade responsibly!

---

**Happy Trading! 📈🚀**
