# 🚀 Natron Transformer - Quick Start Guide

Get up and running with Natron in 5 minutes!

---

## ⚡ Fast Track

### Step 1: Generate Sample Data (1 min)

```bash
python generate_sample_data.py --rows 5000 --output data/data_export.csv --with-events
```

### Step 2: Train the Model (30-60 min on GPU)

```bash
python train_natron.py
```

This will:
- Generate 100+ technical features
- Phase 1: Pretrain the encoder (unsupervised)
- Phase 2: Fine-tune for multi-task prediction
- Save model to `models/natron_v2.pt`

### Step 3: Test Inference (30 sec)

```bash
python inference.py --csv data/data_export.csv
```

You'll see:
```
🎯 NATRON PREDICTION
Buy Probability:  71.2%
Sell Probability: 24.3%
Direction UP:     69.4%
Regime: BULL_WEAK
```

### Step 4: Start API Server (instant)

```bash
./start_natron.sh
```

Or:
```bash
python server_natron.py
```

Server runs on:
- HTTP API: `http://localhost:8888`
- Socket: `tcp://localhost:9999`

### Step 5: Connect MT5 (5 min)

1. Copy `natron_ea.mq5` to MT5 Experts folder
2. Compile in MetaEditor (F7)
3. Attach to chart
4. Set parameters:
   - ServerHost: `127.0.0.1`
   - ServerPort: `9999`
5. Enable AutoTrading

✅ **Done!** Natron is now trading automatically.

---

## 🧪 Quick Test (No Training Required)

Want to test the system before training? Use the mock test:

```bash
# Generate sample data
python generate_sample_data.py --rows 200 --output data/test.csv

# Test feature generation
python -c "
from src.feature_engine import FeatureEngine
import pandas as pd

df = pd.read_csv('data/test.csv')
engine = FeatureEngine()
features = engine.generate_all_features(df)
print(f'Features generated: {features.shape[1] - 6} indicators')
"
```

---

## 📊 Architecture Overview

```
OHLCV Data (96 candles)
    ↓
Feature Engineering (100+ indicators)
    ↓
Transformer Encoder (6 layers, 8 heads, 256d)
    ↓
Multi-Task Heads
    ├── Buy Signal    (sigmoid)
    ├── Sell Signal   (sigmoid)
    ├── Direction     (softmax 2)
    └── Regime        (softmax 6)
```

---

## 🎯 Training Phases

### Phase 1: Pretraining (Unsupervised)
- **Goal**: Learn market structure
- **Method**: Masked reconstruction + Contrastive learning
- **Duration**: ~50 epochs (1-2 hours on GPU)
- **Output**: `models/pretrain/best_pretrain.pt`

### Phase 2: Supervised Fine-Tuning
- **Goal**: Multi-task prediction
- **Method**: Weighted multi-task loss
- **Duration**: ~100 epochs (2-3 hours on GPU)
- **Output**: `models/natron_v2.pt`

---

## 🔧 Configuration

Key settings in `config.yaml`:

```yaml
# Model size
model:
  d_model: 256          # Increase for more capacity
  num_encoder_layers: 6 # Increase for deeper model

# Training
supervised:
  batch_size: 32        # Reduce if OOM
  learning_rate: 0.00005

# Trading thresholds (in MQL5 EA)
buy_threshold: 0.65     # Higher = fewer, more confident trades
sell_threshold: 0.65
```

---

## 🐛 Common Issues

### Issue: "CUDA out of memory"
```yaml
# Solution: Reduce batch size in config.yaml
supervised:
  batch_size: 16  # or 8
```

### Issue: "Not enough data for sequences"
```bash
# Solution: Generate more data
python generate_sample_data.py --rows 10000
```

### Issue: "Server connection failed" (MT5)
```bash
# Solution 1: Check server is running
curl http://localhost:8888/health

# Solution 2: Check firewall
sudo ufw allow 8888
sudo ufw allow 9999

# Solution 3: Test socket directly
python -c "import socket; s = socket.socket(); s.connect(('localhost', 9999)); print('OK')"
```

---

## 📈 Expected Performance

On validation set:
- **Buy/Sell Accuracy**: 60-70%
- **Direction Accuracy**: 55-65%
- **Regime Classification**: 65-75%

**Note**: Financial markets are noisy. The model provides probabilistic signals, not guarantees.

---

## 🎓 Learning Path

### Beginner
1. Run with sample data
2. Test inference
3. Understand outputs

### Intermediate
1. Use real market data
2. Adjust thresholds
3. Backtest strategies

### Advanced
1. Add custom features
2. Tune hyperparameters
3. Implement custom loss functions
4. Add reinforcement learning

---

## 📚 Key Files

| File | Purpose |
|------|---------|
| `train_natron.py` | Main training script |
| `server_natron.py` | API server |
| `inference.py` | Standalone predictions |
| `natron_ea.mq5` | MT5 Expert Advisor |
| `config.yaml` | Configuration |
| `src/model_natron.py` | Model architecture |
| `src/feature_engine.py` | Feature generation |

---

## 🔥 Pro Tips

1. **Always pretrain first** - Improves convergence
2. **Use more data** - 5000+ candles recommended
3. **Monitor training** - Check logs regularly
4. **Start small** - Use demo account first
5. **Risk management** - Always use stop losses

---

## 🆘 Getting Help

1. Check `README.md` for detailed docs
2. Review error messages carefully
3. Test components individually
4. Check server logs: `tail -f logs/natron.log`
5. Monitor predictions: `python monitor_natron.py --predictions`

---

## ✅ Checklist

Before going live:

- [ ] Trained on sufficient data (5000+ candles)
- [ ] Tested inference accuracy
- [ ] API server running stable
- [ ] MT5 EA connected successfully
- [ ] Tested on demo account
- [ ] Stop loss configured
- [ ] Position size appropriate
- [ ] Monitoring in place

---

## 🎉 Next Steps

Once everything works:

1. **Backtest**: Test on historical data
2. **Paper trade**: Run on demo account for 1-2 weeks
3. **Optimize**: Tune thresholds based on performance
4. **Scale**: Gradually increase position sizes
5. **Monitor**: Set up alerts and dashboards

---

**Ready to build?** Start with:

```bash
python generate_sample_data.py --rows 5000 --with-events
python train_natron.py
```

Good luck! 🚀🧠📈
