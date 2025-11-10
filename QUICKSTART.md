# 🚀 Natron Quick Start Guide

## Installation (One-Time Setup)

```bash
# Install Python dependencies
pip install -r requirements.txt

# Verify CUDA (optional, for GPU training)
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

## Training Workflow

### Step 1: Prepare Data

Place your OHLCV data in `data_export.csv`:
```csv
time,open,high,low,close,volume
2024-01-01 00:00:00,1.1000,1.1050,1.0990,1.1030,1000
...
```

### Step 2: Phase 1 - Pretraining

```bash
python train_pretrain.py --data data_export.csv --epochs 50
```

**Output**: `models/natron_pretrain_best.pt`

### Step 3: Phase 2 - Supervised Fine-Tuning

```bash
python train_natron.py \
    --data data_export.csv \
    --pretrain_model ./models/natron_pretrain_best.pt \
    --epochs 100
```

**Output**: `models/natron_v2.pt`

### Step 4: Phase 3 - Reinforcement Learning (Optional)

```bash
python train_rl.py \
    --data data_export.csv \
    --model ./models/natron_v2.pt \
    --episodes 100
```

## Deployment

### Option A: REST API Server

```bash
python api_server.py --model ./models/natron_v2.pt --port 5000
```

Test:
```bash
curl http://localhost:5000/health
```

### Option B: Socket Server (MQL5)

```bash
python socket_server.py --host localhost --port 8888 --model ./models/natron_v2.pt
```

### Option C: Automated Startup (All Services)

```bash
./start_natron.sh
```

Stops with:
```bash
./stop_natron.sh
```

## MQL5 Integration

1. Copy `natron_ea.mq5` to `MetaTrader 5/MQL5/Experts/`
2. Compile in MetaEditor
3. Attach to chart:
   - Server Host: `localhost`
   - Server Port: `8888`
   - Enable Trading: `true`

## Testing

Run system test:
```bash
python test_natron.py
```

Monitor system:
```bash
python monitor_natron.py --once
```

## File Structure

```
/workspace/
├── Core Modules
│   ├── feature_engine.py      # ~100 technical features
│   ├── label_generator.py     # Buy/sell/regime labels
│   ├── dataset_loader.py      # PyTorch datasets
│   ├── model_natron.py        # Transformer model
│   └── losses.py              # Loss functions
│
├── Training Scripts
│   ├── train_pretrain.py      # Phase 1: Pretraining
│   ├── train_natron.py        # Phase 2: Supervised
│   └── train_rl.py            # Phase 3: RL (optional)
│
├── Deployment
│   ├── api_server.py          # Flask REST API
│   ├── socket_server.py       # TCP socket (MQL5)
│   └── natron_ea.mq5          # MetaTrader 5 EA
│
├── Utilities
│   ├── monitor_natron.py      # System monitoring
│   ├── test_natron.py         # System test
│   ├── start_natron.sh        # Startup script
│   └── stop_natron.sh         # Stop script
│
└── Configuration
    ├── config.yaml             # System config
    └── requirements.txt        # Dependencies
```

## Common Commands

```bash
# Test system
python test_natron.py

# Train model
python train_natron.py --data data_export.csv

# Start API
python api_server.py

# Start socket server
python socket_server.py

# Monitor
python monitor_natron.py

# Full system
./start_natron.sh
```

## Troubleshooting

**Import errors**: Install dependencies: `pip install -r requirements.txt`

**CUDA errors**: Use CPU: Add `--device cpu` to training scripts

**Model not found**: Train first: `python train_natron.py --data data_export.csv`

**Socket connection failed**: Check firewall, verify server is running

**MQL5 not connecting**: Verify Python server on `localhost:8888`

## Next Steps

1. ✅ Install dependencies
2. ✅ Prepare `data_export.csv`
3. ✅ Train model (Phase 1 → Phase 2)
4. ✅ Deploy (API or Socket server)
5. ✅ Connect MQL5 EA (optional)

For detailed documentation, see `README.md`.
