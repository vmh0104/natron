# 🚀 Natron Quick Start Guide

## 1. Installation (5 minutes)

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## 2. Prepare Data

Place your OHLCV data as `data_export.csv` with columns:
- `time`: Timestamp
- `open`: Open price
- `high`: High price
- `low`: Low price
- `close`: Close price
- `volume`: Volume

Validate data:
```bash
python validate_data.py
```

## 3. Train Model (30 minutes - several hours depending on data size)

### Option A: Train All Phases
```bash
python train_all.py
```

### Option B: Train Individually
```bash
# Phase 1: Pretraining
python train_pretrain.py

# Phase 2: Supervised Fine-tuning
python train_supervised.py

# Phase 3: RL (optional)
python train_rl.py
```

## 4. Run Inference

### Flask API Server
```bash
python api_server.py
```

Test with:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"candles": [...]}'
```

### Socket Server (for MQL5)
```bash
python socket_server.py
```

## 5. MetaTrader 5 Integration

1. Copy `natron_ea.mq5` to `MetaTrader 5/MQL5/Experts/`
2. Compile in MetaEditor
3. Attach to chart
4. Configure parameters (server host, port, lot size, etc.)

## 6. Monitor System

```bash
python monitor_natron.py
```

## Common Commands

```bash
# Start API server
bash start_natron.sh api

# Start socket server
bash start_natron.sh socket

# Train all phases
bash start_natron.sh train-all

# See examples
python example_usage.py
```

## Troubleshooting

**CUDA not available?**
- Model will automatically use CPU (slower but works)

**Out of memory?**
- Reduce `batch_size` in `config.yaml`
- Reduce `sequence_length` in `config.yaml`

**Model not found?**
- Ensure training completed successfully
- Check `./models/` directory

**Socket connection failed?**
- Verify socket server is running: `python socket_server.py`
- Check firewall settings
- Ensure MQL5 EA uses correct host/port

## Next Steps

- Read `README.md` for detailed documentation
- Customize `config.yaml` for your needs
- Experiment with different feature sets
- Tune hyperparameters for your data
