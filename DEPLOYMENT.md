# 🚀 Natron Deployment Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place your OHLCV data file as `data_export.csv` in the workspace root:

```csv
time,open,high,low,close,volume
2024-01-01 00:00:00,1.1000,1.1050,1.0990,1.1040,1000
2024-01-01 00:15:00,1.1040,1.1060,1.1030,1.1050,1200
...
```

### 3. Train Model

```bash
# Full pipeline (pretraining + supervised)
python natron/train_pipeline.py \
    --data data_export.csv \
    --phase both \
    --pretrain_epochs 10 \
    --supervised_epochs 50 \
    --batch_size 32 \
    --device cuda
```

### 4. Start API Server

**Option A: Flask HTTP Server (for REST API)**
```bash
python natron/api_server.py
```

**Option B: Socket Server (for MQL5)**
```bash
python natron/socket_server.py --port 8888
```

### 5. Test API

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "candles": [
      {"time": "2024-01-01 00:00:00", "open": 1.1000, "high": 1.1050, 
       "low": 1.0990, "close": 1.1040, "volume": 1000},
      ...
    ]
  }'
```

## MetaTrader 5 Integration

### Setup Steps

1. **Copy EA to MT5**:
   - Copy `NatronEA.mq5` to `MetaTrader 5/MQL5/Experts/`

2. **Compile EA**:
   - Open MetaEditor
   - Open `NatronEA.mq5`
   - Click Compile (F7)

3. **Start Python Server**:
   ```bash
   python natron/socket_server.py --port 8888
   ```

4. **Configure EA**:
   - Drag `NatronEA` to chart
   - Set `ServerIP`: `127.0.0.1` (or your server IP)
   - Set `ServerPort`: `8888`
   - Set `UseSocket`: `true`
   - Configure other parameters (lot size, SL/TP, etc.)

5. **Enable AutoTrading**:
   - Click AutoTrading button in MT5
   - EA will start making predictions and trading

## Configuration

Edit `config.yaml` to customize:

- **Model architecture**: `d_model`, `nhead`, `num_layers`
- **Training**: `batch_size`, learning rates, epochs
- **API**: Server host/port

## File Structure

```
workspace/
├── natron/                    # Main package
│   ├── feature_engine.py     # ~100 technical features
│   ├── label_generator.py    # Buy/sell/direction/regime labels
│   ├── sequence_creator.py   # 96-candle sequences
│   ├── model.py              # Transformer architecture
│   ├── training.py           # Training scripts
│   ├── train_pipeline.py     # End-to-end training
│   ├── api_server.py         # Flask HTTP API
│   └── socket_server.py      # TCP socket server (MQL5)
├── NatronEA.mq5              # MetaTrader 5 Expert Advisor
├── config.yaml               # Configuration
├── requirements.txt          # Python dependencies
└── README.md                 # Full documentation
```

## GPU Requirements

- **CUDA**: NVIDIA GPU with CUDA support
- **PyTorch**: Install CUDA-enabled PyTorch:
  ```bash
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

## Troubleshooting

### Model Not Found
- Ensure training completed successfully
- Check `model/natron_v2.pt` exists

### CUDA Out of Memory
- Reduce `batch_size` in config
- Reduce `d_model` or `num_layers`

### MQL5 Connection Failed
- Check firewall settings
- Verify Python server is running
- Check IP/port configuration

### Feature Count Mismatch
- Ensure data has required columns
- Check feature engineering completed successfully

## Production Deployment

### Docker (Recommended)

```dockerfile
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 5000 8888

CMD ["python", "natron/api_server.py"]
```

### Systemd Service

Create `/etc/systemd/system/natron.service`:

```ini
[Unit]
Description=Natron AI Trading Server
After=network.target

[Service]
Type=simple
User=trader
WorkingDirectory=/opt/natron
ExecStart=/usr/bin/python3 natron/api_server.py
Restart=always

[Install]
WantedBy=multi-user.target
```

## Performance Monitoring

- Monitor GPU usage: `nvidia-smi`
- Check API logs: `tail -f logs/natron.log`
- Monitor MT5 EA logs in MetaTrader terminal

## Security Notes

- Use HTTPS/WSS in production
- Authenticate API requests
- Secure MQL5 server connection
- Limit API access to trusted IPs
