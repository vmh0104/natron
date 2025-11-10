# 🚀 Natron Quick Start Guide

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Generate Sample Data (or use your own)

```bash
python generate_sample_data.py --n 2000 --timeframe 15 --output data_export.csv
```

Or use your own `data_export.csv` with columns: `time`, `open`, `high`, `low`, `close`, `volume`

## Step 3: Test Components

```bash
python test_natron.py
```

This verifies:
- Feature generation (~100 features)
- Label generation (buy/sell/direction/regime)
- Sequence creation
- Model forward pass

## Step 4: Train Model

```bash
python train_natron.py --data data_export.csv --config config.yaml
```

Training includes:
- **Phase 1**: Unsupervised pretraining (10 epochs by default)
- **Phase 2**: Supervised fine-tuning (50 epochs by default)

Outputs:
- `model/natron_v2.pt` - Trained model
- `model/scaler.pkl` - Feature scaler

## Step 5: Start Inference Server

```bash
python server_natron.py --model model/natron_v2.pt
```

Or use the startup script:
```bash
./start_natron.sh
```

Server provides:
- **Flask API**: `http://localhost:5000/predict`
- **Socket API**: `localhost:8888` (for MQL5)

## Step 6: Test Server

```bash
python monitor_natron.py
```

## Step 7: Deploy MQL5 EA

1. Copy `natron_ea.mq5` to MetaTrader 5 `Experts` folder
2. Configure EA:
   - `ServerHost`: `localhost` (or server IP)
   - `ServerPort`: `8888`
   - `BuyThreshold`: `0.65`
   - `SellThreshold`: `0.65`
3. Attach to chart

## API Examples

### Python Client

```python
import requests

candles = [
    {"time": 1234567890, "open": 1.1000, "high": 1.1050, 
     "low": 1.0990, "close": 1.1030, "volume": 1000},
    # ... 95 more candles
]

response = requests.post(
    'http://localhost:5000/predict',
    json={'candles': candles}
)

print(response.json())
# {
#   "buy_prob": 0.71,
#   "sell_prob": 0.24,
#   "direction_up": 0.69,
#   "regime": "BULL_WEAK",
#   "confidence": 0.82
# }
```

### cURL

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d @sample_request.json
```

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in `config.yaml`
- Reduce `sequence_length` or `d_model`

### Model Not Found
- Ensure training completed successfully
- Check `model/natron_v2.pt` exists

### Connection Issues (MQL5)
- Verify server is running: `python server_natron.py`
- Check firewall settings
- Test with `monitor_natron.py`

### Low Accuracy
- Increase training epochs
- Check data quality
- Adjust loss weights in `config.yaml`

## Next Steps

- **Backtesting**: Test on historical data
- **Hyperparameter Tuning**: Adjust model architecture in `config.yaml`
- **Reinforcement Learning**: Implement Phase 3 (PPO/SAC)
- **Production Deployment**: Use gunicorn/nginx for Flask API

---

For detailed documentation, see `README.md`
