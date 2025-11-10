# 🚀 Natron Quick Start Guide

## Prerequisites

```bash
# Install Python dependencies
pip install -r requirements.txt

# Ensure CUDA is available (optional but recommended)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Step 1: Generate Sample Data (Optional)

If you don't have data yet, generate sample data:

```bash
python demo.py
```

This creates `data/sample_data.csv` with 1000 candles.

## Step 2: Train the Model

### Full Training (Pretrain + Supervised)

```bash
python train.py --data data/sample_data.csv --phase both
```

This will:
1. Extract ~100 features from OHLCV data
2. Generate buy/sell/direction/regime labels
3. Create sequences of 96 candles
4. Train Phase 1 (Pretraining) - 10 epochs
5. Train Phase 2 (Supervised) - 50 epochs
6. Save model to `models/natron_v2.pt`

### Training Options

```bash
# Pretraining only
python train.py --data data/sample_data.csv --phase 1

# Supervised only
python train.py --data data/sample_data.csv --phase 2

# Resume from checkpoint
python train.py --data data/sample_data.csv --phase 2 --resume models/natron_pretrained.pt
```

## Step 3: Test Inference

```bash
python inference.py --model models/natron_v2.pt --data data/sample_data.csv
```

## Step 4: Start API Server

```bash
python api_server.py --model models/natron_v2.pt --port 5000
```

Test the API:

```bash
curl http://localhost:5000/health
```

## Step 5: Start Socket Server (for MQL5)

```bash
python socket_server.py --host localhost --port 8888 --model models/natron_v2.pt
```

## Step 6: Deploy MQL5 EA

1. Copy `mql5_ea.mq5` to MetaTrader 5 `Experts` folder
2. Open MetaEditor and compile the EA
3. In MT5, drag the EA onto a chart
4. Configure:
   - ServerHost: `localhost`
   - ServerPort: `8888`
   - LotSize: `0.01`
   - MinConfidence: `0.6`
5. Enable AutoTrading

## Configuration

Edit `config.yaml` to customize:
- Model architecture (d_model, num_layers, etc.)
- Training parameters (batch_size, learning_rate, epochs)
- API/Socket server settings

## Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` in `config.yaml`
- Reduce `d_model` or `num_layers`

### Not Enough Data
- Need at least 96 candles for one sequence
- Recommended: 1000+ candles for training

### MQL5 Connection Failed
- Ensure socket server is running
- Check firewall settings
- Verify host/port in EA settings

## File Structure

```
.
├── feature_engine.py      # Feature extraction
├── label_generator.py     # Label generation
├── sequence_creator.py    # Sequence construction
├── model.py              # Transformer architecture
├── trainer.py            # Training pipeline
├── train.py              # Main training script
├── inference.py          # Standalone inference
├── api_server.py          # Flask API
├── socket_server.py       # Socket server for MQL5
├── mql5_ea.mq5           # MetaTrader 5 EA
├── demo.py               # Generate sample data
├── config.yaml           # Configuration
└── requirements.txt      # Dependencies
```

## Next Steps

1. **Backtest**: Test on historical data before live trading
2. **Tune Parameters**: Adjust confidence thresholds, lot sizes
3. **Monitor**: Watch model performance and adjust as needed
4. **Scale**: Increase model size or add more features for better performance
