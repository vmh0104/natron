# Natron Quick Start Guide

## 🚀 5-Minute Setup

### Step 1: Install Dependencies

```bash
cd natron
pip install -r requirements.txt
```

### Step 2: Generate Sample Data (for testing)

```bash
python scripts/generate_sample_data.py --num-candles 10000 --output data/raw_data.csv
```

### Step 3: Run Data Pipeline

```bash
python -m src.data_pipeline.data_pipeline \
    --input data/raw_data.csv \
    --output data/processed_data.csv \
    --config configs/config_train.yaml
```

### Step 4: Train Model

```bash
python -m src.training.train_model --config configs/config_train.yaml
```

Training will:
- Load processed data
- Create train/val/test splits
- Train transformer model for ~100 epochs
- Save best model to `models/natron_v6.pt`

### Step 5: Evaluate Model

```bash
python -m src.evaluation.evaluate_model \
    --model models/natron_v6.pt \
    --data data/processed_data.csv \
    --output logs/natron_predictions.csv \
    --report logs/metrics_report.json \
    --split test
```

### Step 6: Visualize Results

```bash
python -m src.evaluation.plot_regime_distribution \
    --predictions logs/natron_predictions.csv \
    --output logs
```

---

## 📊 Expected Outputs

After running all phases, you should have:

1. **Processed Data**: `data/processed_data.csv` (with 50-70 features)
2. **Trained Model**: `models/natron_v6.pt`
3. **Predictions**: `logs/natron_predictions.csv`
4. **Metrics Report**: `logs/metrics_report.json`
5. **Visualizations**: `logs/regime_distribution.png`

---

## 🔧 Configuration Tips

### For GPU Training (Colab/Vertex AI)

Edit `configs/config_train.yaml`:
```yaml
hardware:
  device: "cuda"
  mixed_precision: true
```

### For CPU Training

```yaml
hardware:
  device: "cpu"
  mixed_precision: false
```

Reduce batch size if memory issues:
```yaml
data:
  batch_size: 16  # or 8
```

---

## 🐛 Troubleshooting

**Issue**: `ModuleNotFoundError`
**Solution**: Ensure you're in the natron directory and PYTHONPATH is set:
```bash
export PYTHONPATH=$PWD:$PYTHONPATH
```

**Issue**: CUDA out of memory
**Solution**: Reduce batch_size in config or use CPU:
```yaml
data:
  batch_size: 8
hardware:
  device: "cpu"
```

**Issue**: Missing features in processed data
**Solution**: Check that input CSV has required columns: `time,open,high,low,close,volume`

---

## 📚 Next Steps

1. **Realtime Trading**: Set up MT5 and configure `configs/realtime_config.yaml`
2. **Feedback Learning**: Run adaptation after collecting trade data
3. **Monitoring**: Set up Telegram alerts for production deployment

See `README_Natron_EndToEnd.md` for full documentation.
