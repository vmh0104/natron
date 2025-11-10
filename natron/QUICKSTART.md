# Natron Quick Start Guide

## Step-by-Step Execution on Google Colab / Vertex AI Studio

### Prerequisites Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create directories
mkdir -p data/raw data/processed models logs configs
```

### Phase 1: Data Pipeline

```python
# Prepare your data: data/raw/market_data.csv
# Format: time,open,high,low,close,volume

from src.data_pipeline.data_pipeline import DataPipeline

pipeline = DataPipeline()
processed_df = pipeline.process(
    input_file="data/raw/market_data.csv",
    output_file="data/processed/processed_data.csv",
    create_target=True,
    forecast_horizon=5
)

print(f"Processed {len(processed_df)} rows")
print(f"Features: {len(pipeline.get_feature_names())}")
print(f"Regime distribution:\n{processed_df['regime_name'].value_counts()}")
```

### Phase 2: Model Training

```bash
# Train the model (requires GPU)
python -m src.training.train_model configs/config_train.yaml
```

**Expected output:**
- Model saved to: `models/natron_v6.pt`
- Training history: `logs/training/training_history.json`
- Training time: ~2-4 hours on GPU (T4/A100)

### Phase 3: Evaluation

```bash
# Evaluate model
python -m src.evaluation.evaluate_model \
    models/natron_v6.pt \
    configs/config_train.yaml \
    logs/evaluation

# Generate visualizations
python -m src.evaluation.plot_regime_distribution \
    logs/evaluation/natron_predictions.csv \
    logs/evaluation
```

**Output files:**
- `logs/evaluation/metrics_report.json`
- `logs/evaluation/natron_predictions.csv`
- `logs/evaluation/regime_distribution.png`

### Phase 4: Realtime Trading (Local/MT5)

```bash
# Start Python server
python -m src.realtime.natron_server_v5 configs/realtime_config.yaml

# In MetaTrader 5:
# 1. Compile natron_ea.mq5
# 2. Attach EA to chart
# 3. Configure: ServerHost=localhost, ServerPort=8888
```

### Phase 5: Feedback Learning

```bash
# After collecting trading data, run feedback learning
python -m src.feedback.auto_feedback_engine \
    logs/evaluation/natron_predictions.csv \
    logs/realtime_events.csv \
    configs/realtime_config.yaml \
    configs/adaptive_config.yaml
```

### Phase 6: Deployment

```bash
# Docker deployment
docker build -t natron:latest .
docker run -d -p 8888:8888 natron:latest

# Monitoring
python -m src.deployment.monitor_natron localhost 8888 60
```

---

## Google Colab Specific Steps

### 1. Upload Project
```python
from google.colab import files
import zipfile

# Upload natron.zip
uploaded = files.upload()
with zipfile.ZipFile('natron.zip', 'r') as zip_ref:
    zip_ref.extractall('.')
```

### 2. Install Dependencies
```python
!pip install -r natron/requirements.txt
```

### 3. Upload Data
```python
# Upload your CSV file
from google.colab import files
uploaded = files.upload()
# Move to data/raw/market_data.csv
```

### 4. Run Phases
```python
# Phase 1
!cd natron && python -m src.data_pipeline.data_pipeline

# Phase 2 (enable GPU: Runtime > Change runtime type > GPU)
!cd natron && python -m src.training.train_model configs/config_train.yaml

# Phase 3
!cd natron && python -m src.evaluation.evaluate_model models/natron_v6.pt configs/config_train.yaml logs/evaluation
```

### 5. Download Results
```python
from google.colab import files

# Download model
files.download('natron/models/natron_v6.pt')

# Download predictions
files.download('natron/logs/evaluation/natron_predictions.csv')
```

---

## Vertex AI Studio Steps

### 1. Create Custom Training Job

```yaml
# vertex_config.yaml
trainingJobSpec:
  workerPoolSpecs:
    - machineSpec:
        machineType: n1-standard-4
        acceleratorType: NVIDIA_TESLA_T4
        acceleratorCount: 1
      replicaCount: 1
      containerSpec:
        imageUri: gcr.io/your-project/natron-trainer:latest
        args:
          - "python"
          - "-m"
          - "src.training.train_model"
          - "configs/config_train.yaml"
```

### 2. Submit Job
```bash
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name="natron-training" \
    --config=vertex_config.yaml
```

### 3. Deploy Model
```bash
gcloud ai models deploy natron-model \
    --model-dir=gs://your-bucket/models/natron_v6.pt \
    --region=us-central1 \
    --machine-type=n1-standard-4
```

---

## Troubleshooting

### Common Issues

1. **CUDA out of memory:**
   - Reduce `batch_size` in `config_train.yaml`
   - Reduce `seq_len` (e.g., 96 → 64)

2. **Data loading errors:**
   - Check CSV format: `time,open,high,low,close,volume`
   - Ensure no missing values

3. **Socket connection failed:**
   - Check firewall settings
   - Verify MQL5 EA configuration

4. **Model not converging:**
   - Increase learning rate
   - Check data quality
   - Verify feature scaling

---

## Expected Performance

- **Training time:** 2-4 hours (GPU), 10-20 hours (CPU)
- **Model size:** ~11 MB (natron_v6.pt)
- **Inference speed:** ~10-50 ms per prediction (GPU)
- **Memory usage:** ~2-4 GB (training), ~500 MB (inference)

---

## Next Steps

1. Experiment with different model architectures
2. Add more features (order flow, sentiment)
3. Implement G9-G13 enhancements (see roadmap)
4. Backtest on historical data
5. Paper trade before live trading

---

**Happy Trading! 🚀**
