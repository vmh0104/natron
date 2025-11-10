# Natron Examples

## Example Training Run Log

```
2024-01-15 10:00:00 - INFO - DataPipeline initialized
2024-01-15 10:00:05 - INFO - Loading raw data from data/raw_data.csv
2024-01-15 10:00:10 - INFO - Loaded 10000 candles
2024-01-15 10:00:15 - INFO - Engineering features...
2024-01-15 10:00:45 - INFO - Engineered 65 features
2024-01-15 10:00:50 - INFO - Labeling regimes...
2024-01-15 10:01:00 - INFO - Regime distribution:
BULL_STRONG    1500
BULL_WEAK      1200
BEAR_STRONG    1300
BEAR_WEAK      1100
RANGE          2800
VOLATILE       2100
2024-01-15 10:01:05 - INFO - Removed 200 rows with NaN
2024-01-15 10:01:10 - INFO - Saved processed data to processed_data.csv

2024-01-15 10:05:00 - INFO - Using device: cuda
2024-01-15 10:05:05 - INFO - Loading processed data...
2024-01-15 10:05:10 - INFO - Creating dataloaders...
2024-01-15 10:05:15 - INFO - Data split: Train=7000, Val=1500, Test=1500
2024-01-15 10:05:20 - INFO - Using 65 features
2024-01-15 10:05:25 - INFO - Created dataset with 6900 valid sequences
2024-01-15 10:05:30 - INFO - Initializing model...
2024-01-15 10:05:35 - INFO - Model parameters: 2,345,678
2024-01-15 10:05:40 - INFO - Starting training for 100 epochs...

Epoch 1/100: 100%|████████████| 216/216 [02:15<00:00,  1.60it/s, loss=2.3456, regime=1.2345, forecast=0.9876]
2024-01-15 10:07:55 - INFO - Epoch 1/100 | Train Loss: 2.3456 | Val Loss: 2.1234 | LR: 0.000100
2024-01-15 10:07:55 - INFO - Saved best model to checkpoints/natron_v6.pt

Epoch 2/100: 100%|████████████| 216/216 [02:14<00:00,  1.61it/s, loss=1.9876, regime=1.0123, forecast=0.8765]
2024-01-15 10:10:09 - INFO - Epoch 2/100 | Train Loss: 1.9876 | Val Loss: 1.8765 | LR: 0.000099

...

Epoch 50/100: 100%|████████████| 216/216 [02:12<00:00,  1.63it/s, loss=0.5432, regime=0.3210, forecast=0.1987]
2024-01-15 11:55:00 - INFO - Epoch 50/100 | Train Loss: 0.5432 | Val Loss: 0.6123 | LR: 0.000050
2024-01-15 11:55:00 - INFO - Saved best model to checkpoints/natron_v6.pt

...

Epoch 100/100: 100%|████████████| 216/216 [02:11<00:00,  1.64it/s, loss=0.3210, regime=0.1987, forecast=0.1234]
2024-01-15 13:45:00 - INFO - Epoch 100/100 | Train Loss: 0.3210 | Val Loss: 0.3456 | LR: 0.000010
2024-01-15 13:45:00 - INFO - Training completed!
2024-01-15 13:45:00 - INFO - Best validation loss: 0.3456
```

## Example YAML Configuration

### config_train.yaml
```yaml
# Natron Training Configuration

# Data paths
data_path: "processed_data.csv"
output_dir: "checkpoints"
model_name: "natron_v6.pt"

# Data split
train_ratio: 0.7
val_ratio: 0.15

# Sequence parameters
sequence_length: 96  # Number of candles per sequence
forecast_horizon: 5  # Number of candles ahead to forecast

# Model architecture
d_model: 256
nhead: 8
num_layers: 6
dim_feedforward: 1024
dropout: 0.1
num_regime_classes: 6
num_forecast_classes: 2

# Training hyperparameters
batch_size: 32
num_epochs: 100
learning_rate: 1e-4
min_lr: 1e-6
weight_decay: 1e-5

# Loss weights
regime_weight: 1.0
context_weight: 0.5
forecast_weight: 1.0

# Training settings
use_mixed_precision: true
num_workers: 4
```

### realtime_config.yaml
```yaml
# Natron Realtime Trading Configuration

# Model settings
model_path: "checkpoints/natron_v6.pt"
sequence_length: 96
input_dim: 65

# Trading parameters
symbol: "EURUSD"
entry_threshold: 0.6
cancel_threshold: 0.05
atr_multiplier: 2.0

# Socket settings
socket_host: "localhost"
socket_port: 8888

# Data feed settings
data_feed_host: "localhost"
data_feed_port: 9999

# Logging
event_log_path: "realtime_events.csv"
log_level: "INFO"

# Risk management
max_position_size: 0.1
max_daily_trades: 10
max_drawdown_percent: 5.0
```

### adaptive_config.yaml (Auto-updated)
```yaml
# Natron Adaptive Configuration
# This file is automatically updated by the feedback learning engine

confidence_scaling: 1.15  # Updated from 1.0 based on performance
entry_threshold: 0.65     # Updated from 0.6 based on win rate

regime_weights:
  BULL_STRONG: 1.1   # Increased (performing well)
  BULL_WEAK: 0.75    # Decreased (underperforming)
  BEAR_STRONG: 1.05
  BEAR_WEAK: 0.8
  RANGE: 0.25        # Decreased (avoid range markets)
  VOLATILE: 0.6      # Increased (volatile markets profitable)

learning_rate: 0.1
min_samples: 50
```

## Example Evaluation Output

```
==================================================
EVALUATION SUMMARY
==================================================
Regime Accuracy: 0.8234
Regime F1 (weighted): 0.8156
Forecast Accuracy: 0.7123
Context Correlation: 0.6789
==================================================

Per-Class Regime Metrics:
BULL_STRONG: Precision=0.85, Recall=0.82, F1=0.83, Support=450
BULL_WEAK: Precision=0.78, Recall=0.75, F1=0.76, Support=360
BEAR_STRONG: Precision=0.84, Recall=0.81, F1=0.82, Support=390
BEAR_WEAK: Precision=0.76, Recall=0.73, F1=0.74, Support=330
RANGE: Precision=0.88, Recall=0.91, F1=0.89, Support=840
VOLATILE: Precision=0.79, Recall=0.77, F1=0.78, Support=630

Forecast Metrics:
Up: Precision=0.72, Recall=0.71, F1=0.71, Support=750
Down: Precision=0.70, Recall=0.71, F1=0.70, Support=750
```

## Example Trading Event Log (realtime_events.csv)

```csv
timestamp,event_type,action,symbol,price,stop_loss,take_profit,comment,regime,context
2024-01-15T14:30:00,ENTRY,BUY,EURUSD,1.08500,1.08100,1.09300,Natron_BULL_CTX0.72,0,0.72
2024-01-15T14:45:00,EXIT,CLOSE,EURUSD,1.09250,0,0,Natron_TAKE_PROFIT,0,0.68
2024-01-15T15:00:00,ENTRY,SELL,EURUSD,1.09200,1.09600,1.08400,Natron_BEAR_CTX0.75,2,0.75
2024-01-15T15:20:00,EXIT,CLOSE,EURUSD,1.08800,0,0,Natron_TAKE_PROFIT,2,0.71
```

## Example Performance Analysis

```json
{
  "total_trades": 150,
  "winning_trades": 87,
  "losing_trades": 63,
  "win_rate": 0.58,
  "total_pnl": 1250.50,
  "avg_pnl": 8.34,
  "avg_win": 25.30,
  "avg_loss": -15.20,
  "profit_factor": 1.66,
  "regime_performance": {
    "BULL_STRONG": {
      "total_trades": 45,
      "winning_trades": 28,
      "win_rate": 0.62,
      "avg_pnl": 12.50,
      "total_pnl": 562.50
    },
    "BEAR_STRONG": {
      "total_trades": 38,
      "winning_trades": 22,
      "win_rate": 0.58,
      "avg_pnl": 10.20,
      "total_pnl": 387.60
    }
  }
}
```

## Example Step-by-Step Execution (Google Colab)

```python
# Step 1: Install dependencies
!pip install pandas numpy torch scikit-learn matplotlib pyyaml

# Step 2: Upload data
from google.colab import files
uploaded = files.upload()  # Upload raw_data.csv

# Step 3: Run data pipeline
!python natron/data_pipeline.py raw_data.csv processed_data.csv

# Step 4: Train model (with GPU)
!python natron/train_model.py natron/config_train.yaml

# Step 5: Evaluate
!python natron/evaluate_model.py natron/config_train.yaml checkpoints/natron_v6.pt

# Step 6: Visualize
!python natron/plot_regime_distribution.py
```

## Example Step-by-Step Execution (Vertex AI)

```bash
# Step 1: Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/natron:latest

# Step 2: Create training job config (vertex_config.yaml)
cat > vertex_config.yaml << EOF
workerPoolSpecs:
  machineSpec:
    machineType: n1-standard-4
    acceleratorType: NVIDIA_TESLA_T4
    acceleratorCount: 1
  replicaCount: 1
  containerSpec:
    imageUri: gcr.io/PROJECT_ID/natron:latest
    args:
      - python
      - train_model.py
      - config_train.yaml
EOF

# Step 3: Submit job
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name="natron-training" \
  --config=vertex_config.yaml

# Step 4: Monitor job
gcloud ai custom-jobs describe JOB_ID --region=us-central1
```
