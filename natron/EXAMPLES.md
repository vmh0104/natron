# Natron Examples and Usage

## Example YAML Configuration

### Training Configuration (`config_train.yaml`)
```yaml
# Model architecture
model:
  d_model: 128
  nhead: 8
  num_layers: 6
  dim_feedforward: 512
  dropout: 0.1
  max_seq_len: 96
  n_regime_classes: 6

# Optimizer
optimizer:
  type: AdamW
  lr: 0.0001
  weight_decay: 0.00001

# Loss weights
loss:
  regime_weight: 1.0
  context_weight: 0.5
  forecast_weight: 1.0

# Training
training:
  epochs: 50
  batch_size: 32
  use_mixed_precision: true
```

### Realtime Configuration (`realtime_config.yaml`)
```yaml
model:
  checkpoint_path: models/natron_v6/natron_v6.pt
  n_features: 65

trading:
  entry_threshold: 0.6
  cancel_threshold: 0.3
  atr_multiplier: 2.0
  max_positions: 1
  lot_size: 0.01
```

---

## Example Training Run Output

```
============================================================
NATRON DATA PIPELINE - Processing Started
============================================================

[1/4] Loading raw data...
Loaded 10000 candles from data/raw/EURUSD_M15.csv
[2/4] Standardizing data...
   Cleaned dataset: 9985 candles
[3/4] Computing features (50-70 features)...
   Computed 65 features
   Feature names: atr, ema_20, ema_50, ema_200, rsi, rsi_9, rsi_21, macd, macd_signal, macd_histogram...
[4/4] Labeling regimes...

=== Regime Distribution ===
BULL_STRONG    :   1245 (12.47%)
BULL_WEAK      :   1856 (18.60%)
BEAR_STRONG    :   1123 (11.25%)
BEAR_WEAK      :   1654 (16.57%)
RANGE          :   2890 (28.97%)
VOLATILE       :   1217 (12.19%)
Total          :   9985 (100.00%)

[SAVE] Saving processed data to data/processed/processed_data.csv...
   Saved 9985 rows × 71 columns

============================================================
DATA PIPELINE - Processing Complete!
============================================================

============================================================
NATRON MODEL TRAINING - Starting
============================================================

Using device: cuda
GPU: NVIDIA T4
CUDA Version: 11.8

Model architecture:
  Input features: 65
  Model dimension: 128
  Attention heads: 8
  Encoder layers: 6
  Total parameters: 2,345,678

Created 9890 sequences:
  X shape: (9890, 96, 65)
  y_regime shape: (9890,)
  y_context shape: (9890,)
  y_forecast shape: (9890,)

Dataset splits:
  Train: 6923 samples
  Val:   1483 samples
  Test:  1484 samples

Epoch 1/50: 100%|████████| 217/217 [02:15<00:00,  1.60it/s]
  Train Loss: 1.234567 (R:0.8234 C:0.1234 F:0.2879)
  Val Loss:   1.198765 (R:0.8123 C:0.1156 F:0.2709)
  LR: 1.00e-04

Epoch 2/50: 100%|████████| 217/217 [02:14<00:00,  1.61it/s]
  Train Loss: 0.987654 (R:0.7123 C:0.0987 F:0.1765)
  Val Loss:   1.012345 (R:0.7234 C:0.1023 F:0.1867)
  LR: 9.80e-05

...

Epoch 50/50: 100%|████████| 217/217 [02:13<00:00,  1.62it/s]
  Train Loss: 0.456789 (R:0.3456 C:0.0456 F:0.0657)
  Val Loss:   0.478901 (R:0.3567 C:0.0489 F:0.0732)
  LR: 1.00e-06

✓ Saved best model to models/natron_v6/natron_v6.pt

============================================================
TRAINING COMPLETE!
Best validation loss: 0.456789
============================================================
```

---

## Example Evaluation Output

```
============================================================
NATRON MODEL EVALUATION RESULTS
============================================================

[1] Regime Classification:
  Accuracy:  0.8234
  F1 (macro): 0.7891
  F1 (weighted): 0.8156
  Precision: 0.8012
  Recall:    0.7891

[2] Context Strength:
  MSE:         0.002345
  MAE:         0.0389
  R²:          0.7234
  Correlation: 0.8512

[3] Forecast Direction:
  Accuracy:  0.7123
  F1:        0.6987
  Precision: 0.7234
  Recall:    0.6789

✓ Saved metrics to evaluation_results/metrics_report.json
✓ Exported predictions to evaluation_results/natron_predictions.csv
```

---

## Example Realtime Event Log

```csv
timestamp,event_type,regime,context_strength,forecast_direction,forecast_confidence,entry_zone
2024-01-15T10:30:00,ENTRY_SIGNAL,0,0.7234,1,0.8123,"{'direction': 'BUY', 'entry_price': 1.0850, 'stop_loss': 1.0820, 'take_profit': 1.0910}"
2024-01-15T11:45:00,ENTRY_SIGNAL,2,0.6456,0,0.7891,"{'direction': 'SELL', 'entry_price': 1.0875, 'stop_loss': 1.0905, 'take_profit': 1.0795}"
2024-01-15T12:20:00,TRADE_EXECUTED,BUY,1.0850,1.0820,1.0910
2024-01-15T13:10:00,TRADE_CLOSED,BUY,1.0850,1.0895,150.0
```

---

## Example Feedback Learning Output

```
Starting feedback learning cycle...
Loaded 1500 predictions
Loaded 45 events

Adapted config: Entry threshold 0.600 -> 0.675, Cancel threshold 0.300 -> 0.350

✓ Saved adapted config to feedback_output/adaptive_config.yaml
✓ Saved learning history to feedback_output/learning_history.json
✓ Saved analysis report to feedback_output/feedback_analysis_report.json

Feedback learning cycle complete
```

---

## Step-by-Step Usage on Google Colab

### 1. Setup Environment
```python
# Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install pandas numpy scikit-learn matplotlib seaborn pyyaml tqdm

# Upload files
from google.colab import files
files.upload()  # Upload natron folder and data files
```

### 2. Phase 1: Data Pipeline
```python
import sys
sys.path.append('natron')

from data_pipeline import DataPipeline

pipeline = DataPipeline()
feature_df, labels = pipeline.process(
    'data/raw/market_data.csv',
    'data/processed/processed_data.csv'
)
```

### 3. Phase 2: Training
```python
from train_model import Trainer
from dataset_loader import create_dataloaders
import pandas as pd
import numpy as np
import torch

# Load processed data
processed_df = pd.read_csv('data/processed/processed_data.csv', index_col=0)
feature_cols = [col for col in processed_df.columns 
                if col not in ['open', 'high', 'low', 'close', 'volume', 'regime']]
feature_df = processed_df[feature_cols]
labels = processed_df['regime'].astype(int)

# Create sequences
from data_pipeline import DataPipeline
pipeline = DataPipeline()
X, y_regime, y_context, y_forecast = pipeline.create_sequences(
    feature_df, labels, sequence_length=96, forecast_horizon=5
)

# Create dataloaders
train_loader, val_loader, test_loader, norm_stats = create_dataloaders(
    X, y_regime, y_context, y_forecast, batch_size=32
)

# Train
trainer = Trainer('natron/config_train.yaml')
trainer.build_model(n_features=X.shape[-1])
trainer.build_optimizer_and_scheduler()

regime_counts = np.bincount(y_regime)
class_weights = torch.tensor(1.0 / (regime_counts + 1e-6), dtype=torch.float32)
trainer.build_criterion(class_weights=class_weights)

trainer.train(train_loader, val_loader)
```

### 4. Phase 3: Evaluation
```python
from evaluate_model import ModelEvaluator
from dataset_loader import NatronDataset, DataLoader
import pandas as pd

# Load data and create test dataset
processed_df = pd.read_csv('data/processed/processed_data.csv', index_col=0)
# ... (same sequence creation as above)

# Evaluate
evaluator = ModelEvaluator(
    'models/natron_v6/natron_v6.pt',
    'natron/config_train.yaml'
)
evaluator.load_model(n_features=X.shape[-1])
metrics = evaluator.evaluate(test_loader)
evaluator.print_metrics(metrics)
```

---

## Docker Deployment Example

```bash
# Build image
docker build -t natron:v1.0 .

# Run container
docker run -d \
    --name natron \
    --restart unless-stopped \
    -p 8888:8888 \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/logs:/app/logs \
    natron:v1.0

# View logs
docker logs -f natron

# Stop container
docker stop natron
```

---

## MQL5 EA Setup Example

1. Copy `natron_ea.mq5` to `MetaTrader 5/MQL5/Experts/`
2. Open MetaEditor and compile
3. In MT5, drag EA onto chart
4. Set parameters:
   - Lot Size: 0.01
   - Magic Number: 123456
   - Socket Port: 8888
   - Symbol: EURUSD
   - Timeframe: M15
5. Enable AutoTrading
6. Start Python server: `python natron_server_v5.py --config realtime_config.yaml`

---

## Monitoring Example

```bash
# Start monitoring
python monitor_natron.py --config monitoring_config.yaml

# Output:
# Starting Natron monitoring...
# [2024-01-15 10:00:00] System Health: CPU: 45.2%, Memory: 62.1%, Disk: 34.5%, Status: healthy
# [2024-01-15 10:00:00] Process Status: Running: True, Status: running
```

---

## Troubleshooting

### Issue: Import errors
**Solution:** Ensure you're in the natron directory or add it to PYTHONPATH:
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/natron"
```

### Issue: CUDA out of memory
**Solution:** Reduce batch size in `config_train.yaml`:
```yaml
training:
  batch_size: 16  # Reduce from 32
```

### Issue: Socket connection failed
**Solution:** Check firewall and ensure Python server is running:
```bash
netstat -an | grep 8888
```

### Issue: MQL5 EA not receiving signals
**Solution:** Verify socket port matches in both `realtime_config.yaml` and EA settings.
