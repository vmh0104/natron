# Natron Example Outputs

This document shows example outputs from each phase of the Natron system.

---

## Phase 1: Data Pipeline Output

### Processed Data CSV (`data/processed_data.csv`)

Sample columns:
```
time,open,high,low,close,volume,ATR,EMA_20,EMA_50,EMA_200,RSI,MACD,MACD_signal,MACD_hist,BB_upper,BB_middle,BB_lower,BB_width,body_pct,wick_ratio,volatility_regime,regime,regime_name,...
```

### Console Output:
```
2024-01-15 10:00:00 - INFO - ============================================================
2024-01-15 10:00:00 - INFO - NATRON DATA PIPELINE - Starting
2024-01-15 10:00:00 - INFO - ============================================================
2024-01-15 10:00:01 - INFO - Loading data from data/raw_data.csv
2024-01-15 10:00:02 - INFO - Loaded 100000 candles
2024-01-15 10:00:02 - INFO - Starting feature engineering...
2024-01-15 10:00:15 - INFO - Engineered 68 features
2024-01-15 10:00:15 - INFO - Labeling regimes...
2024-01-15 10:00:18 - INFO - Regime distribution:
2024-01-15 10:00:18 - INFO -   BULL_STRONG: 15234 (15.23%)
2024-01-15 10:00:18 - INFO -   BULL_WEAK: 18456 (18.46%)
2024-01-15 10:00:18 - INFO -   BEAR_STRONG: 14234 (14.23%)
2024-01-15 10:00:18 - INFO -   BEAR_WEAK: 17543 (17.54%)
2024-01-15 10:00:18 - INFO -   RANGE: 22345 (22.35%)
2024-01-15 10:00:18 - INFO -   VOLATILE: 12188 (12.19%)
2024-01-15 10:00:19 - INFO - Saved 100000 rows to data/processed_data.csv
```

---

## Phase 2: Training Output

### Training History JSON (`logs/training_history.json`)

```json
[
  {
    "epoch": 1,
    "train": {
      "total_loss": 1.2345,
      "regime_loss": 0.4523,
      "context_loss": 0.1234,
      "forecast_loss": 0.6588
    },
    "val": {
      "total_loss": 1.1892,
      "regime_loss": 0.4389,
      "context_loss": 0.1198,
      "forecast_loss": 0.6305
    },
    "lr": 0.0001
  },
  ...
]
```

### Console Output:
```
2024-01-15 10:00:20 - INFO - ============================================================
2024-01-15 10:00:20 - INFO - NATRON MODEL TRAINING - Starting
2024-01-15 10:00:20 - INFO - ============================================================
2024-01-15 10:00:21 - INFO - Using device: cuda
2024-01-15 10:00:21 - INFO - Loading processed data from data/processed_data.csv
2024-01-15 10:00:23 - INFO - Train: 70000, Val: 15000, Test: 15000
2024-01-15 10:00:25 - INFO - Input dimension: 68
2024-01-15 10:00:26 - INFO - Model parameters: 2,456,789

Epoch 1/100: 100%|████████| 2188/2188 [02:15<00:00, 16.15it/s, loss=1.234]
Validating: 100%|████████| 469/469 [00:28<00:00, 16.45it/s]
2024-01-15 10:02:49 - INFO - Train Loss: 1.2345 | Val Loss: 1.1892
2024-01-15 10:02:49 - INFO -   Regime: 0.4523 / 0.4389
2024-01-15 10:02:49 - INFO -   Context: 0.1234 / 0.1198
2024-01-15 10:02:49 - INFO -   Forecast: 0.6588 / 0.6305
2024-01-15 10:02:49 - INFO - Saved best model to models/natron_v6.pt

Epoch 2/100: 100%|████████| 2188/2188 [02:14<00:00, 16.20it/s, loss=1.189]
...
```

---

## Phase 3: Evaluation Output

### Predictions CSV (`logs/natron_predictions.csv`)

Sample rows:
```csv
regime_pred,regime_name,regime_true,regime_true_name,regime_confidence,forecast_pred,forecast_name,forecast_true,forecast_confidence,context_score,context_true,prob_BULL_STRONG,prob_BULL_WEAK,prob_BEAR_STRONG,prob_BEAR_WEAK,prob_RANGE,prob_VOLATILE
0,BULL_STRONG,0,BULL_STRONG,0.8523,1,UP,1,0.7234,0.6543,0.6123,0.8523,0.0821,0.0234,0.0123,0.0234,0.0065
1,BULL_WEAK,1,BULL_WEAK,0.7123,0,DOWN,0,0.6892,0.5234,0.5123,0.1234,0.7123,0.0456,0.0234,0.0823,0.0130
...
```

### Metrics Report JSON (`logs/metrics_report.json`)

```json
{
  "regime": {
    "accuracy": 0.7234,
    "f1_macro": 0.7123,
    "f1_weighted": 0.7289,
    "precision_macro": 0.7156,
    "recall_macro": 0.7098,
    "per_class": {
      "BULL_STRONG": {
        "precision": 0.8234,
        "recall": 0.7892,
        "f1-score": 0.8059,
        "support": 15234
      },
      ...
    }
  },
  "forecast": {
    "accuracy": 0.6456,
    "f1": 0.6234,
    "precision": 0.6345,
    "recall": 0.6123
  },
  "context": {
    "mse": 0.0234,
    "mae": 0.1456,
    "correlation": 0.7234
  }
}
```

### Console Output:
```
2024-01-15 12:00:00 - INFO - Evaluating on test set...
Evaluating test: 100%|████████| 469/469 [00:28<00:00, 16.45it/s]
2024-01-15 12:00:28 - INFO - Exported 15000 predictions to logs/natron_predictions.csv

============================================================
EVALUATION SUMMARY
============================================================

Regime Classification:
  Accuracy: 0.7234
  F1 (macro): 0.7123
  F1 (weighted): 0.7289

Forecast Direction:
  Accuracy: 0.6456
  F1: 0.6234
  Precision: 0.6345
  Recall: 0.6123

Context Strength:
  MSE: 0.0234
  MAE: 0.1456
  Correlation: 0.7234
============================================================
```

---

## Phase 4: Realtime Server Output

### Events Log CSV (`logs/realtime_events.csv`)

Sample rows:
```csv
timestamp,type,trade_type,lot_size,price,stop_loss,take_profit,regime_name,forecast_name
2024-01-15T14:30:15.123456,prediction,,,,,BULL_STRONG,UP
2024-01-15T14:30:16.234567,trade_executed,BUY,0.10,1.10234,1.09876,1.10890,BULL_STRONG,UP
2024-01-15T14:35:20.345678,prediction,,,,,BEAR_WEAK,DOWN
...
```

### Console Output:
```
2024-01-15 14:00:00 - INFO - ============================================================
2024-01-15 14:00:00 - INFO - NATRON SERVER V5 - Starting
2024-01-15 14:00:00 - INFO - ============================================================
2024-01-15 14:00:01 - INFO - Connected to MetaTrader 5
2024-01-15 14:00:01 - INFO - Loaded model from models/natron_v6.pt
2024-01-15 14:00:02 - INFO - Started listening for realtime data
2024-01-15 14:00:02 - INFO - Server running. Press Ctrl+C to stop.
2024-01-15 14:30:15 - INFO - Trade executed: BUY 0.10 lots at 1.10234
2024-01-15 14:45:22 - INFO - Trade executed: SELL 0.10 lots at 1.09876
...
```

---

## Phase 5: Feedback Learning Output

### Learning Curve JSON (`logs/learning_curve.json`)

```json
[
  {
    "timestamp": "2024-01-15T16:00:00",
    "performance": {
      "trade_performance": {
        "total_trades": 45,
        "win_rate": 0.6222,
        "forecast_accuracy": 0.6456
      },
      "regime_performance": {
        "BULL_STRONG": {
          "count": 12,
          "regime_accuracy": 0.8333,
          "forecast_accuracy": 0.6667
        },
        ...
      }
    },
    "updates": {
      "entry_confidence_threshold": 0.63,
      "loss_weights": {
        "regime": 1.05,
        "context": 0.48,
        "forecast": 1.55
      }
    }
  },
  ...
]
```

### Console Output:
```
2024-01-15 16:00:00 - INFO - Running adaptation cycle...
2024-01-15 16:00:01 - INFO - Loaded 45 predictions
2024-01-15 16:00:01 - INFO - Loaded 23 events
2024-01-15 16:00:05 - INFO - Adaptation cycle complete
2024-01-15 16:00:05 - INFO -   Entry threshold: 0.63
2024-01-15 16:00:05 - INFO -   Loss weights: {'regime': 1.05, 'context': 0.48, 'forecast': 1.55}
2024-01-15 16:00:05 - INFO - Updated configuration saved to configs/realtime_config_adapted.yaml
```

---

## Phase 6: Monitoring Output

### Daily Summary (Telegram/Console)

```
📊 Natron Daily Summary - 2024-01-15
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trades Executed: 12
Predictions Made: 96
Errors: 0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Health Check Output

```json
{
  "status": "healthy",
  "last_event": "2024-01-15T16:45:30.123456",
  "recent_errors": 0,
  "recent_trades": 3
}
```

---

## Visualization Outputs

### Regime Distribution Plot (`logs/regime_distribution.png`)

- Pie charts showing predicted vs true regime distribution
- Time series showing regime changes over time
- Confusion matrix for regime classification

### Forecast Accuracy Plot (`logs/forecast_accuracy_over_time.png`)

- Line chart showing rolling forecast accuracy
- Reference line at 50% (random baseline)

### Context Score Analysis (`logs/context_score_analysis.png`)

- Histogram of context score distribution
- Scatter plot: predicted vs true context scores

---

## Model Checkpoint Structure

### `models/natron_v6.pt` (PyTorch checkpoint)

```python
{
    'model_state_dict': {...},  # Model weights
    'config': {...},             # Training configuration
    'epoch': 45,                 # Best epoch
    'val_loss': 0.8234,          # Best validation loss
    'input_dim': 68              # Input feature dimension
}
```

---

## Summary Statistics

After a complete run:

- **Data Processed**: 100,000 candles
- **Features Generated**: 68 features
- **Model Parameters**: ~2.5M parameters
- **Training Time**: ~4 hours (on GPU)
- **Regime Accuracy**: ~72%
- **Forecast Accuracy**: ~65%
- **Context Correlation**: ~72%

---

These outputs demonstrate the complete end-to-end workflow of the Natron AI Trading System.
