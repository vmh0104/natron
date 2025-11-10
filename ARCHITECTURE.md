# 🏗️ Natron Transformer Architecture

## System Overview

Natron is an end-to-end AI trading system that combines deep learning, multi-task learning, and reinforcement learning to make trading decisions based on market data.

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Pipeline                             │
│  OHLCV Data → Feature Engineering → Labeling → Sequences   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Training Pipeline (3 Phases)                    │
│  Phase 1: Pretraining (Unsupervised)                       │
│  Phase 2: Supervised Fine-tuning (Multi-task)               │
│  Phase 3: Reinforcement Learning (PPO)                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    Inference Layer                           │
│  Flask API Server  │  Socket Server  │  MQL5 EA            │
└─────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Data Processing Layer

#### FeatureEngine (`feature_engine.py`)
- **Input**: OHLCV DataFrame (time, open, high, low, close, volume)
- **Output**: ~100 engineered features
- **Groups**:
  - Moving Averages (13)
  - Momentum (13)
  - Volatility (15)
  - Volume (9)
  - Price Patterns (8)
  - Returns (8)
  - Trend Strength (6)
  - Statistical (6)
  - Support/Resistance (4)
  - SMC (6)
  - Market Profile (10)

#### LabelGenerator (`label_generator.py`)
- **Buy/Sell Signals**: Multi-condition rule-based labeling
- **Direction**: Future price movement prediction
- **Regime**: 6-class market regime classification

#### SequenceCreator (`dataset_loader.py`)
- Creates sequences of 96 consecutive candles
- Handles train/val/test splits
- Normalizes features using StandardScaler
- PyTorch Dataset/DataLoader integration

### 2. Model Architecture

#### NatronTransformerEncoder (`model_natron.py`)
```
Input: (batch_size, 96, 100)
  ↓
Input Projection: Linear(100 → 256)
  ↓
Positional Encoding
  ↓
Transformer Encoder (6 layers, 8 heads)
  ↓
Output: (batch_size, 96, 256)
```

#### NatronTransformer (`model_natron.py`)
```
Encoder Output
  ↓
Sequence Pooling (AdaptiveAvgPool1d or Last Token)
  ↓
┌──────────┬──────────┬──────────────┬─────────────┐
│ Buy Head │Sell Head │Direction Head│ Regime Head  │
│ Sigmoid  │ Sigmoid  │ LogSoftmax(2)│LogSoftmax(6)│
└──────────┴──────────┴──────────────┴─────────────┘
```

### 3. Training Phases

#### Phase 1: Pretraining (`train_pretrain.py`)
**Objective**: Learn latent market representations

**Methods**:
- **Masked Modeling**: Reconstruct masked feature tokens
- **Contrastive Learning**: InfoNCE loss with data augmentation

**Loss**: MaskedModelingLoss or ContrastiveLoss

#### Phase 2: Supervised Fine-tuning (`train_supervised.py`)
**Objective**: Multi-task prediction

**Tasks**:
- Buy classification (binary)
- Sell classification (binary)
- Direction prediction (up/down)
- Regime classification (6 classes)

**Loss**: MultiTaskLoss (weighted combination)

#### Phase 3: Reinforcement Learning (`train_rl.py`)
**Objective**: Optimize trading performance

**Algorithm**: PPO (Proximal Policy Optimization)

**Reward Function**:
```
R = profit - α * turnover - β * drawdown
```

**Environment**: TradingEnvironment simulates trading with positions, PnL tracking

### 4. Inference Layer

#### Flask API Server (`api_server.py`)
- REST API endpoint: `POST /predict`
- Accepts OHLCV candles, returns predictions
- Health check endpoint: `GET /health`

#### Socket Server (`socket_server.py`)
- TCP socket server for MQL5 integration
- JSON message protocol
- Threaded client handling
- Real-time inference

#### MQL5 Expert Advisor (`natron_ea.mq5`)
- Connects to Python socket server
- Collects OHLCV data from MT5
- Sends requests, receives predictions
- Executes trades based on signals

## Data Flow

### Training Flow
```
data_export.csv
  ↓
FeatureEngine → features_df (N, 100)
  ↓
LabelGenerator → labels_df (N, 4)
  ↓
SequenceCreator → sequences (N-95, 96, 100)
  ↓
DataLoader → batches
  ↓
Model Training
  ↓
Saved Model (models/natron_v2.pt)
```

### Inference Flow
```
OHLCV Candles (96)
  ↓
FeatureEngine → features (96, 100)
  ↓
Scaler → normalized features
  ↓
Model → predictions
  ↓
JSON Response
  {
    buy_prob: 0.71,
    sell_prob: 0.24,
    direction_up: 0.69,
    regime: "BULL_WEAK",
    confidence: 0.82
  }
```

## Key Design Decisions

### 1. Sequence Length (96)
- Balances context vs. computational cost
- Captures multiple timeframes (e.g., 96 M15 candles = 24 hours)

### 2. Feature Dimension (100)
- Comprehensive feature set without excessive dimensionality
- Covers technical, statistical, and market microstructure

### 3. Multi-Task Learning
- Shared encoder learns common representations
- Task-specific heads allow specialization
- Weighted loss balances task importance

### 4. Three-Phase Training
- **Pretraining**: Unsupervised learning captures market structure
- **Supervised**: Task-specific fine-tuning
- **RL**: Optimizes for actual trading performance

### 5. Dual Inference Interfaces
- **REST API**: General-purpose, easy integration
- **Socket Server**: Low-latency, MQL5-specific

## Performance Considerations

### GPU Optimization
- CUDA automatic detection
- Batch processing for parallel inference
- Mixed precision training (can be enabled)

### Memory Management
- Gradient clipping (max_norm=1.0)
- Efficient data loading (pin_memory, num_workers)
- Model checkpointing

### Latency Optimization
- Model caching in memory
- Pre-computed feature scaler
- Efficient sequence processing

## Extensibility

### Adding New Features
1. Extend `FeatureEngine._*_features()` methods
2. Update `feature_dim` in config.yaml
3. Retrain model

### Adding New Tasks
1. Add head in `NatronTransformer`
2. Update `MultiTaskLoss`
3. Modify label generation

### Custom Training
- Modify loss functions in `losses.py`
- Adjust training loops in `train_*.py`
- Customize reward functions in RL

## Security Considerations

- Input validation in API endpoints
- Error handling and logging
- Model versioning
- Configuration management

## Monitoring

- TensorBoard for training metrics
- System monitoring via `monitor_natron.py`
- API health checks
- Performance tracking

---

**Architecture Version**: 1.0  
**Last Updated**: 2024
