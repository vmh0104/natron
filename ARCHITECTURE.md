# 🏗️ Natron Architecture Overview

## System Components

### 1. Feature Engineering (`feature_engine.py`)
- **Input**: OHLCV data (time, open, high, low, close, volume)
- **Output**: ~100 technical features
- **Groups**:
  - Moving Averages (13): MA, EMA, slopes, crossovers
  - Momentum (13): RSI, ROC, CCI, Stochastic, MACD
  - Volatility (15): ATR, Bollinger Bands, Keltner Channels
  - Volume (9): OBV, VWAP, MFI, ratios
  - Price Patterns (8): Doji, gaps, shadows, body%
  - Returns (8): Log returns, intraday, cumulative
  - Trend Strength (6): ADX, DI+, DI-, Aroon
  - Statistical (6): Skewness, Kurtosis, Z-score, Hurst
  - Support/Resistance (4): Distance to highs/lows
  - SMC (6): Swing High/Low, BOS/CHOCH
  - Market Profile (10): POC, VAH, VAL, Entropy

### 2. Label Generation (`label_generator.py`)
- **Buy Signals**: ≥2 conditions from 6 technical rules
- **Sell Signals**: ≥2 conditions from 5 technical rules
- **Direction**: Binary (up/down) based on future returns
- **Regime**: 6-class classification (BULL_STRONG, BULL_WEAK, RANGE, BEAR_WEAK, BEAR_STRONG, VOLATILE)

### 3. Sequence Creation (`sequence_creator.py`)
- **Input**: Labeled DataFrame with features
- **Output**: 
  - X: (N, 96, 100) sequences
  - y: Dictionary with buy/sell/direction/regime labels
- **Scaling**: StandardScaler for feature normalization

### 4. Model Architecture (`model.py`)

#### Transformer Encoder
- **Input Projection**: 100 features → d_model (128)
- **Positional Encoding**: Sinusoidal encoding for sequence position
- **Transformer Layers**: 6 layers, 8 heads, 512 FFN dimension
- **Output**: (batch_size, seq_len, d_model)

#### Multi-Task Heads
- **Buy Head**: Linear → ReLU → Dropout → Linear → Sigmoid
- **Sell Head**: Linear → ReLU → Dropout → Linear → Sigmoid
- **Direction Head**: Linear → ReLU → Dropout → Linear → Softmax(2)
- **Regime Head**: Linear → ReLU → Dropout → Linear → Softmax(6)

#### Pooling Strategy
- **Mean Pooling**: Average over sequence length
- Alternative: Last token or CLS token

### 5. Training Pipeline (`training.py`)

#### Phase 1: Pretraining
- **Method**: Masked Modeling
- **Process**:
  1. Randomly mask 15% of sequence tokens
  2. Encode masked sequence
  3. Reconstruct masked features
  4. MSE loss on masked positions only
- **Goal**: Learn latent market representations

#### Phase 2: Supervised Fine-Tuning
- **Loss Functions**:
  - Buy/Sell: BCE Loss
  - Direction/Regime: CrossEntropy Loss
- **Multi-Task Loss**: Weighted sum of all task losses
- **Optimizer**: AdamW (lr=1e-4, weight_decay=1e-5)
- **Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Early Stopping**: Based on validation loss

### 6. API Servers

#### Flask HTTP Server (`api_server.py`)
- **Endpoints**:
  - `GET /health`: Health check
  - `POST /predict`: Single prediction
  - `POST /predict_batch`: Batch predictions
- **Port**: 5000 (configurable)

#### Socket Server (`socket_server.py`)
- **Protocol**: TCP Socket (better for MQL5)
- **Format**: JSON over TCP
- **Port**: 8888 (configurable)
- **Threading**: Multi-client support

### 7. MetaTrader 5 Integration (`NatronEA.mq5`)

#### EA Components
- **Socket Connection**: Connects to Python server
- **Data Collection**: Collects last 96 candles
- **JSON Serialization**: Formats candles as JSON
- **Request/Response**: Sends request, receives predictions
- **Trade Execution**: Opens/closes positions based on signals

#### Trading Logic
- **Buy Signal**: buy_prob ≥ threshold AND regime allowed
- **Sell Signal**: sell_prob ≥ threshold AND regime allowed
- **Position Management**: Closes opposite position before opening new one
- **Risk Management**: Stop Loss / Take Profit

## Data Flow

```
OHLCV Data (CSV)
    ↓
FeatureEngine → 100 Features
    ↓
LabelGenerator → Buy/Sell/Direction/Regime
    ↓
SequenceCreator → (N, 96, 100) Sequences
    ↓
Train/Val Split
    ↓
Phase 1: Pretraining (Masked Modeling)
    ↓
Phase 2: Supervised Fine-Tuning
    ↓
Model Saved → model/natron_v2.pt
    ↓
API Server / Socket Server
    ↓
MQL5 EA → Live Trading
```

## Model Specifications

- **Input**: (batch_size, 96, 100)
- **Encoder**: Transformer (6 layers, 8 heads, d_model=128)
- **Outputs**:
  - Buy probability: (batch_size,)
  - Sell probability: (batch_size,)
  - Direction probabilities: (batch_size, 2)
  - Regime probabilities: (batch_size, 6)
- **Parameters**: ~2-3M (configurable)

## Training Configuration

### Default Hyperparameters
- **d_model**: 128
- **nhead**: 8
- **num_layers**: 6
- **dim_feedforward**: 512
- **dropout**: 0.1
- **batch_size**: 32
- **learning_rate**: 1e-4
- **weight_decay**: 1e-5
- **pretrain_epochs**: 10
- **supervised_epochs**: 50

### GPU Requirements
- **VRAM**: ~2-4GB (batch_size=32)
- **CUDA**: 11.8+ recommended
- **Compute Capability**: 7.0+

## Performance Considerations

### Optimization Tips
1. **Batch Size**: Adjust based on GPU memory
2. **Sequence Length**: 96 is optimal; can reduce to 64 for speed
3. **Model Size**: Reduce `d_model` or `num_layers` for faster inference
4. **Feature Count**: Currently 100; can extend or reduce

### Inference Speed
- **CPU**: ~50-100ms per prediction
- **GPU**: ~5-10ms per prediction
- **Throughput**: ~100-200 predictions/second (GPU)

## Extensibility

### Adding New Features
1. Extend `FeatureEngine` with new feature groups
2. Update `num_features` in config
3. Retrain model

### Adding New Tasks
1. Add new head in `NatronModel`
2. Add loss function in `SupervisedTrainer`
3. Update label generation if needed

### Custom Regimes
1. Modify `_generate_regime()` in `LabelGenerator`
2. Update `regime_head` output size
3. Retrain model
