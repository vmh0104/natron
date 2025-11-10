# Natron Transformer Architecture

## System Overview

The Natron Transformer is an end-to-end multi-task deep learning system for financial trading, designed to run on GPU servers (Ubuntu/Debian, GCP, Vertex AI).

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Data Pipeline                            │
├─────────────────────────────────────────────────────────────┤
│  data_export.csv → FeatureEngine → LabelGenerator          │
│                    (100 features)    (buy/sell/dir/regime) │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Sequence Creation                              │
├─────────────────────────────────────────────────────────────┤
│  SequenceCreator: 96 consecutive candles → (N, 96, 100)   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Model Architecture                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: (batch, 96, 100)                                   │
│    ↓                                                        │
│  Input Projection: Linear(100 → 256)                      │
│    ↓                                                        │
│  Positional Encoding                                        │
│    ↓                                                        │
│  Transformer Encoder (6 layers, 8 heads)                    │
│    ↓                                                        │
│  Pooling: [last_timestep || global_avg]                    │
│    ↓                                                        │
│  ┌──────────────┬──────────────┬──────────────┬──────────┐│
│  │ Buy Head     │ Sell Head    │ Direction    │ Regime   ││
│  │ (Sigmoid)    │ (Sigmoid)    │ (Softmax(2)) │ (Softmax ││
│  │              │              │              │ (6))     ││
│  └──────────────┴──────────────┴──────────────┴──────────┘│
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Training Phases                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: Pretraining                                       │
│  ├─ Masked Modeling: Reconstruct masked tokens             │
│  └─ Contrastive Learning: InfoNCE loss                     │
│                                                             │
│  Phase 2: Supervised Fine-Tuning                           │
│  └─ Multi-task weighted loss                                │
│     ├─ Buy/Sell: BCE Loss                                   │
│     ├─ Direction: CrossEntropy Loss                         │
│     └─ Regime: CrossEntropy Loss                            │
│                                                             │
│  Phase 3: Reinforcement Learning (Optional)                │
│  └─ PPO Algorithm                                           │
│     └─ Reward: profit - α×turnover - β×drawdown            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Deployment                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Flask API Server (REST)                                    │
│  ├─ /predict: Single sequence prediction                    │
│  └─ /predict_batch: Batch predictions                      │
│                                                             │
│  MQL5 Socket Server (Real-time)                            │
│  ├─ Receives candle data from MT5 EA                        │
│  ├─ Returns JSON predictions                                │
│  └─ Supports streaming candle updates                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. FeatureEngine (`feature_engine.py`)

Generates ~100 technical features from OHLCV data:

- **Moving Averages (13)**: MA5/10/20/50/100/200, EMA12/26, slopes, crossovers, ratios
- **Momentum (13)**: RSI, ROC, CCI, Stochastic, MACD, Williams %R
- **Volatility (15)**: ATR, Bollinger Bands, Keltner Channels, StdDev
- **Volume (9)**: OBV, VWAP, MFI, Volume ratios, spikes
- **Price Patterns (8)**: Doji, gaps, shadows, body%, position
- **Returns (8)**: Log returns, intraday, cumulative
- **Trend Strength (6)**: ADX, +DI, -DI, Aroon
- **Statistical (6)**: Skewness, Kurtosis, Z-score, Hurst exponent
- **Support/Resistance (4)**: Distance to highs/lows
- **SMC (6)**: Swing High/Low, BOS, CHOCH, Order Blocks
- **Market Profile (10)**: POC, VAH, VAL, Entropy, Balance

### 2. LabelGenerator (`label_generator.py`)

Creates trading labels using institutional rules:

**Buy Signal** (≥2 conditions):
- close > MA20 > MA50
- RSI > 50 or oversold recovery
- close > BB midband + MA20_slope > 0
- volume > 1.5× rolling20
- close near high (≥70%)
- MACD_hist > 0 and increasing

**Sell Signal** (≥2 conditions):
- close < MA20 < MA50
- RSI < 50 or overbought turn
- close < BB midband + MA20_slope < 0
- volume spike + close near low
- MACD_hist < 0 and decreasing

**Direction**: Forward-looking return (up/down)

**Regime** (6 classes):
- BULL_STRONG: trend > +2%, ADX > 25
- BULL_WEAK: 0 < trend ≤ 2%, ADX ≤ 25
- RANGE: lateral market
- BEAR_WEAK: -2% ≤ trend < 0, ADX ≤ 25
- BEAR_STRONG: trend < -2%, ADX > 25
- VOLATILE: ATR > 90th percentile or volume spike

### 3. Model Architecture (`model.py`)

**NatronEncoder**:
- Input projection: Linear(100 → 256)
- Positional encoding (sinusoidal)
- Transformer encoder: 6 layers, 8 heads, 1024 FFN
- Layer normalization

**NatronModel**:
- Encoder output pooling: [last_timestep || global_avg]
- Multi-task heads:
  - Buy/Sell: Sigmoid (binary classification)
  - Direction: Softmax(2) (up/down)
  - Regime: Softmax(6) (6 market regimes)

### 4. Training Phases

**Phase 1 - Pretraining**:
- **Masked Modeling**: Randomly mask 15% of tokens, reconstruct
- **Contrastive Learning**: InfoNCE loss on augmented views
- Goal: Learn latent market representations

**Phase 2 - Supervised Fine-Tuning**:
- Multi-task weighted loss
- Optional encoder freezing
- Learning rate scheduling (ReduceLROnPlateau)

**Phase 3 - Reinforcement Learning**:
- PPO algorithm
- Trading environment simulation
- Reward function: profit - α×turnover - β×drawdown

### 5. Deployment

**Flask API** (`api_server.py`):
- RESTful endpoints
- JSON input/output
- Batch prediction support

**MQL5 Socket Server** (`mql5_socket_server.py`):
- TCP socket communication
- Real-time candle streaming
- Thread-safe buffer management

**MQL5 EA** (`mql5_ea.mq5`):
- Connects to Python socket server
- Sends OHLCV data on new bars
- Executes trades based on predictions

## Data Flow

1. **Training**:
   ```
   CSV → FeatureEngine → LabelGenerator → SequenceCreator → Model
   ```

2. **Inference (API)**:
   ```
   JSON (96 candles) → FeatureEngine → Model → JSON (predictions)
   ```

3. **Inference (MQL5)**:
   ```
   MT5 EA → Socket → FeatureEngine → Model → Socket → MT5 EA
   ```

## File Structure

```
.
├── feature_engine.py          # Feature generation
├── label_generator.py          # Label creation
├── sequence_creator.py         # Sequence construction
├── model.py                   # Transformer architecture
├── train_phase1_pretrain.py   # Pretraining
├── train_phase2_supervised.py # Supervised training
├── train_phase3_rl.py         # RL training
├── train.py                   # Main pipeline
├── api_server.py              # Flask API
├── mql5_socket_server.py      # Socket server
├── mql5_ea.mq5                # MT5 Expert Advisor
├── test_pipeline.py            # Component tests
├── generate_sample_data.py    # Sample data generator
├── config.yaml                # Configuration
├── requirements.txt            # Dependencies
└── README.md                  # Documentation
```

## Key Design Decisions

1. **96-candle sequences**: Captures medium-term patterns (24 hours on M15, 4 days on H1)
2. **100 features**: Comprehensive technical analysis coverage
3. **Multi-task learning**: Jointly learn related tasks (buy/sell/direction/regime)
4. **3-phase training**: Unsupervised → Supervised → RL for robust learning
5. **Real-time integration**: Socket server for low-latency MT5 execution

## Performance Considerations

- **GPU acceleration**: All training/inference uses CUDA when available
- **Batch processing**: Efficient DataLoader usage
- **Gradient clipping**: Prevents exploding gradients
- **Normalization**: Feature normalization for stable training
- **Memory efficiency**: Sequence padding handled efficiently

## Extensibility

The system is designed for easy extension:

- **New features**: Add methods to `FeatureEngine`
- **New tasks**: Add heads to `NatronModel`
- **New training phases**: Extend `train.py` pipeline
- **New deployment**: Add servers following existing patterns
