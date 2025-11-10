# 🧠 Natron Transformer - Complete System Overview

## 📦 Complete File Inventory

### Core Components

1. **config.py** - Central configuration
   - Model architecture parameters
   - Training hyperparameters
   - API/Socket server settings
   - Paths and directories

2. **feature_engine.py** - Feature Generation (~100 features)
   - Moving Averages (13): MA, EMA, slopes, crossovers
   - Momentum (13): RSI, ROC, CCI, Stochastic, MACD
   - Volatility (15): ATR, Bollinger Bands, Keltner, StdDev
   - Volume (9): OBV, VWAP, MFI, volume ratios
   - Price Patterns (8): Doji, gaps, shadows, body%
   - Returns (8): Log return, intraday, cumulative
   - Trend Strength (6): ADX, +DI, -DI, Aroon
   - Statistical (6): Skewness, Kurtosis, Z-score, Hurst
   - Support/Resistance (4): Distance to High/Low
   - SMC (6): Swing High/Low, BOS/CHOCH
   - Market Profile (10): POC, VAH, VAL, Entropy

3. **label_generator.py** - Label Creation
   - Buy signals (≥2 conditions from 6 rules)
   - Sell signals (≥2 conditions from 5 rules)
   - Direction labels (up/down based on future return)
   - Regime classification (6 classes)

4. **sequence_creator.py** - Sequence Construction
   - Builds sequences of 96 consecutive candles
   - Creates (N, 96, 100) feature arrays
   - Aligns labels with sequence endpoints

5. **model.py** - Transformer Architecture
   - `NatronTransformer`: Main model class
   - `TransformerEncoder`: Multi-layer encoder
   - `PositionalEncoding`: Position embeddings
   - `PretrainingLoss`: Masked modeling loss
   - `MultiTaskLoss`: Supervised multi-task loss
   - Task heads: Buy, Sell, Direction, Regime

6. **train.py** - Training Pipeline
   - Phase 1: Pretraining (masked modeling)
   - Phase 2: Supervised fine-tuning
   - Train/validation split
   - Model checkpointing
   - Progress tracking

### Deployment Components

7. **api_server.py** - Flask REST API
   - `/health`: Health check
   - `/predict`: Single prediction endpoint
   - `/predict_batch`: Batch predictions
   - JSON request/response format

8. **mql5_socket_server.py** - Socket Server
   - TCP socket server for MQL5 communication
   - Multi-threaded client handling
   - Real-time inference
   - Error handling and reconnection

9. **NatronEA.mq5** - MetaTrader 5 Expert Advisor
   - Connects to Python socket server
   - Collects OHLCV data
   - Sends requests and receives signals
   - Executes trades based on predictions

### Utility Scripts

10. **create_sample_data.py** - Sample Data Generator
    - Creates synthetic OHLCV data for testing
    - Configurable number of candles
    - Random walk price generation

11. **test_inference.py** - Inference Test
    - Loads trained model
    - Tests prediction pipeline
    - Displays results

12. **rl_trainer.py** - RL Components (Optional)
    - `TradingEnvironment`: RL environment
    - `PPOTrainer`: PPO algorithm (placeholder)
    - Phase 3 RL training framework

### Documentation

13. **README.md** - Complete documentation
14. **QUICKSTART.md** - Quick start guide
15. **SYSTEM_OVERVIEW.md** - This file

### Configuration Files

16. **requirements.txt** - Python dependencies
17. **.gitignore** - Git ignore rules
18. **run_training.sh** - Training script

## 🔄 Data Flow

```
OHLCV Data (CSV)
    ↓
FeatureEngine → ~100 Technical Features
    ↓
LabelGenerator → Buy/Sell/Direction/Regime Labels
    ↓
SequenceCreator → (N, 96, 100) Sequences
    ↓
Transformer Model → Predictions
    ↓
API/Socket Server → Trading Signals
    ↓
MQL5 EA → Order Execution
```

## 🏗️ Model Architecture

```
Input: (batch_size, 96, 100)
    ↓
Input Projection: Linear(100 → 256)
    ↓
Positional Encoding
    ↓
Transformer Encoder (6 layers, 8 heads)
    ↓
Pooling (last token or mean)
    ↓
Task Heads:
    ├─ Buy Head: Sigmoid → (batch_size,)
    ├─ Sell Head: Sigmoid → (batch_size,)
    ├─ Direction Head: Linear → Softmax(2) → (batch_size, 2)
    └─ Regime Head: Linear → Softmax(6) → (batch_size, 6)
```

## 🎯 Training Phases

### Phase 1: Pretraining
- **Objective**: Learn market structure
- **Method**: Masked modeling (15% mask ratio)
- **Loss**: MSE reconstruction loss
- **Epochs**: 50 (configurable)

### Phase 2: Supervised Fine-tuning
- **Objective**: Multi-task prediction
- **Tasks**: Buy, Sell, Direction, Regime
- **Loss**: Weighted multi-task loss
- **Epochs**: 100 (configurable)
- **Optimizer**: AdamW (lr=1e-4)
- **Scheduler**: ReduceLROnPlateau

### Phase 3: Reinforcement Learning (Optional)
- **Objective**: Maximize trading performance
- **Algorithm**: PPO/SAC (placeholder)
- **Reward**: Profit - α×turnover - β×drawdown
- **Status**: Framework provided, needs full implementation

## 📡 API Endpoints

### Flask REST API (Port 5000)

**GET /health**
```json
{"status": "healthy", "model_loaded": true}
```

**POST /predict**
```json
Request:
{
  "candles": [
    {"time": "...", "open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 1000},
    ...
  ]
}

Response:
{
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_up": 0.69,
  "regime": "BULL_WEAK",
  "confidence": 0.82
}
```

### Socket Server (Port 8888)

**Request Format:**
```json
{
  "action": "predict",
  "candles": [...]
}
```

**Response Format:**
```json
{
  "status": "success",
  "buy_prob": 0.71,
  "sell_prob": 0.24,
  "direction_up": 0.69,
  "regime": "BULL_WEAK",
  "confidence": 0.82
}
```

## 🔧 Configuration

Key parameters in `config.py`:

- **SEQUENCE_LENGTH**: 96 (candles per sequence)
- **NUM_FEATURES**: 100 (technical features)
- **D_MODEL**: 256 (transformer dimension)
- **N_HEADS**: 8 (attention heads)
- **N_LAYERS**: 6 (transformer layers)
- **BATCH_SIZE**: 32
- **LEARNING_RATE**: 1e-4
- **NUM_EPOCHS_PRETRAIN**: 50
- **NUM_EPOCHS_SUPERVISED**: 100

## 🚀 Deployment Options

### Option 1: Local Development
```bash
python train.py
python api_server.py
```

### Option 2: GPU Server (GCP/Vertex AI)
```bash
# On GPU VM
python train.py  # Uses CUDA automatically
python mql5_socket_server.py  # For MQL5
```

### Option 3: Docker (Future)
```dockerfile
# Dockerfile can be added for containerized deployment
```

## 📊 Performance Metrics

Model tracks:
- Buy/Sell accuracy
- Direction accuracy
- Regime classification accuracy
- Multi-task loss components
- Validation loss

## 🔒 Security Considerations

- Socket server accepts connections (configure firewall)
- API server runs on 0.0.0.0 (restrict access in production)
- Model files should be secured
- Trading execution requires careful risk management

## 🐛 Known Limitations

1. **RL Implementation**: Placeholder, needs full PPO/SAC
2. **MQL5 JSON Parsing**: Basic implementation, consider library
3. **Error Handling**: Basic, can be enhanced
4. **Backtesting**: Not included, add separately
5. **Model Monitoring**: No logging/metrics export yet

## 🔮 Future Enhancements

- Full RL implementation with stable-baselines3
- TensorBoard logging
- Model versioning
- Backtesting framework
- Real-time monitoring dashboard
- Docker containerization
- Kubernetes deployment
- Model serving with TorchServe

## ✅ System Status

**Complete Components**: ✅
- Feature Engineering
- Label Generation
- Sequence Creation
- Transformer Model
- Training Pipeline
- API Server
- Socket Server
- MQL5 EA
- Documentation

**Optional/Placeholder**: ⚠️
- RL Training (framework ready, needs implementation)
- Advanced monitoring
- Production deployment scripts

## 📝 Usage Summary

1. **Train**: `python train.py`
2. **Test**: `python test_inference.py`
3. **API**: `python api_server.py`
4. **MQL5**: `python mql5_socket_server.py` + deploy EA
5. **Sample Data**: `python create_sample_data.py`

---

**System is production-ready for training and inference. RL component is optional and can be enhanced later.**
