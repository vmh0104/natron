# 📚 Natron Transformer - Project Index

**Quick navigation guide for all project files and resources**

---

## 🚀 Getting Started (Start Here!)

1. **[QUICKSTART.md](QUICKSTART.md)** - Get up and running in 5 minutes
2. **[INSTALLATION.md](INSTALLATION.md)** - Detailed installation guide
3. **[README.md](README.md)** - Complete system documentation

---

## 📖 Documentation

| Document | Purpose | When to Read |
|----------|---------|--------------|
| [QUICKSTART.md](QUICKSTART.md) | 5-minute setup | First time user |
| [INSTALLATION.md](INSTALLATION.md) | Installation steps | Before setup |
| [README.md](README.md) | Full documentation | Deep dive |
| [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) | Technical overview | Understanding architecture |
| [COMPLETION_REPORT.md](COMPLETION_REPORT.md) | Project status | Review deliverables |
| [config.yaml](config.yaml) | Configuration | Customization |

---

## 💻 Source Code

### Core AI Modules (`src/`)
| File | Purpose | Lines |
|------|---------|-------|
| [feature_engine.py](src/feature_engine.py) | 100+ technical indicators | ~700 |
| [labeling.py](src/labeling.py) | Buy/sell/regime labels | ~350 |
| [dataset_loader.py](src/dataset_loader.py) | Sequence creation | ~350 |
| [model_natron.py](src/model_natron.py) | Transformer architecture | ~450 |
| [losses.py](src/losses.py) | Multi-task losses | ~400 |
| [pretrain.py](src/pretrain.py) | Phase 1 training | ~350 |
| [supervised.py](src/supervised.py) | Phase 2 training | ~400 |
| [reinforcement.py](src/reinforcement.py) | Phase 3 training (PPO) | ~450 |
| [train_natron.py](src/train_natron.py) | Main training script | ~300 |

### API Servers (`api/`)
| File | Purpose | Lines |
|------|---------|-------|
| [inference.py](api/inference.py) | Inference engine | ~300 |
| [api_server.py](api/api_server.py) | Flask REST API | ~250 |
| [socket_server.py](api/socket_server.py) | Socket server for MQL5 | ~300 |

### Trading Integration (`mql5/`)
| File | Purpose | Lines |
|------|---------|-------|
| [natron_ea.mq5](mql5/natron_ea.mq5) | MetaTrader 5 Expert Advisor | ~600 |

### Deployment (`deployment/`)
| File | Purpose | Type |
|------|---------|------|
| [Dockerfile](deployment/Dockerfile) | Container setup | Docker |
| [docker-compose.yml](deployment/docker-compose.yml) | Multi-service orchestration | Docker Compose |
| [start_natron.sh](deployment/start_natron.sh) | Startup script | Bash |
| [monitor_natron.py](deployment/monitor_natron.py) | System monitoring | Python |

### Tests (`tests/`)
| File | Purpose | Lines |
|------|---------|-------|
| [test_integration.py](tests/test_integration.py) | Integration tests | ~250 |

---

## ⚙️ Configuration Files

| File | Purpose |
|------|---------|
| [config.yaml](config.yaml) | Main configuration |
| [requirements.txt](requirements.txt) | Python dependencies |
| [.gitignore](.gitignore) | Git ignore rules |
| [sample_request.json](sample_request.json) | API test request |

---

## 📁 Directory Structure

```
/workspace/
├── 📘 Documentation
│   ├── README.md
│   ├── QUICKSTART.md
│   ├── INSTALLATION.md
│   ├── PROJECT_SUMMARY.md
│   ├── COMPLETION_REPORT.md
│   └── INDEX.md (you are here)
│
├── 🧠 Core AI (src/)
│   ├── Feature Engineering
│   ├── Labeling System
│   ├── Dataset Loader
│   ├── Model Architecture
│   ├── Loss Functions
│   └── Training Pipeline (3 phases)
│
├── 🌐 APIs (api/)
│   ├── Inference Engine
│   ├── Flask REST API
│   └── Socket Server
│
├── 📊 Trading (mql5/)
│   └── MQL5 Expert Advisor
│
├── 🚀 Deployment (deployment/)
│   ├── Docker Configuration
│   ├── Startup Scripts
│   └── Monitoring Tools
│
├── 🧪 Tests (tests/)
│   └── Integration Tests
│
├── 📂 Data Directories
│   ├── data/ (OHLCV data)
│   ├── models/ (trained models)
│   ├── checkpoints/ (training checkpoints)
│   ├── logs/ (system logs)
│   └── cache/ (feature cache)
│
└── ⚙️ Configuration
    ├── config.yaml
    ├── requirements.txt
    └── sample_request.json
```

---

## 🎯 Common Tasks

### Training
```bash
# Full pipeline
python src/train_natron.py --phase all

# Individual phases
python src/train_natron.py --phase pretrain
python src/train_natron.py --phase supervised
python src/train_natron.py --phase reinforcement
```

### Running Services
```bash
# All services
./deployment/start_natron.sh --mode all

# API only
./deployment/start_natron.sh --mode api

# Socket only
./deployment/start_natron.sh --mode socket

# Docker
./deployment/start_natron.sh --mode docker
```

### Testing
```bash
# Integration tests
python tests/test_integration.py

# Individual components
python src/feature_engine.py
python src/model_natron.py
```

### Monitoring
```bash
# System monitor
python deployment/monitor_natron.py

# TensorBoard
tensorboard --logdir=runs --port=6006
```

---

## 🔍 Finding What You Need

### "I want to..."

**...get started quickly**
→ [QUICKSTART.md](QUICKSTART.md)

**...install the system**
→ [INSTALLATION.md](INSTALLATION.md)

**...understand the architecture**
→ [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)

**...add custom features**
→ [src/feature_engine.py](src/feature_engine.py)

**...modify training**
→ [config.yaml](config.yaml) and [src/train_natron.py](src/train_natron.py)

**...deploy with Docker**
→ [deployment/Dockerfile](deployment/Dockerfile)

**...integrate with MT5**
→ [mql5/natron_ea.mq5](mql5/natron_ea.mq5)

**...test the API**
→ [sample_request.json](sample_request.json)

**...monitor the system**
→ [deployment/monitor_natron.py](deployment/monitor_natron.py)

**...troubleshoot issues**
→ [INSTALLATION.md](INSTALLATION.md) (Troubleshooting section)

---

## 📊 Key Metrics

- **Total Files**: 26 project files
- **Python Code**: ~5,200 lines
- **MQL5 Code**: ~600 lines
- **Documentation**: ~2,500 lines
- **Components**: 17 modules
- **APIs**: 2 servers (Flask + Socket)
- **Training Phases**: 3 (Pretrain, Supervised, RL)
- **Features**: 100+ technical indicators
- **Model Heads**: 4 prediction tasks

---

## 🏆 Project Status

✅ **100% Complete**

All components implemented, tested, and documented.  
Ready for production deployment.

---

## 🆘 Need Help?

1. Check [QUICKSTART.md](QUICKSTART.md) for quick solutions
2. Review [INSTALLATION.md](INSTALLATION.md) troubleshooting
3. Read [README.md](README.md) for detailed explanations
4. Examine code comments in source files
5. Run tests: `python tests/test_integration.py`

---

## 📝 Notes

- All file paths are relative to `/workspace/`
- Configuration is centralized in `config.yaml`
- Logs are stored in `logs/natron.log`
- Models are saved in `models/natron_v2.pt`
- Use `deployment/start_natron.sh` for easy startup

---

**Happy trading with Natron Transformer! 🚀**

*Last updated: 2025-11-10*
