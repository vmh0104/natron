"""
Natron Transformer Configuration
"""
import os
from pathlib import Path

class Config:
    # Paths
    DATA_PATH = "data_export.csv"
    MODEL_DIR = Path("model")
    MODEL_PATH = MODEL_DIR / "natron_v2.pt"
    
    # Data parameters
    SEQUENCE_LENGTH = 96
    NUM_FEATURES = 100
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_EPOCHS_PRETRAIN = 50
    NUM_EPOCHS_SUPERVISED = 100
    NUM_EPOCHS_RL = 50
    
    # Model architecture
    D_MODEL = 256
    N_HEADS = 8
    N_LAYERS = 6
    D_FF = 1024
    DROPOUT = 0.1
    
    # Multi-task weights
    WEIGHT_BUY = 1.0
    WEIGHT_SELL = 1.0
    WEIGHT_DIRECTION = 1.0
    WEIGHT_REGIME = 1.0
    
    # Pretraining
    MASK_RATIO = 0.15
    
    # RL parameters
    RL_ALPHA = 0.1  # turnover penalty
    RL_BETA = 0.05  # drawdown penalty
    
    # API Server
    API_HOST = "0.0.0.0"
    API_PORT = 5000
    
    # MQL5 Socket Server
    SOCKET_HOST = "0.0.0.0"
    SOCKET_PORT = 8888
    
    # Device
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    
    @classmethod
    def ensure_dirs(cls):
        """Create necessary directories"""
        cls.MODEL_DIR.mkdir(exist_ok=True)
