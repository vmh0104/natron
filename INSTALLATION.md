# 🔧 Natron Transformer - Installation Guide

Complete installation instructions for Ubuntu/Debian systems.

---

## 📋 System Requirements

### Hardware
- **CPU**: 4+ cores (8+ recommended)
- **RAM**: 16GB minimum (32GB recommended)
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)
- **Storage**: 50GB free space

### Software
- **OS**: Ubuntu 20.04+, Debian 11+, or compatible Linux
- **Python**: 3.10 or higher
- **CUDA**: 12.1+ (if using GPU)
- **cuDNN**: 8.9+ (if using GPU)

---

## 🚀 Installation Methods

### Method 1: Native Installation (Recommended for Development)

#### Step 1: Update System
```bash
sudo apt update && sudo apt upgrade -y
```

#### Step 2: Install System Dependencies
```bash
sudo apt install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    pkg-config
```

#### Step 3: Install CUDA (if using GPU)
```bash
# Download CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt update
sudo apt install -y cuda-toolkit-12-1

# Add to PATH
echo 'export PATH=/usr/local/cuda-12.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Verify installation
nvidia-smi
nvcc --version
```

#### Step 4: Clone Repository
```bash
cd /workspace
# Or if cloning from a repository:
# git clone https://github.com/your-repo/natron-transformer.git
# cd natron-transformer
```

#### Step 5: Create Virtual Environment
```bash
python3.10 -m venv venv
source venv/bin/activate
```

#### Step 6: Install Python Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

#### Step 7: Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

### Method 2: Docker Installation (Recommended for Production)

#### Step 1: Install Docker
```bash
# Remove old versions
sudo apt remove docker docker-engine docker.io containerd runc

# Install dependencies
sudo apt update
sudo apt install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify
docker --version
docker compose version
```

#### Step 2: Install NVIDIA Container Toolkit (for GPU)
```bash
# Add repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

# Install
sudo apt update
sudo apt install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

#### Step 3: Build Docker Image
```bash
cd /workspace
docker build -t natron-transformer -f deployment/Dockerfile .
```

#### Step 4: Run with Docker Compose
```bash
docker compose -f deployment/docker-compose.yml up -d
```

---

## 🔍 Verification

### Test Python Environment
```bash
python tests/test_integration.py
```

### Test Feature Generation
```bash
python src/feature_engine.py
```

### Test Model Creation
```bash
python src/model_natron.py
```

### Test API Server
```bash
# Terminal 1: Start server
python api/api_server.py

# Terminal 2: Test endpoint
curl http://localhost:5000/health
```

---

## 🐛 Troubleshooting

### Issue: CUDA not detected
```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA installation
nvcc --version

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Issue: Import errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python version
python --version  # Should be 3.10+
```

### Issue: Permission denied
```bash
# Make scripts executable
chmod +x deployment/start_natron.sh

# Fix ownership
sudo chown -R $USER:$USER /workspace
```

### Issue: Out of memory
```bash
# Reduce batch size in config.yaml
# Edit: training.pretrain.batch_size and training.supervised.batch_size
```

### Issue: Docker GPU not working
```bash
# Check NVIDIA Container Toolkit
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If fails, reconfigure
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

---

## 📦 Optional Dependencies

### TA-Lib (for additional technical indicators)
```bash
# Install TA-Lib C library
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
cd ..

# Install Python wrapper
pip install TA-Lib
```

### TensorBoard
```bash
pip install tensorboard

# Start TensorBoard
tensorboard --logdir=runs --port=6006
```

---

## 🔄 Updating

### Update Python Packages
```bash
pip install -r requirements.txt --upgrade
```

### Update Docker Image
```bash
docker compose -f deployment/docker-compose.yml down
docker build -t natron-transformer -f deployment/Dockerfile . --no-cache
docker compose -f deployment/docker-compose.yml up -d
```

### Pull Latest Code (if from Git)
```bash
git pull origin main
pip install -r requirements.txt
```

---

## 🎯 Post-Installation

### 1. Prepare Your Data
```bash
# Place OHLCV data in data/data_export.csv
# Format: time, open, high, low, close, volume
```

### 2. Configure System
```bash
# Edit config.yaml to customize
vim config.yaml
```

### 3. Train Model
```bash
./deployment/start_natron.sh --mode train
```

### 4. Start Services
```bash
./deployment/start_natron.sh --mode all
```

---

## 🆘 Support

If you encounter issues:

1. Check logs: `tail -f logs/natron.log`
2. Review configuration: `cat config.yaml`
3. Verify installation: `python tests/test_integration.py`
4. Check system resources: `python deployment/monitor_natron.py`

---

## ✅ Installation Checklist

- [ ] System dependencies installed
- [ ] CUDA and cuDNN installed (if using GPU)
- [ ] Python 3.10+ installed
- [ ] Virtual environment created
- [ ] Python packages installed
- [ ] CUDA detected by PyTorch (if GPU)
- [ ] Tests passed
- [ ] Data prepared
- [ ] Configuration customized
- [ ] Docker installed (if using Docker)
- [ ] NVIDIA Container Toolkit installed (if GPU + Docker)

---

**Installation Complete! Ready to train your AI trading model! 🚀**
