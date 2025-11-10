# 📦 Natron V2 Installation Guide

Complete installation instructions for different environments.

---

## 🖥️ Option 1: Local Installation (Ubuntu/Debian)

### Prerequisites
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Python 3.10+
sudo apt install python3.10 python3.10-venv python3-pip -y

# Install CUDA (for GPU support)
# Follow: https://developer.nvidia.com/cuda-downloads
```

### Installation Steps

```bash
# 1. Clone repository
cd /path/to/your/projects
git clone <repository-url> natron-v2
cd natron-v2

# 2. Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip setuptools wheel

# 4. Install dependencies
pip install -r requirements.txt

# 5. Install PyTorch with CUDA (for GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Or install PyTorch for CPU only
pip install torch torchvision torchaudio

# 6. Verify installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 7. Generate sample data (optional)
python utils/generate_sample_data.py --candles 10000 --output data/data_export.csv

# 8. Test training
python train_natron.py --data data/data_export.csv --skip-pretrain --skip-rl

# 9. Test inference
python src/inference/socket_server.py
```

---

## 🐳 Option 2: Docker Installation

### Prerequisites
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo apt install docker-compose -y

# Install NVIDIA Docker (for GPU support)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Build and Run

```bash
# 1. Clone repository
git clone <repository-url> natron-v2
cd natron-v2

# 2. Place your data
cp /path/to/your/data.csv data/data_export.csv

# 3. Build Docker image
cd deployment
docker-compose build

# 4. Start services
docker-compose up -d

# 5. View logs
docker-compose logs -f natron-server

# 6. Access services
# - Socket server: localhost:9090
# - Flask API: localhost:5000
# - Tensorboard: localhost:6006

# 7. Stop services
docker-compose down
```

### Training in Docker

```bash
# Run training inside container
docker-compose exec natron-server python train_natron.py --data data/data_export.csv

# Or run as one-off command
docker-compose run --rm natron-server python train_natron.py --data data/data_export.csv
```

---

## ☁️ Option 3: Google Cloud Platform (Vertex AI)

### Using Vertex AI Workbench

```bash
# 1. Create a new notebook instance with GPU
# - Machine type: n1-standard-8
# - GPU: NVIDIA Tesla T4
# - Environment: PyTorch 2.0

# 2. In the notebook terminal:
git clone <repository-url> natron-v2
cd natron-v2

# 3. Install dependencies
pip install -r requirements.txt

# 4. Upload your data
# - Use Jupyter file browser
# - Or gsutil: gsutil cp gs://your-bucket/data.csv data/data_export.csv

# 5. Train
python train_natron.py --data data/data_export.csv

# 6. Download trained model
# gsutil cp model/natron_v2.pt gs://your-bucket/
```

### Using Vertex AI Training Jobs

```bash
# 1. Prepare Docker image
gcloud builds submit --tag gcr.io/YOUR_PROJECT/natron-v2

# 2. Submit training job
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=natron-training \
  --worker-pool-spec=machine-type=n1-highmem-8,replica-count=1,accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1,container-image-uri=gcr.io/YOUR_PROJECT/natron-v2 \
  --args="--data,gs://your-bucket/data.csv"

# 3. Monitor job
gcloud ai custom-jobs list --region=us-central1

# 4. Download results
gsutil cp -r gs://your-bucket/model ./
```

---

## 🔧 Post-Installation Configuration

### 1. Configure Settings

Edit `config/config.yaml`:

```yaml
# Adjust based on your hardware
training:
  batch_size: 32  # Reduce if OOM
  
model:
  d_model: 256    # Reduce for smaller models
  
deployment:
  device: "cuda"  # or "cpu"
```

### 2. Prepare Data

Your CSV should have this format:

```csv
time,open,high,low,close,volume
1699000000,1.0850,1.0865,1.0840,1.0858,15000
1699000900,1.0858,1.0870,1.0850,1.0862,12500
...
```

**Requirements:**
- Minimum 1,000 candles (10,000+ recommended)
- Columns: time, open, high, low, close, volume
- Time in Unix timestamp
- Chronological order

### 3. Test Installation

```bash
# Check system health
python deployment/monitor_natron.py --once

# Expected output:
# ✓ CPU, RAM, Disk usage
# ✓ GPU detected (if available)
# ✓ Model files status
```

---

## 🚀 Quick Validation

### Test 1: Generate Sample Data
```bash
python utils/generate_sample_data.py --candles 5000
# Should create: data/data_export.csv (5000 candles)
```

### Test 2: Quick Training
```bash
python train_natron.py \
    --data data/data_export.csv \
    --skip-pretrain \
    --skip-rl
# Should complete in 10-20 minutes (GPU) or 1-2 hours (CPU)
```

### Test 3: Inference Test
```bash
# Start server
python src/inference/flask_api.py &

# Test prediction
curl http://localhost:5000/health
# Expected: {"status": "healthy", "model_loaded": true}
```

---

## 🛠️ Troubleshooting

### Issue: "CUDA out of memory"

**Solution:**
```yaml
# Edit config/config.yaml
training:
  batch_size: 16  # or 8
```

### Issue: "No module named 'torch'"

**Solution:**
```bash
pip install torch torchvision torchaudio
```

### Issue: "Model not found"

**Solution:**
```bash
# Train the model first
python train_natron.py --data data/data_export.csv
```

### Issue: "Socket connection refused"

**Solution:**
```bash
# Check if server is running
netstat -an | grep 9090

# Start server
python src/inference/socket_server.py
```

### Issue: "Not enough data"

**Solution:**
```bash
# Generate more sample data
python utils/generate_sample_data.py --candles 10000

# Or use real data with 10,000+ candles
```

---

## 📊 Hardware Requirements

### Minimum (CPU Training)
- CPU: 4 cores
- RAM: 8 GB
- Storage: 20 GB
- OS: Ubuntu 20.04+

### Recommended (GPU Training)
- CPU: 8+ cores
- RAM: 16+ GB
- GPU: NVIDIA with 6+ GB VRAM (RTX 3060+)
- Storage: 50 GB SSD
- OS: Ubuntu 22.04

### Production Inference
- CPU: 4+ cores OR
- GPU: NVIDIA with 4+ GB VRAM
- RAM: 8+ GB
- Network: Low latency (< 10ms to MT5)

---

## 🔐 Security Checklist

- [ ] Change default ports if exposed to internet
- [ ] Use firewall to restrict access
- [ ] Enable HTTPS for Flask API (if public)
- [ ] Use authentication for API endpoints
- [ ] Keep dependencies updated
- [ ] Monitor logs for suspicious activity

---

## 📈 Performance Optimization

### For Training
1. Use GPU (10-50x faster)
2. Increase batch size (if RAM allows)
3. Use mixed precision training (AMP)
4. Enable cudnn benchmarking

### For Inference
1. Use GPU for < 20ms latency
2. Batch predictions when possible
3. Keep model loaded (don't reload)
4. Use socket server for lowest latency

---

## 📚 Next Steps

1. ✅ Installation complete
2. ✅ Data prepared
3. ✅ Model trained
4. 📖 Read [QUICKSTART.md](QUICKSTART.md)
5. 📖 Read [README.md](README.md)
6. 🚀 Deploy to production
7. 📊 Monitor performance
8. 🔄 Iterate and improve

---

## 🆘 Getting Help

**Common Issues:**
- Check [Troubleshooting](#troubleshooting) section
- Review [README.md](README.md)
- Check system monitor: `python deployment/monitor_natron.py --once`

**Still Stuck?**
- Open an issue on GitHub
- Check logs in `logs/` directory
- Verify system requirements

---

**Installation Complete! 🎉**

Now you're ready to train and deploy Natron V2!
