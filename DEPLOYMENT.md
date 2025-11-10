# 🚀 Natron Transformer - Deployment Guide

This guide covers deploying Natron Transformer to production environments.

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Local Deployment](#local-deployment)
3. [Cloud Deployment (GCP)](#cloud-deployment-gcp)
4. [Docker Deployment](#docker-deployment)
5. [MetaTrader 5 Setup](#metatrader-5-setup)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

**Minimum**:
- CPU: 4 cores
- RAM: 8GB
- GPU: NVIDIA GPU with 4GB VRAM (for training)
- Storage: 20GB

**Recommended**:
- CPU: 8+ cores
- RAM: 16GB+
- GPU: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- Storage: 50GB SSD

### Software Requirements

- Python 3.10+
- CUDA 11.8+ (for GPU)
- cuDNN 8.6+
- Docker (optional)
- MetaTrader 5 (for live trading)

---

## Local Deployment

### Step 1: Environment Setup

```bash
# Clone repository
git clone https://github.com/yourusername/natron-transformer.git
cd natron-transformer

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Step 2: Prepare Data

```bash
# Place your OHLCV data
cp /path/to/your/data.csv data/data_export.csv

# Verify data format
head -n 5 data/data_export.csv
```

### Step 3: Train Models

```bash
# Full pipeline (all 3 phases)
python scripts/train_full_pipeline.py

# Monitor training
tensorboard --logdir logs/tensorboard
```

**Training Time Estimates** (RTX 3080):
- Phase 1 (Pretraining): ~3 hours
- Phase 2 (Supervised): ~4 hours
- Phase 3 (RL): ~2 hours
- **Total**: ~9 hours

### Step 4: Start Server

```bash
# Start inference server
./scripts/start_server.sh

# Or manually
python src/server/api_server.py
```

**Verify server is running**:
```bash
curl http://localhost:5000/health
```

Expected response:
```json
{"status": "healthy", "model_loaded": true}
```

---

## Cloud Deployment (GCP)

### Google Cloud Platform Setup

#### 1. Create VM Instance

```bash
# Create GPU instance
gcloud compute instances create natron-gpu \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE \
  --boot-disk-size=50GB \
  --boot-disk-type=pd-ssd
```

#### 2. SSH and Setup

```bash
# SSH to instance
gcloud compute ssh natron-gpu --zone=us-central1-a

# Install dependencies
sudo apt-get update
sudo apt-get install -y git python3-pip

# Clone repo
git clone https://github.com/yourusername/natron-transformer.git
cd natron-transformer

# Install requirements
pip3 install -r requirements.txt
```

#### 3. Configure Firewall

```bash
# Allow Flask API port
gcloud compute firewall-rules create allow-natron-api \
  --allow tcp:5000 \
  --source-ranges 0.0.0.0/0

# Allow socket port
gcloud compute firewall-rules create allow-natron-socket \
  --allow tcp:9090 \
  --source-ranges 0.0.0.0/0
```

#### 4. Start as Service

Create systemd service file:

```bash
sudo nano /etc/systemd/system/natron.service
```

```ini
[Unit]
Description=Natron AI Inference Server
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/home/youruser/natron-transformer
ExecStart=/home/youruser/natron-transformer/venv/bin/python src/server/api_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable natron
sudo systemctl start natron

# Check status
sudo systemctl status natron
```

---

## Docker Deployment

### 1. Create Dockerfile

```bash
# Already included in workspace
cat > Dockerfile << 'EOF'
FROM pytorch/pytorch:2.0.1-cuda11.8-cudnn8-runtime

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create necessary directories
RUN mkdir -p model logs data

# Expose ports
EXPOSE 5000 9090

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Start server
CMD ["python", "src/server/api_server.py"]
EOF
```

### 2. Build and Run

```bash
# Build image
docker build -t natron-transformer:latest .

# Run container
docker run -d \
  --name natron \
  --gpus all \
  -p 5000:5000 \
  -p 9090:9090 \
  -v $(pwd)/model:/app/model \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  --restart unless-stopped \
  natron-transformer:latest

# Check logs
docker logs -f natron
```

### 3. Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  natron:
    image: natron-transformer:latest
    container_name: natron-ai
    runtime: nvidia
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - PYTHONPATH=/app
    ports:
      - "5000:5000"
      - "9090:9090"
    volumes:
      - ./model:/app/model
      - ./data:/app/data
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

```bash
# Start with compose
docker-compose up -d

# View logs
docker-compose logs -f
```

---

## MetaTrader 5 Setup

### 1. Install MQL5 EA

```bash
# Find MetaTrader 5 data folder
# Windows: C:\Users\[Username]\AppData\Roaming\MetaQuotes\Terminal\[ID]\MQL5\Experts
# macOS: ~/Library/Application Support/MetaTrader 5/Experts

# Copy EA
cp mql5/natron_ea.mq5 /path/to/MT5/Experts/
```

### 2. Compile EA

1. Open MetaEditor (F4 in MT5)
2. Open `natron_ea.mq5`
3. Click Compile (F7)
4. Verify: "0 errors, 0 warnings"

### 3. Configure EA

Attach EA to chart and configure:

**Connection Settings**:
- Server IP: `YOUR_SERVER_IP` (or `127.0.0.1` for local)
- Server Port: `9090`

**Trading Settings**:
- Lot Size: `0.01` (start small!)
- Min Buy Prob: `0.60`
- Min Sell Prob: `0.60`
- Min Confidence: `0.50`
- Stop Loss: `100` points
- Take Profit: `200` points

**Risk Management**:
- Max Spread: `30` points
- Max Positions: `1`
- Enable Trailing Stop: `false` (initially)

### 4. Test in Demo Account

⚠️ **Important**: Always test in demo account first!

1. Switch to demo account
2. Enable AutoTrading in MT5
3. Attach EA to chart
4. Monitor for 1-2 weeks
5. Review performance metrics

---

## Monitoring & Maintenance

### 1. Server Monitoring

**Monitor logs**:
```bash
tail -f logs/natron.log
```

**Monitor GPU usage**:
```bash
watch -n 1 nvidia-smi
```

**Monitor server resources**:
```bash
htop
```

### 2. TensorBoard

```bash
tensorboard --logdir logs/tensorboard --host 0.0.0.0 --port 6006
```

Access at: `http://your-server:6006`

### 3. Trading Metrics

Monitor in MQL5 EA panel:
- Signal strength (buy/sell probabilities)
- Market regime
- Confidence levels
- Active positions
- P&L statistics

### 4. Alerts

Set up alerts for:
- Server downtime
- GPU memory issues
- High prediction latency (>100ms)
- Trading losses exceeding threshold
- Unusual market conditions

### 5. Retraining Schedule

**Recommended**:
- **Daily**: Update feature scalers with recent data
- **Weekly**: Retrain if market regime shifts
- **Monthly**: Full retraining (all 3 phases)
- **Quarterly**: Architecture review and updates

```bash
# Automated retraining script
crontab -e

# Add monthly retraining (1st day of month, 2am)
0 2 1 * * cd /path/to/natron && ./scripts/train_full_pipeline.py
```

---

## Troubleshooting

### Issue: Model Not Loading

**Symptoms**: `Failed to load model` error

**Solutions**:
```bash
# Verify model file exists
ls -lh model/natron_supervised_best.pt

# Check file permissions
chmod 644 model/natron_supervised_best.pt

# Verify checkpoint format
python -c "import torch; torch.load('model/natron_supervised_best.pt', map_location='cpu')"
```

### Issue: CUDA Out of Memory

**Symptoms**: `CUDA out of memory` during training

**Solutions**:
```yaml
# Reduce batch size in config.yaml
supervised:
  batch_size: 16  # Was 32

# Or use gradient accumulation
# Edit training scripts
```

```python
# Add gradient accumulation
accumulation_steps = 2
for i, batch in enumerate(dataloader):
    loss = loss / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Issue: Socket Connection Failed (MQL5)

**Symptoms**: `Failed to connect` in EA logs

**Solutions**:

1. **Check server is running**:
```bash
netstat -an | grep 9090
```

2. **Verify firewall**:
```bash
# Linux
sudo ufw allow 9090

# Check Windows firewall settings
```

3. **Test connection manually**:
```python
import socket
s = socket.socket()
s.connect(('localhost', 9090))
s.send(b'{"test": "data"}\n')
print(s.recv(1024))
s.close()
```

### Issue: Slow Predictions

**Symptoms**: Latency >500ms

**Solutions**:

1. **Use FP16 inference**:
```python
# In api_server.py
model = model.half()  # Convert to FP16
```

2. **Batch predictions**:
```python
# Process multiple requests together
```

3. **Reduce sequence length**:
```yaml
# In config.yaml
data:
  sequence_length: 64  # Was 96
```

### Issue: Poor Trading Performance

**Symptoms**: Low win rate, frequent losses

**Diagnostic Steps**:

1. **Check prediction quality**:
```bash
# Run validation
python scripts/evaluate_model.py
```

2. **Review market conditions**:
- Is model trained on similar data?
- Has market regime changed?
- Are spreads too high?

3. **Adjust thresholds**:
```mql5
// In MQL5 EA
input double MinBuyProb = 0.70;   // Increase threshold
input double MinConfidence = 0.60;
```

4. **Retrain on recent data**:
```bash
# Use latest data
python scripts/train_full_pipeline.py --no-skip
```

---

## Performance Optimization

### 1. Model Optimization

**Quantization**:
```python
# Convert to INT8
import torch.quantization
model_int8 = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

**TorchScript**:
```python
# Export to TorchScript
traced_model = torch.jit.trace(model, example_input)
traced_model.save('model/natron_traced.pt')
```

### 2. Server Optimization

**Use Waitress (Production WSGI)**:
```bash
pip install waitress
```

```python
# In api_server.py
from waitress import serve
serve(app, host='0.0.0.0', port=5000, threads=4)
```

**Enable Multi-Processing**:
```bash
gunicorn -w 4 -b 0.0.0.0:5000 src.server.api_server:app
```

### 3. Caching

Add Redis caching for repeated predictions:
```python
import redis
cache = redis.Redis()

# Cache predictions
cache.setex(f'pred:{hash}', 60, json.dumps(prediction))
```

---

## Security Best Practices

1. **API Authentication**:
```python
# Add API key verification
@app.before_request
def verify_api_key():
    api_key = request.headers.get('X-API-KEY')
    if api_key != os.getenv('NATRON_API_KEY'):
        abort(401)
```

2. **HTTPS/TLS**:
```bash
# Use nginx reverse proxy with SSL
sudo apt-get install nginx certbot
sudo certbot --nginx
```

3. **Rate Limiting**:
```python
from flask_limiter import Limiter
limiter = Limiter(app, default_limits=["100 per minute"])
```

4. **Secure Socket Communication**:
```python
# Use SSL socket in production
ssl_context = ssl.create_default_context()
```

---

## Backup & Recovery

### 1. Model Checkpoints

```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d)
tar -czf backups/natron_$DATE.tar.gz model/ config/
```

### 2. Database Backups (if using)

```bash
# Backup training history, predictions, etc.
```

### 3. Disaster Recovery Plan

1. Keep model checkpoints in cloud storage (S3, GCS)
2. Version control all code and configs
3. Document deployment procedures
4. Test recovery process quarterly

---

## Scaling

### Horizontal Scaling

**Load Balancer + Multiple Instances**:

```yaml
# kubernetes deployment
apiVersion: apps/v1
kind: Deployment
metadata:
  name: natron
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: natron
        image: natron-transformer:latest
        resources:
          limits:
            nvidia.com/gpu: 1
```

### Vertical Scaling

- Upgrade to larger GPU (T4 → V100 → A100)
- Increase batch size for throughput
- Optimize model architecture

---

## Support

For deployment issues:
- GitHub Issues: https://github.com/yourusername/natron-transformer/issues
- Email: support@natron-ai.com

---

**Happy Trading! 📈🚀**
