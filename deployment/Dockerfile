# Natron Transformer - Docker Image
# GPU-optimized container for training and inference

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /workspace

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data models checkpoints logs cache runs

# Expose ports
EXPOSE 5000 9090 6006

# Set default command
CMD ["python3", "src/train_natron.py", "--config", "config.yaml", "--phase", "all"]
