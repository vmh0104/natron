FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN python -m pip install --upgrade pip
RUN pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
ENV PYTHONPATH=/app

EXPOSE 8000
CMD ["bash", "start_natron.sh"]
