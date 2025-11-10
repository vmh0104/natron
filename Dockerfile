FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8080 7600

ENTRYPOINT ["scripts/start_natron.sh", "configs/natron_config.yaml", "logs"]
