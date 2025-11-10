# Natron AI Trading System - Docker Container
# Version: Natron V1.0 - Galaxy-class

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/processed models logs configs

# Set environment variables
ENV PYTHONPATH=/app
ENV NATRON_VERSION=1.0
ENV NATRON_CODENAME=Galaxy-class

# Expose socket port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost', 8888)); s.close()" || exit 1

# Default command
CMD ["python", "scripts/start_natron.sh"]
