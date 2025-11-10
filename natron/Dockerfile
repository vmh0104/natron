# Natron End-to-End AI Trading System - Dockerfile
# Galaxy-class Trading System

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

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data models configs logs

# Set environment variables
ENV PYTHONPATH=/app
ENV NATRON_VERSION=1.0
ENV NATRON_CODENAME=Galaxy-class

# Expose socket port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost', 8888)); s.close()" || exit 1

# Default command (can be overridden)
CMD ["python", "scripts/start_natron.sh"]
