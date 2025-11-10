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
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/checkpoints

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NATRON_ENV=production

# Expose ports (for socket communication)
EXPOSE 8888 9999

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import socket; s=socket.socket(); s.connect(('localhost', 8888)); s.close()" || exit 1

# Start script
CMD ["bash", "start_natron.sh"]
