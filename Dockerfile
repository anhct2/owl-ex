FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/app:$PYTHONPATH" \
    PLAYWRIGHT_BROWSERS_PATH=/root/.cache/ms-playwright \
    DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    xvfb \
    xauth \
    x11-utils \
    build-essential \
    python3-dev \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy project files first for better layer caching
COPY pyproject.toml README.md ./
COPY owl/ ./owl/
COPY examples/ ./examples/
COPY .env .
COPY owl_api_server.py .

# Install owl package in development mode and all dependencies
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir python-dotenv fastapi uvicorn sse-starlette google-generativeai

# Install playwright and browsers
RUN pip install playwright
RUN playwright install --with-deps chromium

# Create startup script for xvfb
RUN printf '#!/bin/bash\nxvfb-run --auto-servernum --server-args="-screen 0 1280x960x24" python "$@"' > /usr/local/bin/xvfb-python && \
    chmod +x /usr/local/bin/xvfb-python

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application with xvfb for headless browser support
CMD ["xvfb-python", "owl_api_server.py"] 