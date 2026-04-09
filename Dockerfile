# Advanced AI Research 2026 - Production Docker Image
# Multi-stage build for optimal size and security

# Base stage - Python environment
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libsndfile1 \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Development stage
FROM base as development

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install additional development dependencies
RUN pip install --no-cache-dir \
    streamlit==1.28.0 \
    plotly==5.15.0 \
    jupyter==1.0.0 \
    jupyterlab==4.0.0

# Copy source code
COPY . /app
WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose ports
EXPOSE 8500 8888

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8500/_stcore/health || exit 1

# Development commands
CMD ["streamlit", "run", "app.py", "--server.port=8500", "--server.address=0.0.0.0"]

# Production stage
FROM base as production

# Install only production dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir streamlit==1.28.0 plotly==5.15.0

# Copy application
COPY --chown=app:app . /app
WORKDIR /app

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8500

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8500/_stcore/health || exit 1

# Production command
CMD ["streamlit", "run", "app.py", "--server.port=8500", "--server.address=0.0.0.0", "--server.headless=true"]

# GPU-enabled version
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 as gpu

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3.9-pip \
    python3.9-dev \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set Python environment
ENV PYTHONPATH=/usr/lib/python3.9/site-packages
RUN ln -s /usr/bin/python3.9 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir streamlit==1.28.0 plotly==5.15.0

# Copy application
COPY --chown=app:app . /app
WORKDIR /app

# Create user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose port
EXPOSE 8500

# GPU command
CMD ["streamlit", "run", "app.py", "--server.port=8500", "--server.address=0.0.0.0"]
