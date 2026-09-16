# Dockerfile for CoolMind - Thermal-aware AI Inference Engine
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy source code
COPY coolmind/ coolmind/
COPY README.md .
COPY LICENSE .
COPY config.yaml.template .
COPY docker-entrypoint.sh .

# Make entrypoint executable
RUN chmod +x docker-entrypoint.sh

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Expose ports
EXPOSE 8000  # CLI and web API

# Use custom entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

