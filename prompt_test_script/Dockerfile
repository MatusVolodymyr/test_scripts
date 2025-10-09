FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY test_detector_prompts.py .
COPY sample_prompts.csv .

# Create directories for I/O
RUN mkdir -p /tmp/input /tmp/output /app/test_results

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8
ENV OUTPUT_DIR=/tmp/output

# Health check (optional, useful for ECS)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Set default entrypoint
ENTRYPOINT ["python", "test_detector_prompts.py"]

# Default command (shows help if no args provided)
CMD ["--help"]
