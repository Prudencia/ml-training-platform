# Backend Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (libgl1 replaces libgl1-mesa-glx in newer Debian)
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create storage directories
RUN mkdir -p storage/datasets storage/models storage/venvs storage/logs storage/configs \
    storage/annotations storage/exports storage/pretrained_models storage/detectx_builds \
    database

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
