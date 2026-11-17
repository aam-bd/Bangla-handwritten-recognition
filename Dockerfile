# Production Dockerfile for Bangla Handwritten Word Recognition System
# Assignment 04 - SICIP, BRAC University

FROM python:3.11-slim

# Avoid interactive prompts and python buffering
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    KERAS_BACKEND=torch

# Install system dependencies for OpenCV headless operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application artifacts and source code
COPY labels.json .
COPY app.py .
COPY src/ ./src/
COPY models/ ./models/

# Expose Streamlit default port
EXPOSE 8501

# Container healthcheck
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Launch Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
