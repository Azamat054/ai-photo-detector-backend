FROM python:3.10-slim

WORKDIR /app

# Install system dependencies  
RUN apt-get update && apt-get install -y \
    libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies with CPU-only PyTorch
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.3.1 && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py detector.py ./

# Create cache directory for models
RUN mkdir -p ./models_cache

# Expose port
EXPOSE 8000

# Run the FastAPI application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
