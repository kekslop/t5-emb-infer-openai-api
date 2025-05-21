FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (FRIDA directory is excluded via .dockerignore)
COPY . .

# Create the FRIDA directory for mounting
RUN mkdir -p FRIDA

# Expose the port the app runs on
EXPOSE 8000

# Check if .env exists, and copy from env.sample if not
CMD if [ ! -f .env ]; then echo "No .env file found, copying from env.sample..."; cp -n env.sample .env; fi && \
    uvicorn src.main:app --host 0.0.0.0 --port 8000 