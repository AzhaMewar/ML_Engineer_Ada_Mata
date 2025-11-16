# Start from a slim Python base
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies (for OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY bsort/ /app/bsort/
COPY settings.yaml /app/

# Set the entrypoint to the CLI
ENTRYPOINT ["python", "bsort/main.py"]

# Example: CMD ["run-infer", "--config", "settings.yaml", "--image", "sample.jpg"]