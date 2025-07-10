# Multi-stage build to download models in a separate stage
# This helps manage disk space during build

# Stage 1: Download models
FROM python:3.10-slim as model-downloader

# Install required packages for downloading
RUN pip install --no-cache-dir huggingface_hub[cli,hf_transfer]

# Copy download script
COPY download_models_docker.sh /download_models_docker.sh
RUN chmod +x /download_models_docker.sh

# Download models
WORKDIR /workspace
RUN /download_models_docker.sh

# Stage 2: Final image
FROM runpod/pytorch:2.1.1-py3.10-cuda12.1.1-devel-ubuntu22.04

# Set the working directory inside the container
WORKDIR /workspace

# Install system dependencies if any are needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to reduce image size
# Note: The base image already includes PyTorch 2.1.1, we'll use that to save space
# If you need PyTorch 2.7.0 specifically, consider building locally or using a larger build environment
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Copy pre-downloaded models from the first stage
COPY --from=model-downloader /workspace/models /workspace/models

# Check PyTorch compatibility
COPY check_pytorch_compatibility.py .
RUN python check_pytorch_compatibility.py

# Make scripts executable
RUN chmod +x download_models.sh start.sh

# Expose the port Gradio runs on (default is 7860)
EXPOSE 7860

# The default command to run when starting the container.
# This will launch the Gradio web UI.
CMD ["bash", "start.sh"]