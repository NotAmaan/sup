# Use a base image with PyTorch and CUDA pre-installed.
# The version is chosen to be compatible with the requirements.txt
FROM runpod/pytorch:0.7.1-dev-ubuntu2404-cu1263-torch271

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
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Create a script to download models
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN chmod +x download_models.sh && ./download_models.sh

# Expose the port Gradio runs on (default is 7860)
EXPOSE 7860

# The default command to run when starting the container.
# This will launch the Gradio web UI.
CMD ["bash", "start.sh"]
