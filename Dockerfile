# Use a base image with PyTorch and CUDA pre-installed.
# The version is chosen to be compatible with the requirements.txt
FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04

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

# The project requires models to be downloaded.
# The download_models.sh script is provided for this purpose.
# You can uncomment the following line to download the models during the build process.
# Note: This will increase the image size significantly.
# An alternative is to mount the models directory as a volume during runtime.
RUN chmod +x download_models.sh && ./download_models.sh

# Expose the port Gradio runs on (default is 7860)
EXPOSE 7860

# The default command to run when starting the container.
# This will launch the Gradio web UI.
CMD ["bash", "start.sh"]
