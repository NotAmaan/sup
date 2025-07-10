# Use a base image with PyTorch and CUDA pre-installed.
FROM runpod/pytorch:0.7.1-dev-ubuntu2404-cu1263-torch271

# Set the working directory inside the container
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file (after you've removed the torch lines)
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# --- START OF RECOMMENDED FIX ---
#
# Stage 1: Install PyTorch, torchvision, and torchaudio for the correct CUDA version.
# This command explicitly uses the PyTorch index URL to get GPU-enabled wheels.
# This solves the primary issue of having the right Torch version available.
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Stage 2: Install difficult-to-compile packages that depend on PyTorch.
# Now that a proper torch is installed, the build for xformers and triton will succeed.
# The `|| true` is a safeguard in case one of these isn't in your requirements.txt
RUN pip install --no-cache-dir -r <(egrep 'xformers|triton' requirements.txt || true)

# Stage 3: Install the rest of the requirements.
# The '-v' flag inverts the match, installing everything that ISN'T xformers or triton.
RUN pip install --no-cache-dir -r <(egrep -v 'xformers|triton' requirements.txt)
#
# --- END OF RECOMMENDED FIX ---

# Copy the rest of the application code
COPY . .

# Create a script to download models
ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN chmod +x download_models.sh && ./download_models.sh

# Expose the port
EXPOSE 7860

# Default command
CMD ["bash", "start.sh"]