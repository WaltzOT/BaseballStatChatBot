# Use NVIDIA PyTorch base image with CPU support
FROM pytorch/pytorch:latest

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    wget \
    build-essential \
    && apt-get clean

    # Upgrade pip
RUN pip install --upgrade pip

# Install Cython first
RUN pip install Cython

# Clean any existing installations of huggingface_hub
RUN pip uninstall -y huggingface_hub

# Install compatible huggingface_hub version
RUN pip install "huggingface_hub==0.10.1"

# Upgrade pip and install dependencies
RUN pip install \
    numpy \
    pandas \
    transformers==4.21.1 \
    nemo-toolkit[nlp]==1.21.0 \
    torchmetrics

# Copy your project files into the container
COPY . /workspace /workspace

# Expose ports (optional)
EXPOSE 8080

# Default entry point for training
CMD ["python3", "scripts/initialTraining.py"]
