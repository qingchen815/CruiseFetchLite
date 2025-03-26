# Use NVIDIA CUDA base image
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python3.8
RUN ln -sf /usr/bin/python3.8 /usr/bin/python

# Upgrade pip
RUN python -m pip install --no-cache-dir pip setuptools wheel

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install TensorFlow with GPU support
RUN pip install --no-cache-dir tensorflow==2.12.0

# Create working directory
WORKDIR /app

# Copy project files
COPY . /app/

# Create data directory
RUN mkdir -p /data

# Set default command
CMD ["python", "example.py", "--simulate"] 