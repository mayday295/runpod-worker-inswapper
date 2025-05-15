FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=on \
    SHELL=/bin/bash

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Upgrade apt packages and install required dependencies
RUN apt update && \
    apt upgrade -y && \
    apt install -y \
      python3-dev \
      python3-pip \
      python3.10-venv \
      fonts-dejavu-core \
      rsync \
      git \
      git-lfs \
      jq \
      moreutils \
      aria2 \
      wget \
      curl \
      libglib2.0-0 \
      libsm6 \
      libgl1 \
      libxrender1 \
      libxext6 \
      ffmpeg \
      unzip \
      libgoogle-perftools-dev \
      procps && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean -y

# Set working directory
WORKDIR /workspace

# Install Torch
RUN pip3 install --no-cache-dir torch==2.4.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Inswapper Serverless Worker
COPY . /workspace/runpod-worker-inswapper
WORKDIR /workspace/runpod-worker-inswapper
RUN pip3 install -r requirements.txt && \
    pip3 uninstall -y onnxruntime && \
    pip3 install onnxruntime-gpu

# ── copy the ONNX you downloaded ──────────────────────────────────────────────
COPY checkpoints/inswapper_128.onnx /workspace/runpod-worker-inswapper/checkpoints/inswapper_128.onnx

# ── download buffalo_l (needs RUN!) ───────────────────────────────────────────
RUN cd /workspace/runpod-worker-inswapper/checkpoints && \
    mkdir -p models && cd models && \
    wget -q https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip && \
    mkdir buffalo_l && \
    cd buffalo_l && \
    unzip -q ../buffalo_l.zip


# Install CodeFormer
RUN cd /workspace/runpod-worker-inswapper && \
    git lfs install && \
    git clone https://huggingface.co/spaces/sczhou/CodeFormer

# Download CodeFormer weights
RUN cd /workspace/runpod-worker-inswapper && \
    mkdir -p CodeFormer/CodeFormer/weights/CodeFormer && \
    wget -O CodeFormer/CodeFormer/weights/CodeFormer/codeformer.pth "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth" && \
    mkdir -p CodeFormer/CodeFormer/weights/facelib && \
    wget -O CodeFormer/CodeFormer/weights/facelib/detection_Resnet50_Final.pth "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth" && \
    wget -O CodeFormer/CodeFormer/weights/facelib/parsing_parsenet.pth "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth" && \
    mkdir -p CodeFormer/CodeFormer/weights/realesrgan && \
    wget -O CodeFormer/CodeFormer/weights/realesrgan/RealESRGAN_x2plus.pth "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/RealESRGAN_x2plus.pth"

# Copy handler to ensure its the latest
COPY --chmod=755 rp_handler.py /workspace/runpod-worker-inswapper/rp_handler.py

# Docker container start script
COPY --chmod=755 start.sh /start.sh

# Start the container
ENTRYPOINT /start.sh
