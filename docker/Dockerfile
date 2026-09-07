# CRDD training and evaluation image.
# Runs 01_prepare_data.py through 06_compare_pipelines.py on a CUDA 12.1 host.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

# libsndfile1  -> soundfile
# libportaudio2 -> sounddevice, imported by pi_monitor.py when script 05 loads it
# libgomp1     -> onnxruntime
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        libsndfile1 \
        libportaudio2 \
        libgomp1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# Dependencies are copied first so code edits do not invalidate the pip layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY configs/ ./configs/
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY pi_deploy/ ./pi_deploy/

# data/, models/ and results/ are bind-mounted at runtime, not baked into the image.
CMD ["python", "scripts/02_train.py"]
