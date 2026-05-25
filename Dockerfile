# syntax=docker/dockerfile:1.7
#
# asr-streaming — Qwen3-ASR-1.7B HTTP server (vLLM backend) for the
# Vocence /studio/ops fleet manager. Runs on NVIDIA RTX 4090 (24 GB).
#
# Build:
#   docker build -t docker.io/<ns>/asr-streaming:latest .
# Run (on rented box):
#   docker run -d --gpus all --restart=unless-stopped -p 8114:8114 \
#     -e STT_API_KEY=<key> \
#     -v hf_cache:/cache/hf \
#     docker.io/<ns>/asr-streaming:latest

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/hf \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    TRANSFORMERS_CACHE=/cache/hf/transformers \
    HOST=0.0.0.0 \
    PORT=8114 \
    QWEN3_ASR_MODEL=/cache/hf/Qwen3-ASR-1.7B

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        git curl ca-certificates \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (slow layer, cache hit across iterations).
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install --upgrade hf_transfer "huggingface_hub[cli]" \
    && pip install -r requirements.txt

COPY main.py ./

# Build-time import smoke test — verifies main.py and its module-level
# deps (fastapi/uvicorn/torch) import cleanly. Cheap (no GPU, no vLLM
# warmup), catches ABI mismatches and missing modules BEFORE the image
# reaches Docker Hub. qwen_asr is imported lazily inside a function in
# main.py, so it's deliberately NOT part of this probe (it'd need a GPU).
RUN python3 -c "import torch; print('torch', torch.__version__)" \
    && python3 -c "import main; print('main import OK')"

# The model lives in the /cache/hf volume so it survives container
# recreation. First boot downloads ~4.4 GB; subsequent boots are instant.
VOLUME /cache/hf

EXPOSE 8114

# 180 s start-period covers: huggingface-cli download (first boot only)
# OR vLLM model load + CUDA graph capture (~30-60 s on warm cache).
HEALTHCHECK --interval=15s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -fsS http://127.0.0.1:${PORT:-8114}/health || exit 1

# On first run, fetch the model if missing, then start the server.
CMD bash -c "if [ ! -f \"$QWEN3_ASR_MODEL/config.json\" ]; then \
        echo '>>> Downloading Qwen3-ASR-1.7B weights to '$QWEN3_ASR_MODEL; \
        huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir \"$QWEN3_ASR_MODEL\"; \
    fi && python3 main.py"
