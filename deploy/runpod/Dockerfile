# ============================================================
# Oelala ComfyUI Worker for RunPod Serverless
# ============================================================
# Lightweight ComfyUI container for Cloud Max Wan 2.2 video gen.
# Models loaded from RunPod Network Volume (~46GB) at startup.
# Fallback: downloads from HuggingFace if no volume mounted.
# Uses fp8_scaled precision (near-lossless, fits 48GB GPUs).
#
# Image size: ~8GB (no models baked in)
# Target GPU: A6000/A40/L40S (48GB+ VRAM)
#
# Build:   docker build -t ghcr.io/m0nklabs/oelala-comfyui-worker -f Dockerfile .
# Push:    docker push ghcr.io/m0nklabs/oelala-comfyui-worker:latest
# ============================================================

FROM runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- ComfyUI ----
# ARG CACHE_DATE forces re-clone when changed; update to bust cache for Wan 2.2
ARG CACHE_DATE=2026-03-04
WORKDIR /comfyui
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git . && \
    pip install --no-cache-dir -r requirements.txt

# ---- Custom Nodes ----
WORKDIR /comfyui/custom_nodes

# Video Helper Suite (VHS - video output with h264 mp4)
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    pip install --no-cache-dir -r ComfyUI-VideoHelperSuite/requirements.txt

# KJNodes (ImageResizeKJv2 for input image resize)
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git && \
    pip install --no-cache-dir -r ComfyUI-KJNodes/requirements.txt 2>/dev/null || true

# NOTE: Cloud Max uses native UNETLoader (fp8_scaled) — no GGUF node needed
# NOTE: No MultiGPU/DisTorch2/BlockSwap — single 48GB+ GPU
# NOTE: No SageAttention — enough VRAM on cloud GPUs

# ---- RunPod handler + HF download fallback ----
WORKDIR /comfyui
RUN pip install --no-cache-dir runpod requests httpx huggingface_hub

# Create model directories (will be populated from Network Volume at startup)
RUN mkdir -p models/unet models/clip models/vae models/clip_vision models/loras

COPY handler.py /comfyui/handler.py

# ---- Entrypoint ----
CMD ["python", "handler.py"]
