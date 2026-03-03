# ============================================================
# Oelala ComfyUI Worker for RunPod Serverless
# ============================================================
# Builds a ComfyUI container with all custom nodes needed for
# Wan2.2 I2V/T2V video generation.
#
# Models are loaded from a RunPod Network Volume (not baked in)
# to keep image size manageable (~8GB vs ~30GB+).
#
# Build:   docker build -t oelala-comfyui-worker -f Dockerfile .
# Push:    docker push <registry>/oelala-comfyui-worker:latest
# ============================================================

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y \
    git git-lfs ffmpeg libgl1-mesa-glx libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- ComfyUI ----
WORKDIR /comfyui
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git . && \
    pip install --no-cache-dir -r requirements.txt

# ---- Custom Nodes (only what we need for video gen) ----
WORKDIR /comfyui/custom_nodes

# GGUF support (quantized models)
RUN git clone --depth 1 https://github.com/city96/ComfyUI-GGUF.git && \
    pip install --no-cache-dir -r ComfyUI-GGUF/requirements.txt

# Video Helper Suite (VHS - video output)
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    pip install --no-cache-dir -r ComfyUI-VideoHelperSuite/requirements.txt

# KJNodes (ImageResizeKJv2 for upscale target)
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git && \
    pip install --no-cache-dir -r ComfyUI-KJNodes/requirements.txt 2>/dev/null || true

# SageAttention (VRAM optimization)
RUN pip install --no-cache-dir sageattention 2>/dev/null || true

# Florence2 auto-captioning
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-Florence2.git && \
    pip install --no-cache-dir -r ComfyUI-Florence2/requirements.txt 2>/dev/null || true

# Frame interpolation (RIFE)
RUN git clone --depth 1 https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git comfyui-frame-interpolation && \
    pip install --no-cache-dir -r comfyui-frame-interpolation/requirements.txt 2>/dev/null || true

# NOTE: No MultiGPU/DisTorch2/BlockSwap needed — RunPod = single GPU with 48-80GB VRAM
# NOTE: No GGUF-VAE needed — RunPod has enough VRAM for fp16 VAE

# ---- RunPod handler ----
WORKDIR /comfyui
RUN pip install --no-cache-dir runpod requests

COPY handler.py /comfyui/handler.py

# ---- Network Volume mount point for models ----
# Models will be symlinked from /runpod-volume/models/ at startup
# This keeps the Docker image small

# ---- Entrypoint ----
CMD ["python", "handler.py"]
