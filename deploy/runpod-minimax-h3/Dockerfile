# ============================================================
# Oelala MiniMax-H3 Worker for RunPod Serverless
# ============================================================
# ComfyUI container for MiniMax-H3 video+audio generation (t2v / i2v).
# Models are downloaded at runtime from HuggingFace Hub with
# RunPod caching for fast cold starts.
#
# Required models (~42.5 GB total):
#   - minimax_h3_fl2va_pruned_int8_convrot.safetensors   (20.97 GB, diffusion_models, t2v + i2v keyframes)
#   - qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors       (15.69 GB, text_encoders, no Blackwell needed)
#   - minimax_h3_video_vae_fp16.safetensors              ( 5.21 GB, vae)
#   - minimax_h3_audio_vae_fp32.safetensors              ( 0.61 GB, vae)
#
# Image size: ~8 GB (no models baked in)
# Target GPU: A100/H100/B200/GB200 (80 GB+ VRAM); the int8/nvfp4 quantized
#   files are also small enough to experiment on 48 GB tiers.
# Container disk: 100 GB
#
# ComfyUI: cloned from the official Comfy-Org/ComfyUI master branch — the
# MiniMax-H3 core nodes (MiniMaxH3ImageToVideo, VAEDecodeAudio, ...) landed
# there with the H3 release (July 2026). CACHE_DATE busts the clone when the
# checkout needs refreshing.
#
# Deploy:  ./deploy.sh  (builds, tags, pushes, AND updates RunPod template)
#
# ⚠️  NEVER push :latest manually without updating the RunPod template!
#     RunPod uses explicit dated tags. Use deploy.sh to avoid this mistake.
# ============================================================

FROM runpod/pytorch:1.0.3-cu1300-torch291-ubuntu2404

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git git-lfs ffmpeg libgl1 libglib2.0-0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# ---- ComfyUI (official master — required for MiniMax-H3 core nodes) ----
ARG CACHE_DATE=2026-08-16
WORKDIR /comfyui
RUN git clone --depth 1 https://github.com/Comfy-Org/ComfyUI.git . && \
    pip install --no-cache-dir -r requirements.txt

# ---- Custom Nodes ----
WORKDIR /comfyui/custom_nodes

# Video Helper Suite (VHS - video output with h264 mp4 + audio muxing)
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    pip install --no-cache-dir -r ComfyUI-VideoHelperSuite/requirements.txt

# NOTE: MiniMax-H3 nodes are in ComfyUI core (comfy_extras/nodes_minimax_h3.py),
#       no extra custom nodes required for t2v / i2v generation.

# ---- RunPod handler + HF download fallback ----
WORKDIR /comfyui
RUN pip install --no-cache-dir runpod requests httpx huggingface_hub

# Create model directories (models downloaded at runtime)
RUN mkdir -p models/diffusion_models models/text_encoders models/vae models/loras models/checkpoints

COPY handler.py /comfyui/handler.py

# ---- Entrypoint ----
CMD ["python", "handler.py"]
