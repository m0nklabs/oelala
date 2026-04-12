# ============================================================
# Oelala LTX-2.3 Worker for RunPod Serverless
# ============================================================
# Lightweight ComfyUI container for LTX-2.3 22B video generation.
# Models are downloaded at runtime from HuggingFace Hub with
# RunPod caching for fast cold starts.
#
# Required models (~59 GB total):
#   - ltx-2.3-22b-distilled.safetensors   (46.1 GB, checkpoints)
#   - gemma_3_12B_it_fp8_scaled.safetensors (13.2 GB, text_encoders)
#
# Image size: ~8 GB (no models baked in)
# Target GPU: A100/H100/B200/GB200 (80 GB+ VRAM)
# Container disk: 100 GB
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

# ---- ComfyUI ----
ARG CACHE_DATE=2026-04-12
WORKDIR /comfyui
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git . && \
    pip install --no-cache-dir -r requirements.txt

# ---- Custom Nodes ----
WORKDIR /comfyui/custom_nodes

# Video Helper Suite (VHS - video output with h264 mp4)
RUN git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    pip install --no-cache-dir -r ComfyUI-VideoHelperSuite/requirements.txt

# LTXVideo helpers (LTXVImgToVideoConditionOnly, enhanced LTX nodes)
RUN git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git && \
    pip install --no-cache-dir -r ComfyUI-LTXVideo/requirements.txt 2>/dev/null || true

# KJNodes (ImageResizeKJv2 for input image resize in I2V)
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git && \
    pip install --no-cache-dir -r ComfyUI-KJNodes/requirements.txt 2>/dev/null || true

# NOTE: No GGUF nodes — LTX-2.3 22B uses bf16 checkpoint (no quantized version yet)
# NOTE: No MultiGPU/DisTorch2 — single 80 GB+ GPU has enough VRAM
# NOTE: No SageAttention — enough VRAM headroom on 80 GB GPUs

# ---- RunPod handler + HF download fallback ----
WORKDIR /comfyui
RUN pip install --no-cache-dir runpod requests httpx huggingface_hub

# Create model directories (models downloaded at runtime)
RUN mkdir -p models/checkpoints models/text_encoders models/vae models/loras

COPY handler.py /comfyui/handler.py

# ---- Entrypoint ----
CMD ["python", "handler.py"]
