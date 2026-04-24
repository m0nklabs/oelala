# ============================================================
# Oelala I2I Worker for RunPod Serverless
# ============================================================
# Lightweight ComfyUI container for Image-to-Image editing (SDXL, Qwen).
# Models are downloaded at runtime from HuggingFace Hub with
# RunPod caching for fast cold starts.
#
# Required models (~28.2 GB total):
#   - qwen_image_edit_2511_fp8mixed.safetensors  (19.1 GB, diffusion_models)
#   - qwen_2.5_vl_7b_fp8_scaled.safetensors      (8.8 GB, text_encoders)
#   - qwen_image_vae.safetensors                  (243 MB, vae)
#
# Optional (downloaded on-demand):
#   - jib_mix_qwen_v6_fp8.safetensors             (19.0 GB, diffusion_models)
#   - Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors (850 MB, loras)
#
# Image size: ~8 GB (no models baked in)
# Target GPU: A6000/A40/L40S/A100 (48 GB+ VRAM)
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
ARG CACHE_DATE=2026-04-14
WORKDIR /comfyui
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git . && \
    pip install --no-cache-dir -r requirements.txt

# ---- Custom Nodes ----
WORKDIR /comfyui/custom_nodes

# KJNodes (ImageResizeKJ, various utility nodes)
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git && \
    pip install --no-cache-dir -r ComfyUI-KJNodes/requirements.txt 2>/dev/null || true

# NOTE: Qwen Image Edit uses native ComfyUI nodes only:
#   UNETLoader, CLIPLoader (type=qwen_image), VAELoader, FluxKontextImageScale,
#   FluxKontextMultiReferenceLatentMethod, TextEncodeQwenImageEditPlus,
#   ModelSamplingAuraFlow, CFGNorm, KSampler, VAEDecode, SaveImage,
#   LoraLoaderModelOnly (for Lightning/custom LoRAs)
# No GGUF nodes needed — fp8mixed is native safetensors
# No VHS needed — image output only (SaveImage), no video
# No MultiGPU/DisTorch2 — single 48 GB+ GPU has enough VRAM
# No SageAttention — enough VRAM headroom on cloud GPUs

# ---- RunPod handler + HF download fallback ----
WORKDIR /comfyui
RUN pip install --no-cache-dir runpod requests httpx huggingface_hub

# Create model directories (models downloaded at runtime)
RUN mkdir -p models/diffusion_models models/text_encoders models/vae models/loras

COPY handler.py /comfyui/handler.py

# ---- Entrypoint ----
CMD ["python", "handler.py"]
