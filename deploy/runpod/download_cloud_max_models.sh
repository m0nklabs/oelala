#!/usr/bin/env bash
# ============================================================
# Download Cloud Max bf16 models to RunPod Network Volume
# ============================================================
# Run this inside a RunPod pod with the Network Volume mounted
# at /runpod-volume/ to download all required models.
#
# Usage:
#   bash download_cloud_max_models.sh
#
# Required: ~46GB free disk space
# Source: Comfy-Org/Wan_2.1_ComfyUI_repackaged (HuggingFace)
# ============================================================

set -euo pipefail

MODELS_DIR="${1:-/runpod-volume/models}"
HF_REPO="Comfy-Org/Wan_2.1_ComfyUI_repackaged"

echo "============================================================"
echo "  Cloud Max Model Downloader"
echo "  Target: ${MODELS_DIR}"
echo "============================================================"

# Install huggingface_hub if not present
pip install --quiet huggingface_hub[cli] 2>/dev/null || true

# Create directory structure
mkdir -p "${MODELS_DIR}/diffusion_models"
mkdir -p "${MODELS_DIR}/text_encoders"
mkdir -p "${MODELS_DIR}/vae"
mkdir -p "${MODELS_DIR}/clip_vision"
mkdir -p "${MODELS_DIR}/loras"

echo ""
echo "📦 Downloading diffusion models (bf16)..."

# I2V 720p bf16 — 32.8 GB
if [ ! -f "${MODELS_DIR}/diffusion_models/wan2.1_i2v_720p_14B_bf16.safetensors" ]; then
    echo "  ⬇️  wan2.1_i2v_720p_14B_bf16.safetensors (32.8 GB)"
    huggingface-cli download "${HF_REPO}" \
        "split_files/diffusion_models/wan2.1_i2v_720p_14B_bf16.safetensors" \
        --local-dir "${MODELS_DIR}/_tmp_download" \
        --local-dir-use-symlinks=False
    mv "${MODELS_DIR}/_tmp_download/split_files/diffusion_models/wan2.1_i2v_720p_14B_bf16.safetensors" \
       "${MODELS_DIR}/diffusion_models/"
    echo "  ✅ Done"
else
    echo "  ✅ wan2.1_i2v_720p_14B_bf16.safetensors (already exists)"
fi

# T2V bf16 — 28.6 GB
if [ ! -f "${MODELS_DIR}/diffusion_models/wan2.1_t2v_14B_bf16.safetensors" ]; then
    echo "  ⬇️  wan2.1_t2v_14B_bf16.safetensors (28.6 GB)"
    huggingface-cli download "${HF_REPO}" \
        "split_files/diffusion_models/wan2.1_t2v_14B_bf16.safetensors" \
        --local-dir "${MODELS_DIR}/_tmp_download" \
        --local-dir-use-symlinks=False
    mv "${MODELS_DIR}/_tmp_download/split_files/diffusion_models/wan2.1_t2v_14B_bf16.safetensors" \
       "${MODELS_DIR}/diffusion_models/"
    echo "  ✅ Done"
else
    echo "  ✅ wan2.1_t2v_14B_bf16.safetensors (already exists)"
fi

echo ""
echo "📦 Downloading text encoder..."

# T5-XXL fp16 — 11.4 GB
if [ ! -f "${MODELS_DIR}/text_encoders/umt5_xxl_fp16.safetensors" ]; then
    echo "  ⬇️  umt5_xxl_fp16.safetensors (11.4 GB)"
    huggingface-cli download "${HF_REPO}" \
        "split_files/text_encoders/umt5_xxl_fp16.safetensors" \
        --local-dir "${MODELS_DIR}/_tmp_download" \
        --local-dir-use-symlinks=False
    mv "${MODELS_DIR}/_tmp_download/split_files/text_encoders/umt5_xxl_fp16.safetensors" \
       "${MODELS_DIR}/text_encoders/"
    echo "  ✅ Done"
else
    echo "  ✅ umt5_xxl_fp16.safetensors (already exists)"
fi

echo ""
echo "📦 Downloading VAE..."

# VAE — 254 MB
if [ ! -f "${MODELS_DIR}/vae/wan_2.1_vae.safetensors" ]; then
    echo "  ⬇️  wan_2.1_vae.safetensors (254 MB)"
    huggingface-cli download "${HF_REPO}" \
        "split_files/vae/wan_2.1_vae.safetensors" \
        --local-dir "${MODELS_DIR}/_tmp_download" \
        --local-dir-use-symlinks=False
    mv "${MODELS_DIR}/_tmp_download/split_files/vae/wan_2.1_vae.safetensors" \
       "${MODELS_DIR}/vae/"
    echo "  ✅ Done"
else
    echo "  ✅ wan_2.1_vae.safetensors (already exists)"
fi

echo ""
echo "📦 Downloading CLIP vision encoder..."

# CLIP Vision H — 1.26 GB (for I2V)
if [ ! -f "${MODELS_DIR}/clip_vision/clip_vision_h.safetensors" ]; then
    echo "  ⬇️  clip_vision_h.safetensors (1.26 GB)"
    huggingface-cli download "${HF_REPO}" \
        "split_files/clip_vision/clip_vision_h.safetensors" \
        --local-dir "${MODELS_DIR}/_tmp_download" \
        --local-dir-use-symlinks=False
    mv "${MODELS_DIR}/_tmp_download/split_files/clip_vision/clip_vision_h.safetensors" \
       "${MODELS_DIR}/clip_vision/"
    echo "  ✅ Done"
else
    echo "  ✅ clip_vision_h.safetensors (already exists)"
fi

# Clean up temp download dir
rm -rf "${MODELS_DIR}/_tmp_download"

echo ""
echo "============================================================"
echo "  ✅ All Cloud Max models downloaded!"
echo ""
echo "  Disk usage:"
du -sh "${MODELS_DIR}/diffusion_models/" "${MODELS_DIR}/text_encoders/" \
       "${MODELS_DIR}/vae/" "${MODELS_DIR}/clip_vision/" 2>/dev/null || true
echo ""
echo "  Total:"
du -sh "${MODELS_DIR}/" 2>/dev/null || true
echo ""
echo "  Next steps:"
echo "  1. Copy your LoRAs to ${MODELS_DIR}/loras/"
echo "  2. Build & push Docker image"
echo "  3. Create RunPod endpoint with this Network Volume"
echo "============================================================"
