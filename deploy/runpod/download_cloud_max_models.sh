#!/usr/bin/env bash
# ============================================================
# Download Cloud Max Wan 2.2 fp8_scaled models to RunPod
# Network Volume (EU-RO-1)
# ============================================================
# Run this inside a RunPod pod with the Network Volume mounted
# at /runpod-volume/ to pre-populate the model cache.
#
# The handler.py download_models() does this automatically on
# first boot, but pre-populating avoids the cold start delay.
#
# Usage:
#   bash download_cloud_max_models.sh [models_dir]
#
# Required: ~46GB free disk space
# Sources:
#   Comfy-Org/Wan_2.2_ComfyUI_Repackaged (diffusion, text encoder)
#   Comfy-Org/Wan_2.1_ComfyUI_repackaged (VAE, CLIP vision)
# ============================================================

set -euo pipefail

MODELS_DIR="${1:-/runpod-volume/models}"
HF_REPO_22="Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
HF_REPO_21="Comfy-Org/Wan_2.1_ComfyUI_repackaged"

echo "============================================================"
echo "  Cloud Max Model Downloader (Wan 2.2 fp8_scaled)"
echo "  Target: ${MODELS_DIR}"
echo "============================================================"

# Install huggingface_hub if not present
pip install --quiet huggingface_hub[cli] 2>/dev/null || true

# Create directory structure
mkdir -p "${MODELS_DIR}/unet"
mkdir -p "${MODELS_DIR}/clip"
mkdir -p "${MODELS_DIR}/vae"
mkdir -p "${MODELS_DIR}/clip_vision"
mkdir -p "${MODELS_DIR}/loras"

download_model() {
    local repo="$1" hf_path="$2" dest="$3" desc="$4"
    if [ -f "${dest}" ]; then
        echo "  ✅ $(basename "${dest}") (already exists)"
        return 0
    fi
    echo "  ⬇️  ${desc}"
    huggingface-cli download "${repo}" "${hf_path}" \
        --local-dir "${MODELS_DIR}/_tmp_download" \
        --local-dir-use-symlinks=False
    mv "${MODELS_DIR}/_tmp_download/${hf_path}" "${dest}"
    rm -rf "${MODELS_DIR}/_tmp_download"
    echo "  ✅ Done"
}

echo ""
echo "📦 Downloading Wan 2.2 diffusion models (fp8_scaled, ~14.3 GB each)..."

download_model "${HF_REPO_22}" \
    "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" \
    "${MODELS_DIR}/unet/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" \
    "Wan 2.2 I2V high noise 14B fp8_scaled (14.3 GB)"

download_model "${HF_REPO_22}" \
    "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" \
    "${MODELS_DIR}/unet/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" \
    "Wan 2.2 I2V low noise 14B fp8_scaled (14.3 GB)"

download_model "${HF_REPO_22}" \
    "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors" \
    "${MODELS_DIR}/unet/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors" \
    "Wan 2.2 T2V high noise 14B fp8_scaled (14.3 GB)"

download_model "${HF_REPO_22}" \
    "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors" \
    "${MODELS_DIR}/unet/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors" \
    "Wan 2.2 T2V low noise 14B fp8_scaled (14.3 GB)"

echo ""
echo "📦 Downloading text encoder..."

download_model "${HF_REPO_22}" \
    "split_files/text_encoders/umt5_xxl_fp16.safetensors" \
    "${MODELS_DIR}/clip/umt5_xxl_fp16.safetensors" \
    "UMT5-XXL fp16 text encoder (11.4 GB)"

echo ""
echo "📦 Downloading VAE (Wan 2.1 — required for 14B models)..."

download_model "${HF_REPO_21}" \
    "split_files/vae/wan_2.1_vae.safetensors" \
    "${MODELS_DIR}/vae/wan_2.1_vae.safetensors" \
    "Wan 2.1 VAE (254 MB)"

echo ""
echo "📦 Downloading CLIP vision encoder..."

download_model "${HF_REPO_21}" \
    "split_files/clip_vision/clip_vision_h.safetensors" \
    "${MODELS_DIR}/clip_vision/clip_vision_h.safetensors" \
    "CLIP Vision H — I2V conditioning (1.26 GB)"

echo ""
echo "============================================================"
echo "  ✅ All Cloud Max Wan 2.2 models downloaded!"
echo ""
echo "  Disk usage:"
du -sh "${MODELS_DIR}/unet/" "${MODELS_DIR}/clip/" \
       "${MODELS_DIR}/vae/" "${MODELS_DIR}/clip_vision/" 2>/dev/null || true
echo ""
echo "  Total:"
du -sh "${MODELS_DIR}/" 2>/dev/null || true
echo "============================================================"
