#!/bin/bash
# ============================================
# OELALA - Download Base Models (Required)
# ============================================

set -e

echo "============================================="
echo "  Downloading Base Models (~30GB)"
echo "============================================="
echo

MODELS_DIR="${1:-ComfyUI/models}"

# CLIP Text Encoders
echo "[1/6] Downloading CLIP encoders..."
wget -c "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors" \
    -O "$MODELS_DIR/clip/clip_l.safetensors"

wget -c "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors" \
    -O "$MODELS_DIR/clip/t5xxl_fp8_e4m3fn.safetensors"

echo "[2/6] Downloading UMT5-XXL for WAN 2.2..."
wget -c "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5-xxl-enc-bf16.safetensors" \
    -O "$MODELS_DIR/clip/umt5-xxl-enc-bf16.safetensors"

# CLIP Vision
echo "[3/6] Downloading CLIP Vision..."
wget -c "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" \
    -O "$MODELS_DIR/clip_vision/clip-vit-large.safetensors"

# VAE
echo "[4/6] Downloading VAE models..."
wget -c "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors" \
    -O "$MODELS_DIR/vae/ae.safetensors"

wget -c "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors" \
    -O "$MODELS_DIR/vae/wan_2.1_vae.safetensors"

# Text Encoders for WAN
echo "[5/6] Downloading WAN Text Encoders..."
wget -c "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors" \
    -O "$MODELS_DIR/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

wget -c "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors" \
    -O "$MODELS_DIR/text_encoders/qwen_3_4b.safetensors"

# Flux Dev Checkpoint
echo "[6/6] Downloading Flux Dev FP8..."
wget -c "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors" \
    -O "$MODELS_DIR/checkpoints/flux1-dev-fp8.safetensors"

echo
echo "============================================="
echo "  Base Models Download Complete!"
echo "============================================="
echo
