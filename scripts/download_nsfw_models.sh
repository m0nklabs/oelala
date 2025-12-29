#!/bin/bash
# ============================================
# OELALA - Download NSFW Models
# ============================================

set -e

echo "============================================="
echo "  Downloading NSFW Models (~65GB)"
echo "============================================="
echo
echo "WARNING: These models contain NSFW content"
echo

MODELS_DIR="${1:-ComfyUI/models}"

# NSFW Checkpoints
echo "[1/5] Downloading CyberRealistic Pony V14.1 FP16 (6.5GB)..."
wget -c "https://huggingface.co/cyberdelia/CyberRealistic_Pony/resolve/main/CyberRealistic_Pony_v14.1_FP16.safetensors" \
    -O "$MODELS_DIR/checkpoints/CyberRealistic_Pony_v14.1_FP16.safetensors"

echo "[2/5] Downloading Realistic Vision V5.1 (4GB)..."
wget -c "https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1_fp16-no-ema.safetensors" \
    -O "$MODELS_DIR/checkpoints/Realistic_Vision_V5.1.safetensors"

echo "[3/5] Downloading ReaPony V9.0 (6.5GB)..."
wget -c "https://civitai.com/api/download/models/1119970" \
    -O "$MODELS_DIR/checkpoints/reapony_v90.safetensors"

# NSFW Flux Diffusion Models
echo "[4/5] Downloading FluxedUp NSFW 5.1 FP8 (11GB)..."
wget -c "https://civitai.com/api/download/models/1203915" \
    -O "$MODELS_DIR/diffusion_models/fluxedUpFluxNSFW_51FP8.safetensors"

echo "[5/5] Downloading Persephone NSFW 1.1 FP8 (11GB)..."
wget -c "https://civitai.com/api/download/models/1033850" \
    -O "$MODELS_DIR/diffusion_models/persephoneFluxNSFWSFW_11FP8.safetensors"

# NSFW WAN 2.2 Enhanced Models
echo
echo "[OPTIONAL] Downloading WAN 2.2 Enhanced NSFW Models (24GB)..."
wget -c "https://huggingface.co/MonkeyWitATool/wan22EnhancedNSFW_V2/resolve/main/wan22EnhancedNSFW_V2_Q6K_HIGH.gguf" \
    -O "$MODELS_DIR/unet/wan22EnhancedNSFW_V2_Q6K_HIGH.gguf"

wget -c "https://huggingface.co/MonkeyWitATool/wan22EnhancedNSFW_V2/resolve/main/wan22EnhancedNSFW_V2_Q6K_LOW.gguf" \
    -O "$MODELS_DIR/unet/wan22EnhancedNSFW_V2_Q6K_LOW.gguf"

echo
echo "============================================="
echo "  NSFW Models Download Complete!"
echo "============================================="
echo
