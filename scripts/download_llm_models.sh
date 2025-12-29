#!/bin/bash
# ============================================
# OELALA - Download LLM/Caption Models
# ============================================

set -e

echo "============================================="
echo "  Downloading LLM/Caption Models (~20GB)"
echo "============================================="
echo

MODELS_DIR="${1:-ComfyUI/models}"

# SmolLM2 for Prompt Generation
echo "[1/4] Downloading SmolLM2-1.7B-Instruct (3.2GB)..."
cd "$MODELS_DIR/smol"
if [ ! -d "SmolLM2-1.7B-Instruct" ]; then
    git lfs install
    git clone https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct
else
    echo "     Already exists, skipping..."
fi
cd -

# JoyCaption for Image Captioning
echo "[2/4] Downloading JoyCaption GGUF (6.2GB)..."
mkdir -p "$MODELS_DIR/LLM/GGUF"
wget -c "https://huggingface.co/Joseph717171/llama-joycaption-beta-one-hf-llava-GGUF/resolve/main/llama-joycaption-beta-one-hf-llava.Q6_K.gguf" \
    -O "$MODELS_DIR/LLM/GGUF/llama-joycaption-beta-one-hf-llava.Q6_K.gguf"

echo "[3/4] Downloading JoyCaption mmproj (838MB)..."
wget -c "https://huggingface.co/Joseph717171/llama-joycaption-beta-one-hf-llava-GGUF/resolve/main/llama-joycaption-beta-one-llava-mmproj-model-f16.gguf" \
    -O "$MODELS_DIR/LLM/GGUF/llama-joycaption-beta-one-llava-mmproj-model-f16.gguf"

# QwenVL for Video Captioning
echo "[4/4] Downloading Qwen2.5-VL NSFW Caption (5.8GB)..."
mkdir -p "$MODELS_DIR/llm/GGUF"
wget -c "https://huggingface.co/bartowski/thesby_Qwen2.5-VL-7B-NSFW-Caption-V3-GGUF/resolve/main/thesby_Qwen2.5-VL-7B-NSFW-Caption-V3-Q6_K.gguf" \
    -O "$MODELS_DIR/llm/GGUF/Qwen2.5-VL-7B-NSFW-Caption-V3-Q6_K.gguf"

echo
echo "============================================="
echo "  LLM/Caption Models Download Complete!"
echo "============================================="
echo
echo "Models installed:"
echo "  - SmolLM2-1.7B-Instruct: AI prompt generation"
echo "  - JoyCaption GGUF: Uncensored image captioning"
echo "  - Qwen2.5-VL NSFW: Video + image captioning"
echo
