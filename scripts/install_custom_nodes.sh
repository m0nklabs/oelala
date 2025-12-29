#!/bin/bash
# ============================================
# OELALA - Custom Nodes Installation Script
# ============================================
# Installs all required ComfyUI custom nodes for
# multi-GPU, GGUF, video, and NSFW workflows

set -e

echo "============================================="
echo "  OELALA Custom Nodes Installer"
echo "============================================="
echo

# Check if running in correct directory
if [ ! -d "ComfyUI" ]; then
    echo "ERROR: Please run this script from the oelala root directory"
    exit 1
fi

cd ComfyUI/custom_nodes

echo "[1/11] Installing ComfyUI-MultiGPU (multi-GPU inference)..."
if [ ! -d "ComfyUI-MultiGPU" ]; then
    git clone https://github.com/pollinations/ComfyUI-MultiGPU.git
else
    echo "  Already installed, pulling latest..."
    cd ComfyUI-MultiGPU && git pull && cd ..
fi

echo "[2/11] Installing ComfyUI-GGUF (GGUF model support)..."
if [ ! -d "ComfyUI-GGUF" ]; then
    git clone https://github.com/city96/ComfyUI-GGUF.git
else
    echo "  Already installed, pulling latest..."
    cd ComfyUI-GGUF && git pull && cd ..
fi

echo "[3/11] Installing ComfyUI-WanVideoWrapper (WAN 2.1/2.2 workflows)..."
if [ ! -d "ComfyUI-WanVideoWrapper" ]; then
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
else
    echo "  Already installed, pulling latest..."
    cd ComfyUI-WanVideoWrapper && git pull && cd ..
fi

echo "[4/11] Installing ComfyUI-VideoHelperSuite (video utilities)..."
if [ ! -d "ComfyUI-VideoHelperSuite" ]; then
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
else
    echo "  Already installed, pulling latest..."
    cd ComfyUI-VideoHelperSuite && git pull && cd ..
fi

echo "[5/11] Installing ComfyUI-KJNodes (utility nodes)..."
if [ ! -d "ComfyUI-KJNodes" ]; then
    git clone https://github.com/kijai/ComfyUI-KJNodes.git
else
    echo "  Already installed, pulling latest..."
    cd ComfyUI-KJNodes && git pull && cd ..
fi

echo "[6/11] Installing ComfyUI-JoyCaption (image captioning)..."
if [ ! -d "ComfyUI-JoyCaption" ]; then
    git clone https://github.com/MoonHugo/ComfyUI-JoyCaption.git
else
    echo "  Already installed, pulling latest..."
    cd ComfyUI-JoyCaption && git pull && cd ..
fi

echo "[7/11] Installing ComfyUI-QwenVL (video captioning)..."
if [ ! -d "ComfyUI-QwenVL" ]; then
    git clone https://github.com/IuvenisSapworker/ComfyUI-QwenVL.git
else
    echo "  Already installed, pulling latest..."
    cd ComfyUI-QwenVL && git pull && cd ..
fi

echo "[8/11] Installing comfyui-dynamicprompts (prompt variations)..."
if [ ! -d "comfyui-dynamicprompts" ]; then
    git clone https://github.com/adieyal/comfyui-dynamicprompts.git
else
    echo "  Already installed, pulling latest..."
    cd comfyui-dynamicprompts && git pull && cd ..
fi

echo "[9/11] Installing comfyui-portrait-master (portrait generation)..."
if [ ! -d "comfyui-portrait-master" ]; then
    git clone https://github.com/florestefano1975/comfyui-portrait-master.git
else
    echo "  Already installed, pulling latest..."
    cd comfyui-portrait-master && git pull && cd ..
fi

echo "[10/11] Installing ComfyUI-Custom-Scripts (various utilities)..."
if [ ! -d "ComfyUI-Custom-Scripts" ]; then
    git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git
else
    echo "  Already installed, pulling latest..."
    cd ComfyUI-Custom-Scripts && git pull && cd ..
fi

echo "[11/11] Installing ComfyUI_LayerStyle_Advance (layer styling)..."
if [ ! -d "ComfyUI_LayerStyle_Advance" ]; then
    git clone https://github.com/chflame163/ComfyUI_LayerStyle_Advance.git
else
    echo "  Already installed, pulling latest..."
    cd ComfyUI_LayerStyle_Advance && git pull && cd ..
fi

# Install requirements for each node
echo
echo "Installing Python dependencies for all nodes..."
for node_dir in */; do
    if [ -f "${node_dir}requirements.txt" ]; then
        echo "  Installing ${node_dir} dependencies..."
        pip install -r "${node_dir}requirements.txt" 2>/dev/null || true
    fi
done

# Install additional packages for GGUF/LLM support
echo
echo "Installing GGUF/LLM dependencies..."
pip install gguf sentencepiece transformers accelerate

# Check CUDA for llama-cpp-python
echo
echo "Installing llama-cpp-python with CUDA support..."
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

cd ../..

echo
echo "============================================="
echo "  Custom Nodes Installation Complete!"
echo "============================================="
echo
echo "Installed nodes:"
echo "  - ComfyUI-MultiGPU     : Multi-GPU model offloading"
echo "  - ComfyUI-GGUF         : GGUF quantized models"
echo "  - ComfyUI-WanVideoWrapper : WAN 2.1/2.2 video generation"
echo "  - ComfyUI-VideoHelperSuite : Video utilities"
echo "  - ComfyUI-KJNodes      : Utility nodes (Kijai)"
echo "  - ComfyUI-JoyCaption   : Image captioning (I2T)"
echo "  - ComfyUI-QwenVL       : Video captioning (V2T)"
echo "  - comfyui-dynamicprompts : Dynamic prompt variations"
echo "  - comfyui-portrait-master : Portrait generation"
echo "  - ComfyUI-Custom-Scripts : Various utilities"
echo "  - ComfyUI_LayerStyle_Advance : Layer styling"
echo
echo "Restart ComfyUI to load the new nodes!"
echo
