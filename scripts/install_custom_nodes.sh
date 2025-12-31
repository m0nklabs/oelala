#!/bin/bash
# ============================================
# OELALA - Custom Nodes Installation Script
# ============================================
# Installs all required ComfyUI custom nodes for
# multi-GPU, GGUF, video, and NSFW workflows
# Updated: 2025-01-01

set -e

echo "============================================="
echo "  OELALA Custom Nodes Installer v2.0"
echo "============================================="
echo

# Check if running in correct directory
if [ ! -d "ComfyUI" ]; then
    echo "ERROR: Please run this script from the oelala root directory"
    exit 1
fi

cd ComfyUI/custom_nodes

# Helper function to install or update a node
install_node() {
    local name="$1"
    local repo="$2"
    local dir="${3:-$name}"
    
    echo "Installing $name..."
    if [ ! -d "$dir" ]; then
        git clone "https://github.com/$repo.git" "$dir" 2>/dev/null || git clone "https://github.com/$repo.git"
    else
        echo "  Already installed, pulling latest..."
        cd "$dir" && git pull 2>/dev/null || true && cd ..
    fi
}

echo "=== Core Nodes ==="
install_node "ComfyUI-MultiGPU" "pollinations/ComfyUI-MultiGPU"
install_node "ComfyUI-GGUF" "city96/ComfyUI-GGUF"
install_node "ComfyUI-WanVideoWrapper" "kijai/ComfyUI-WanVideoWrapper"
install_node "ComfyUI-VideoHelperSuite" "Kosinkadink/ComfyUI-VideoHelperSuite"
install_node "ComfyUI-KJNodes" "kijai/ComfyUI-KJNodes"

echo "=== Captioning & Prompt Nodes ==="
install_node "ComfyUI-JoyCaption" "MoonHugo/ComfyUI-JoyCaption"
install_node "ComfyUI-QwenVL" "IuvenisSapworker/ComfyUI-QwenVL"
install_node "ComfyUI_QwenImageEdit" "liubin1777/ComfyUI_QwenImageEdit"
install_node "Comfyui-QwenEditUtils" "lrzjason/Comfyui-QwenEditUtils"
install_node "ComfyUI-Florence2" "kijai/ComfyUI-Florence2"
install_node "comfyui-dynamicprompts" "adieyal/comfyui-dynamicprompts"
install_node "comfyui-portrait-master" "florestefano1975/comfyui-portrait-master"

echo "=== UI & Utility Nodes ==="
install_node "ComfyUI-Custom-Scripts" "pythongosssss/ComfyUI-Custom-Scripts"
install_node "rgthree-comfy" "rgthree/rgthree-comfy"
install_node "ComfyUI-Easy-Use" "yolain/ComfyUI-Easy-Use"
install_node "ComfyUI-Crystools" "crystian/ComfyUI-Crystools"
install_node "ComfyUI_tinyterraNodes" "TinyTerra/ComfyUI_tinyterraNodes"
install_node "comfy_mtb" "melMass/comfy_mtb"
install_node "ComfyUI-Inspire-Pack" "ltdrdata/ComfyUI-Inspire-Pack"
install_node "ComfyUI-Impact-Pack" "ltdrdata/ComfyUI-Impact-Pack"

echo "=== Image Processing Nodes ==="
install_node "ComfyUI_LayerStyle" "chflame163/ComfyUI_LayerStyle"
install_node "ComfyUI_LayerStyle_Advance" "chflame163/ComfyUI_LayerStyle_Advance"
install_node "was-node-suite-comfyui" "WASasquatch/was-node-suite-comfyui"
install_node "ComfyUI_essentials" "cubiq/ComfyUI_essentials"
install_node "ComfyUI-ColorCorrection" "yolanother/ComfyUI-ColorCorrection"
install_node "Comfyui-ColorMatchNodes" "elyetis/Comfyui-ColorMatchNodes"
install_node "ComfyUI-EsesImageAdjustments" "quasiblob/ComfyUI-EsesImageAdjustments"
install_node "comfyui-vrgamedevgirl" "vrgamegirl19/comfyui-vrgamedevgirl"
install_node "ComfyUI-RMBG" "yolain/ComfyUI-RMBG"

echo "=== Video & Frame Interpolation ==="
install_node "ComfyUI-Frame-Interpolation" "Fannovel16/ComfyUI-Frame-Interpolation"
install_node "ComfyUI-GIMM-VFI" "kijai/ComfyUI-GIMM-VFI"
install_node "ComfyUI-FramePackWrapper" "kijai/ComfyUI-FramePackWrapper"
install_node "comfyui-dream-video-batches" "alt-key-project/comfyui-dream-video-batches"

echo "=== Audio Nodes ==="
install_node "ComfyUI-MMAudio" "kijai/ComfyUI-MMAudio"
install_node "ComfyUI-MelBandRoFormer" "kijai/ComfyUI-MelBandRoFormer"

echo "=== ControlNet & Face Processing ==="
install_node "comfyui_controlnet_aux" "Fannovel16/comfyui_controlnet_aux"
install_node "ComfyUI-Workarounds" "alisson-anjos/ComfyUI-Workarounds"
install_node "ComfyUI-WarperNodes" "kijai/ComfyUI-WarperNodes"

echo "=== Math & Utility Nodes ==="
install_node "ComfyMath" "evanspearman/ComfyMath"
install_node "Derfuu_ComfyUI_ModdedNodes" "Derfuu/Derfuu_ComfyUI_ModdedNodes"
install_node "sigmas_tools_and_the_golden_scheduler" "Extraltodeus/sigmas_tools_and_the_golden_scheduler"
install_node "wlsh_nodes" "wallish77/wlsh_nodes"

echo "=== Resolution & Cropping ==="
install_node "Comfyui-Resolution-Master" "Jelosus2/Comfyui-Resolution-Master"
install_node "ComfyUI-mxToolkit" "Smirnov75/ComfyUI-mxToolkit"

echo "=== Specialized Nodes ==="
install_node "Comfyui-ergouzi-Nodes" "11dogzi/Comfyui-ergouzi-Nodes"
install_node "ComfyUI_Comfyroll_CustomNodes" "Suzie1/ComfyUI_Comfyroll_CustomNodes"
install_node "ComfyUI-K3NKImageGrab" "K3NKxx/ComfyUI-K3NKImageGrab"
install_node "comfyui-various" "jamesWalker55/comfyui-various"
install_node "ComfyUI-pause" "wywywywy/ComfyUI-pause"
install_node "fcSuite" "fitCorder/fcSuite"
install_node "ComfyUI-APQNodes" "alpertunga-bile/ComfyUI-APQNodes"
install_node "ComfyUI-Multi-Folder-Loader" "AbnormalDistributions/ComfyUI-Multi-Folder-Loader"
install_node "ComfyUI-Ollama-Describer" "TashaSkyUp/ComfyUI-Ollama-Describer"

# Fix fcSuite __init__.py if missing
if [ -d "fcSuite" ] && [ ! -f "fcSuite/__init__.py" ]; then
    echo "Fixing fcSuite package structure..."
    ln -sf fcSuite.py fcSuite/__init__.py
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

# Install additional essential packages
echo
echo "Installing core dependencies..."
pip install gguf sentencepiece transformers accelerate einops timm omegaconf
pip install librosa torchdiffeq asteval webcolors cachetools
pip install cupy-cuda12x rotary-embedding-torch kornia imageio torchcodec evalidate
pip install open_clip_torch ftfy yacs scikit-image

# Check CUDA for llama-cpp-python
echo
echo "Installing llama-cpp-python with CUDA support..."
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir 2>/dev/null || echo "llama-cpp-python install failed (non-critical)"

cd ../..

echo
echo "============================================="
echo "  Custom Nodes Installation Complete!"
echo "============================================="
echo
echo "Total nodes installed: $(ls ComfyUI/custom_nodes/ | wc -l)"
echo
echo "Key node categories:"
echo "  - Core: MultiGPU, GGUF, WanVideoWrapper, VideoHelperSuite"
echo "  - Captioning: JoyCaption, QwenVL, Florence2"
echo "  - UI: rgthree, Easy-Use, Crystools, tinyterraNodes"
echo "  - Image: LayerStyle, WAS Suite, Essentials, ColorCorrection"
echo "  - Video: Frame-Interpolation, GIMM-VFI, FramePackWrapper"
echo "  - Audio: MMAudio, MelBandRoFormer"
echo "  - Math: ComfyMath, Derfuu nodes, sigmas_tools"
echo
echo "Please restart ComfyUI to load the new nodes."
echo
