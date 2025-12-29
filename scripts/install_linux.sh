#!/bin/bash
# ============================================
# OELALA - ComfyUI NSFW Setup - Linux Install
# ============================================

set -e

echo "============================================="
echo "  OELALA ComfyUI Setup - Linux Installer"
echo "============================================="
echo

# Check if running in correct directory
if [ ! -d "ComfyUI" ]; then
    echo "ERROR: Please run this script from the oelala root directory"
    exit 1
fi

# Check for Python
PYTHON_CMD=""
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "ERROR: Python 3.10+ is required"
    exit 1
fi

echo "[1/5] Setting up Python virtual environment..."
VENV_DIR="${HOME}/venvs/comfyui"
if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_CMD -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

echo "[2/5] Installing PyTorch with CUDA support..."
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "[3/5] Installing ComfyUI dependencies..."
cd ComfyUI
pip install -r requirements.txt

echo "[4/5] Installing custom node dependencies..."
cd custom_nodes

# Install each node's requirements
for node_dir in */; do
    if [ -f "${node_dir}requirements.txt" ]; then
        echo "    Installing ${node_dir} dependencies..."
        pip install -r "${node_dir}requirements.txt" 2>/dev/null || true
    fi
done

# Install additional packages
pip install gguf sentencepiece transformers accelerate
pip install pyogg==0.6.14a1

# Install llama-cpp-python with CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir

cd ../..

echo "[5/5] Creating model directories..."
mkdir -p ComfyUI/models/{checkpoints,clip,clip_vision,diffusion_models,loras,smol,text_encoders,unet,vae,LLM/GGUF,llm/GGUF}

echo
echo "============================================="
echo "  Installation Complete!"
echo "============================================="
echo
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "To start ComfyUI:"
echo "  cd ComfyUI && python main.py --listen 0.0.0.0 --port 8188"
echo
echo "To download models, run the download scripts:"
echo "  ./scripts/download_base_models.sh"
echo "  ./scripts/download_nsfw_models.sh"
echo "  ./scripts/download_wan22_models.sh"
echo "  ./scripts/download_llm_models.sh"
echo
