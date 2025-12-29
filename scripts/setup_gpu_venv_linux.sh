#!/bin/bash
# ============================================
# OELALA - Linux GPU Virtual Environment Setup
# ============================================
# Creates a proper Python venv with CUDA PyTorch
#
# REQUIREMENTS:
#   - Python 3.10 or 3.11
#   - NVIDIA GPU with CUDA support
#   - NVIDIA drivers installed
#   - Git installed
#   - ~20GB free disk space
#
# Run from the oelala root directory!

set -e

echo "============================================="
echo "  OELALA - Linux GPU VENV Setup"
echo "============================================="
echo

# Check if running in correct directory
if [ ! -d "ComfyUI" ]; then
    echo "ERROR: Please run this script from the oelala root directory"
    echo
    echo "Expected structure:"
    echo "  oelala/"
    echo "    ComfyUI/"
    echo "    scripts/"
    echo "    workflows/"
    exit 1
fi

# ============================================
# Step 1: Find Python
# ============================================
echo "[1/8] Checking Python installation..."

PYTHON_CMD=""
if command -v python3.11 &> /dev/null; then
    PYTHON_CMD="python3.11"
elif command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3 &> /dev/null; then
    PYVER=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
    if [[ "$PYVER" =~ ^3\.1[01] ]]; then
        PYTHON_CMD="python3"
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    echo "ERROR: Python 3.10 or 3.11 is required!"
    echo
    echo "Install with:"
    echo "  Ubuntu/Debian: sudo apt install python3.11 python3.11-venv"
    echo "  Fedora: sudo dnf install python3.11"
    echo "  Arch: sudo pacman -S python"
    exit 1
fi

echo "      Using: $PYTHON_CMD"
$PYTHON_CMD --version

# ============================================
# Step 2: Check CUDA
# ============================================
echo
echo "[2/8] Checking NVIDIA CUDA..."

if command -v nvcc &> /dev/null; then
    nvcc --version | grep release
else
    echo "      WARNING: nvcc not found in PATH"
    echo "      CUDA Toolkit may not be installed"
fi

if command -v nvidia-smi &> /dev/null; then
    echo "      NVIDIA driver detected"
    nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | while read gpu; do
        echo "      GPU: $gpu"
    done
else
    echo "      WARNING: nvidia-smi not found"
    echo "      Make sure NVIDIA drivers are installed!"
fi

# ============================================
# Step 3: Create Virtual Environment
# ============================================
echo
echo "[3/8] Creating Python virtual environment..."

VENV_DIR="venv"

if [ -d "$VENV_DIR" ]; then
    read -p "      venv already exists. Delete and recreate? [y/N] " RECREATE
    if [[ "$RECREATE" =~ ^[Yy]$ ]]; then
        echo "      Removing old venv..."
        rm -rf "$VENV_DIR"
    else
        echo "      Keeping existing venv..."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    $PYTHON_CMD -m venv "$VENV_DIR"
    echo "      Created: $VENV_DIR"
fi

echo "      Activating venv..."
source "$VENV_DIR/bin/activate"

# ============================================
# Step 4: Upgrade pip and install build tools
# ============================================
echo
echo "[4/8] Upgrading pip and installing build tools..."

pip install --upgrade pip wheel setuptools

# ============================================
# Step 5: Install PyTorch with CUDA
# ============================================
echo
echo "[5/8] Installing PyTorch with CUDA 12.4 support..."
echo "      This may take 5-10 minutes..."
echo

# PyTorch 2.5+ with CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Verify CUDA is available
echo
echo "      Verifying CUDA availability..."
python -c "
import torch
print(f'      PyTorch: {torch.__version__}')
print(f'      CUDA available: {torch.cuda.is_available()}')
print(f'      CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')
if torch.cuda.is_available():
    print(f'      GPU: {torch.cuda.get_device_name(0)}')
"

# ============================================
# Step 6: Install ComfyUI requirements
# ============================================
echo
echo "[6/8] Installing ComfyUI requirements..."

cd ComfyUI
pip install -r requirements.txt
cd ..

# ============================================
# Step 7: Install custom node dependencies
# ============================================
echo
echo "[7/8] Installing custom node dependencies..."

# Core dependencies for GGUF/LLM nodes
pip install gguf sentencepiece transformers accelerate safetensors

# For video processing
pip install imageio imageio-ffmpeg opencv-python

# For audio (JoyCaption TTS)
pip install pyogg==0.6.14a1

# For dynamic prompts
pip install dynamicprompts

# Install all custom node requirements
echo "      Installing requirements from custom nodes..."
cd ComfyUI/custom_nodes
for node_dir in */; do
    if [ -f "${node_dir}requirements.txt" ]; then
        echo "      Installing ${node_dir} dependencies..."
        pip install -r "${node_dir}requirements.txt" 2>/dev/null || true
    fi
done
cd ../..

# ============================================
# Step 8: Install llama-cpp-python with CUDA
# ============================================
echo
echo "[8/8] Installing llama-cpp-python with CUDA support..."
echo "      This compiles from source, may take 5-15 minutes..."
echo

# Set CMAKE args for CUDA
export CMAKE_ARGS="-DGGML_CUDA=on"
export FORCE_CMAKE=1

pip install llama-cpp-python --force-reinstall --no-cache-dir

# ============================================
# Create start script
# ============================================
cat > scripts/start_comfyui.sh << 'EOF'
#!/bin/bash
# Start ComfyUI with GPU support
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$(dirname "$SCRIPT_DIR")"

source venv/bin/activate
cd ComfyUI
python main.py --listen 0.0.0.0 --port 8188 "$@"
EOF
chmod +x scripts/start_comfyui.sh

# ============================================
# Done!
# ============================================
echo
echo "============================================="
echo "  GPU VENV Setup Complete!"
echo "============================================="
echo
echo "Virtual environment location: $(pwd)/$VENV_DIR"
echo
echo "To activate the environment:"
echo "  source $VENV_DIR/bin/activate"
echo
echo "To start ComfyUI:"
echo "  source $VENV_DIR/bin/activate"
echo "  cd ComfyUI"
echo "  python main.py --listen 0.0.0.0 --port 8188"
echo
echo "Or use the start script:"
echo "  ./scripts/start_comfyui.sh"
echo
echo "Next: Download models with the download scripts!"
echo
