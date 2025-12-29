@echo off
REM ============================================
REM OELALA - ComfyUI NSFW Setup - Windows Install
REM ============================================
REM Run this script in PowerShell or CMD as Administrator

echo =============================================
echo   OELALA ComfyUI Setup - Windows Installer
echo =============================================
echo.

REM Check if running in correct directory
if not exist "ComfyUI" (
    echo ERROR: Please run this script from the oelala root directory
    echo        Make sure you have cloned the repository first
    exit /b 1
)

echo [1/5] Downloading ComfyUI Portable...
if not exist "ComfyUI_windows_portable" (
    curl -L -o ComfyUI_portable.7z "https://github.com/comfyanonymous/ComfyUI/releases/latest/download/ComfyUI_windows_portable_nvidia.7z"
    7z x ComfyUI_portable.7z -y
    del ComfyUI_portable.7z
) else (
    echo      Already exists, skipping...
)

echo.
echo [2/5] Installing custom nodes...
cd ComfyUI_windows_portable\ComfyUI\custom_nodes

REM Clone all required custom nodes
if not exist "ComfyUI-GGUF" git clone https://github.com/city96/ComfyUI-GGUF.git
if not exist "ComfyUI-VideoHelperSuite" git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
if not exist "ComfyUI-MultiGPU" git clone https://github.com/pollockj/ComfyUI-MultiGPU.git
if not exist "ComfyUI-WanVideoWrapper" git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
if not exist "ComfyUI-KJNodes" git clone https://github.com/kijai/ComfyUI-KJNodes.git
if not exist "comfyui-dynamicprompts" git clone https://github.com/adieyal/comfyui-dynamicprompts.git
if not exist "comfyui-portrait-master" git clone https://github.com/florestefano1975/comfyui-portrait-master.git
if not exist "ComfyUI-Custom-Scripts" git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git
if not exist "ComfyUI_LayerStyle_Advance" git clone https://github.com/zombieyang/ComfyUI_LayerStyle_Advance.git
if not exist "ComfyUI-JoyCaption" git clone https://github.com/1038lab/ComfyUI-JoyCaption.git
if not exist "ComfyUI-QwenVL" git clone https://github.com/1038lab/ComfyUI-QwenVL.git

cd ..\..\..

echo.
echo [3/5] Installing Python dependencies...
cd ComfyUI_windows_portable\python_embeded
python.exe -m pip install --upgrade pip
python.exe -m pip install gguf sentencepiece transformers accelerate
python.exe -m pip install pyogg==0.6.14a1

REM Install llama-cpp-python with CUDA support
set CMAKE_ARGS=-DGGML_CUDA=on
python.exe -m pip install llama-cpp-python --force-reinstall --no-cache-dir

cd ..\..

echo.
echo [4/5] Copying workflows from repository...
xcopy /E /I /Y "ComfyUI\user\default\workflows" "ComfyUI_windows_portable\ComfyUI\user\default\workflows"

echo.
echo [5/5] Creating model directories...
cd ComfyUI_windows_portable\ComfyUI\models

REM Create all required directories
if not exist "checkpoints" mkdir checkpoints
if not exist "clip" mkdir clip
if not exist "clip_vision" mkdir clip_vision
if not exist "diffusion_models" mkdir diffusion_models
if not exist "loras" mkdir loras
if not exist "smol" mkdir smol
if not exist "text_encoders" mkdir text_encoders
if not exist "unet" mkdir unet
if not exist "vae" mkdir vae
if not exist "LLM\GGUF" mkdir "LLM\GGUF"
if not exist "llm\GGUF" mkdir "llm\GGUF"

cd ..\..\..

echo.
echo =============================================
echo   Installation Complete!
echo =============================================
echo.
echo Next steps:
echo   1. Run download scripts to get models:
echo      - scripts\download_base_models.bat
echo      - scripts\download_wan22_models.bat
echo      - scripts\download_nsfw_models.bat
echo      - scripts\download_llm_models.bat
echo.
echo   2. Start ComfyUI:
echo      cd ComfyUI_windows_portable
echo      run_nvidia_gpu.bat
echo.
echo   3. Open browser: http://127.0.0.1:8188
echo.
pause
