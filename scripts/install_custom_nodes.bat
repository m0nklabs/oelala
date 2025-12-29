@echo off
REM ============================================
REM OELALA - Custom Nodes Installation Script
REM ============================================
REM Installs all required ComfyUI custom nodes for
REM multi-GPU, GGUF, video, and NSFW workflows

echo =============================================
echo   OELALA Custom Nodes Installer
echo =============================================
echo.

REM Check if running in correct directory
if not exist "ComfyUI" (
    echo ERROR: Please run this script from the oelala root directory
    pause
    exit /b 1
)

cd ComfyUI\custom_nodes

echo [1/11] Installing ComfyUI-MultiGPU (multi-GPU inference)...
if not exist "ComfyUI-MultiGPU" (
    git clone https://github.com/pollinations/ComfyUI-MultiGPU.git
) else (
    echo   Already installed, pulling latest...
    cd ComfyUI-MultiGPU
    git pull
    cd ..
)

echo [2/11] Installing ComfyUI-GGUF (GGUF model support)...
if not exist "ComfyUI-GGUF" (
    git clone https://github.com/city96/ComfyUI-GGUF.git
) else (
    echo   Already installed, pulling latest...
    cd ComfyUI-GGUF
    git pull
    cd ..
)

echo [3/11] Installing ComfyUI-WanVideoWrapper (WAN 2.1/2.2 workflows)...
if not exist "ComfyUI-WanVideoWrapper" (
    git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
) else (
    echo   Already installed, pulling latest...
    cd ComfyUI-WanVideoWrapper
    git pull
    cd ..
)

echo [4/11] Installing ComfyUI-VideoHelperSuite (video utilities)...
if not exist "ComfyUI-VideoHelperSuite" (
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
) else (
    echo   Already installed, pulling latest...
    cd ComfyUI-VideoHelperSuite
    git pull
    cd ..
)

echo [5/11] Installing ComfyUI-KJNodes (utility nodes)...
if not exist "ComfyUI-KJNodes" (
    git clone https://github.com/kijai/ComfyUI-KJNodes.git
) else (
    echo   Already installed, pulling latest...
    cd ComfyUI-KJNodes
    git pull
    cd ..
)

echo [6/11] Installing ComfyUI-JoyCaption (image captioning)...
if not exist "ComfyUI-JoyCaption" (
    git clone https://github.com/MoonHugo/ComfyUI-JoyCaption.git
) else (
    echo   Already installed, pulling latest...
    cd ComfyUI-JoyCaption
    git pull
    cd ..
)

echo [7/11] Installing ComfyUI-QwenVL (video captioning)...
if not exist "ComfyUI-QwenVL" (
    git clone https://github.com/IuvenisSapworker/ComfyUI-QwenVL.git
) else (
    echo   Already installed, pulling latest...
    cd ComfyUI-QwenVL
    git pull
    cd ..
)

echo [8/11] Installing comfyui-dynamicprompts (prompt variations)...
if not exist "comfyui-dynamicprompts" (
    git clone https://github.com/adieyal/comfyui-dynamicprompts.git
) else (
    echo   Already installed, pulling latest...
    cd comfyui-dynamicprompts
    git pull
    cd ..
)

echo [9/11] Installing comfyui-portrait-master (portrait generation)...
if not exist "comfyui-portrait-master" (
    git clone https://github.com/florestefano1975/comfyui-portrait-master.git
) else (
    echo   Already installed, pulling latest...
    cd comfyui-portrait-master
    git pull
    cd ..
)

echo [10/11] Installing ComfyUI-Custom-Scripts (various utilities)...
if not exist "ComfyUI-Custom-Scripts" (
    git clone https://github.com/pythongosssss/ComfyUI-Custom-Scripts.git
) else (
    echo   Already installed, pulling latest...
    cd ComfyUI-Custom-Scripts
    git pull
    cd ..
)

echo [11/11] Installing ComfyUI_LayerStyle_Advance (layer styling)...
if not exist "ComfyUI_LayerStyle_Advance" (
    git clone https://github.com/chflame163/ComfyUI_LayerStyle_Advance.git
) else (
    echo   Already installed, pulling latest...
    cd ComfyUI_LayerStyle_Advance
    git pull
    cd ..
)

REM Install requirements for each node
echo.
echo Installing Python dependencies for all nodes...
for /d %%D in (*) do (
    if exist "%%D\requirements.txt" (
        echo   Installing %%D dependencies...
        pip install -r "%%D\requirements.txt" 2>nul
    )
)

REM Install additional packages for GGUF/LLM support
echo.
echo Installing GGUF/LLM dependencies...
pip install gguf sentencepiece transformers accelerate

REM Install llama-cpp-python with CUDA
echo.
echo Installing llama-cpp-python with CUDA support...
set CMAKE_ARGS=-DGGML_CUDA=on
pip install llama-cpp-python --force-reinstall --no-cache-dir

cd ..\..

echo.
echo =============================================
echo   Custom Nodes Installation Complete!
echo =============================================
echo.
echo Installed nodes:
echo   - ComfyUI-MultiGPU     : Multi-GPU model offloading
echo   - ComfyUI-GGUF         : GGUF quantized models
echo   - ComfyUI-WanVideoWrapper : WAN 2.1/2.2 video generation
echo   - ComfyUI-VideoHelperSuite : Video utilities
echo   - ComfyUI-KJNodes      : Utility nodes (Kijai)
echo   - ComfyUI-JoyCaption   : Image captioning (I2T)
echo   - ComfyUI-QwenVL       : Video captioning (V2T)
echo   - comfyui-dynamicprompts : Dynamic prompt variations
echo   - comfyui-portrait-master : Portrait generation
echo   - ComfyUI-Custom-Scripts : Various utilities
echo   - ComfyUI_LayerStyle_Advance : Layer styling
echo.
echo Restart ComfyUI to load the new nodes!
echo.
pause
