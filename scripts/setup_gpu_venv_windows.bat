@echo off
REM ============================================
REM OELALA - Windows Python Virtual Environment Setup
REM ============================================
REM Creates a proper Python venv with CUDA PyTorch
REM This is the "proper" way vs the portable version
REM
REM REQUIREMENTS:
REM   - Python 3.10 or 3.11 installed and in PATH
REM   - NVIDIA GPU with CUDA support
REM   - CUDA Toolkit 12.4 installed (or 12.1)
REM   - Git installed and in PATH
REM   - ~20GB free disk space for venv + packages
REM
REM Run from the oelala root directory!

setlocal enabledelayedexpansion

echo =============================================
echo   OELALA - Windows GPU VENV Setup
echo =============================================
echo.

REM Check if running in correct directory
if not exist "ComfyUI" (
    echo ERROR: Please run this script from the oelala root directory
    echo.
    echo Expected structure:
    echo   oelala/
    echo     ComfyUI/
    echo     scripts/
    echo     workflows/
    echo.
    pause
    exit /b 1
)

REM ============================================
REM Step 1: Find Python
REM ============================================
echo [1/8] Checking Python installation...

set PYTHON_CMD=

REM Check for Python 3.11 first (preferred)
where python3.11 >nul 2>&1
if %errorlevel%==0 (
    set PYTHON_CMD=python3.11
    goto :found_python
)

REM Check for Python 3.10
where python3.10 >nul 2>&1
if %errorlevel%==0 (
    set PYTHON_CMD=python3.10
    goto :found_python
)

REM Check for generic python
where python >nul 2>&1
if %errorlevel%==0 (
    REM Verify version
    for /f "tokens=2" %%V in ('python --version 2^>^&1') do set PYVER=%%V
    echo      Found Python !PYVER!
    
    REM Check if it's 3.10 or 3.11
    echo !PYVER! | findstr /R "^3\.1[01]" >nul
    if %errorlevel%==0 (
        set PYTHON_CMD=python
        goto :found_python
    )
)

echo ERROR: Python 3.10 or 3.11 is required!
echo.
echo Please install Python from:
echo   https://www.python.org/downloads/
echo.
echo Make sure to check "Add Python to PATH" during installation!
pause
exit /b 1

:found_python
echo      Using: %PYTHON_CMD%
for /f "tokens=*" %%V in ('%PYTHON_CMD% --version 2^>^&1') do echo      Version: %%V

REM ============================================
REM Step 2: Check CUDA
REM ============================================
echo.
echo [2/8] Checking NVIDIA CUDA...

where nvcc >nul 2>&1
if %errorlevel%==0 (
    for /f "tokens=*" %%V in ('nvcc --version ^| findstr "release"') do echo      %%V
) else (
    echo      WARNING: nvcc not found in PATH
    echo      CUDA Toolkit may not be installed or not in PATH
    echo      PyTorch will still work if you have NVIDIA drivers
    echo.
)

nvidia-smi >nul 2>&1
if %errorlevel%==0 (
    echo      NVIDIA driver detected
    for /f "tokens=*" %%V in ('nvidia-smi --query-gpu=name --format=csv,noheader 2^>^&1') do echo      GPU: %%V
) else (
    echo      WARNING: nvidia-smi not found
    echo      Make sure NVIDIA drivers are installed!
)

REM ============================================
REM Step 3: Create Virtual Environment
REM ============================================
echo.
echo [3/8] Creating Python virtual environment...

set VENV_DIR=venv

if exist "%VENV_DIR%" (
    echo      venv already exists. Delete and recreate? [Y/N]
    set /p RECREATE=
    if /i "!RECREATE!"=="Y" (
        echo      Removing old venv...
        rmdir /s /q "%VENV_DIR%"
    ) else (
        echo      Keeping existing venv...
        goto :activate_venv
    )
)

%PYTHON_CMD% -m venv %VENV_DIR%
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment!
    pause
    exit /b 1
)
echo      Created: %VENV_DIR%

:activate_venv
echo      Activating venv...
call %VENV_DIR%\Scripts\activate.bat

REM ============================================
REM Step 4: Upgrade pip and install build tools
REM ============================================
echo.
echo [4/8] Upgrading pip and installing build tools...

python -m pip install --upgrade pip wheel setuptools
if %errorlevel% neq 0 (
    echo ERROR: Failed to upgrade pip!
    pause
    exit /b 1
)

REM ============================================
REM Step 5: Install PyTorch with CUDA
REM ============================================
echo.
echo [5/8] Installing PyTorch with CUDA support...
echo      This may take 5-10 minutes...
echo.

REM PyTorch 2.9.1 with CUDA 12.8 (for RTX 50 series / SM 1.20)
REM Falls back to CUDA 12.4 for older GPUs
echo      Trying CUDA 12.8 first (RTX 50 series)...
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

if %errorlevel% neq 0 (
    echo.
    echo      CUDA 12.8 not available, trying CUDA 12.4...
    pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu124
)

if %errorlevel% neq 0 (
    echo.
    echo      Pinned version failed, trying latest stable...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
)

REM Verify CUDA is available
echo.
echo      Verifying CUDA availability...
python -c "import torch; print(f'      PyTorch: {torch.__version__}'); print(f'      CUDA available: {torch.cuda.is_available()}'); print(f'      CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'      GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

REM ============================================
REM Step 6: Install ComfyUI requirements
REM ============================================
echo.
echo [6/8] Installing ComfyUI requirements...

cd ComfyUI
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo WARNING: Some ComfyUI requirements may have failed
)
cd ..

REM ============================================
REM Step 7: Install custom node dependencies
REM ============================================
echo.
echo [7/8] Installing custom node dependencies...

REM Install from pinned requirements (tested versions)
echo      Installing from requirements-gpu.txt (tested versions)...
if exist "..\requirements-gpu.txt" (
    pip install -r ..\requirements-gpu.txt
) else (
    REM Fallback to manual install with pinned versions
    pip install transformers==4.57.3 accelerate==1.12.0 safetensors==0.7.0
    pip install gguf==0.17.1 sentencepiece==0.2.1
    pip install imageio==2.37.2 imageio-ffmpeg==0.6.0 opencv-python==4.11.0.86
    pip install dynamicprompts==0.31.0
    pip install pyogg==0.6.14a1
    pip install triton==3.5.1
)

REM Install all custom node requirements
echo      Installing requirements from custom nodes...
cd ComfyUI\custom_nodes
for /d %%D in (*) do (
    if exist "%%D\requirements.txt" (
        echo      Installing %%D dependencies...
        pip install -r "%%D\requirements.txt" 2>nul
    )
)
cd ..\..

REM ============================================
REM Step 8: Install llama-cpp-python with CUDA
REM ============================================
echo.
echo [8/8] Installing llama-cpp-python with CUDA support...
echo      This compiles from source, may take 5-15 minutes...
echo.

REM Set CMAKE args for CUDA
set CMAKE_ARGS=-DGGML_CUDA=on
set FORCE_CMAKE=1

REM Try pinned version first
echo      Installing llama-cpp-python==0.3.16 with CUDA...
pip install llama-cpp-python==0.3.16 --prefer-binary 2>nul

REM If that fails, try building from source
if %errorlevel% neq 0 (
    echo      Pre-built wheel not available, building from source...
    pip install llama-cpp-python==0.3.16 --force-reinstall --no-cache-dir
)

REM Final fallback to latest
if %errorlevel% neq 0 (
    echo      Pinned version failed, trying latest...
    pip install llama-cpp-python --force-reinstall --no-cache-dir
)

REM ============================================
REM Done!
REM ============================================
echo.
echo =============================================
echo   GPU VENV Setup Complete!
echo =============================================
echo.
echo Virtual environment location: %CD%\%VENV_DIR%
echo.
echo To activate the environment:
echo   %VENV_DIR%\Scripts\activate.bat
echo.
echo To start ComfyUI:
echo   %VENV_DIR%\Scripts\activate.bat
echo   cd ComfyUI
echo   python main.py --listen 0.0.0.0 --port 8188
echo.
echo Or use the start script:
echo   scripts\start_comfyui.bat
echo.
echo Next: Download models with the download scripts!
echo.

REM Create start script
echo @echo off > scripts\start_comfyui.bat
echo call venv\Scripts\activate.bat >> scripts\start_comfyui.bat
echo cd ComfyUI >> scripts\start_comfyui.bat
echo python main.py --listen 0.0.0.0 --port 8188 %%* >> scripts\start_comfyui.bat

echo Created: scripts\start_comfyui.bat
echo.
pause
