@echo off
REM ============================================
REM OELALA - Download NSFW Models
REM ============================================
REM NSFW-focused models for image and video generation

echo =============================================
echo   Downloading NSFW Models (~65GB)
echo =============================================
echo.
echo WARNING: These models contain NSFW content
echo.

set MODELS_DIR=ComfyUI_windows_portable\ComfyUI\models
if not exist "%MODELS_DIR%" set MODELS_DIR=ComfyUI\models

REM -----------------------------------------
REM NSFW Checkpoints (SDXL/Pony)
REM -----------------------------------------
echo [1/5] Downloading CyberRealistic Pony V14.1 FP16 (6.5GB)...
curl -L -o "%MODELS_DIR%\checkpoints\CyberRealistic_Pony_v14.1_FP16.safetensors" ^
    "https://huggingface.co/cyberdelia/CyberRealistic_Pony/resolve/main/CyberRealistic_Pony_v14.1_FP16.safetensors"

echo [2/5] Downloading Realistic Vision V5.1 (4GB)...
curl -L -o "%MODELS_DIR%\checkpoints\Realistic_Vision_V5.1.safetensors" ^
    "https://huggingface.co/SG161222/Realistic_Vision_V5.1_noVAE/resolve/main/Realistic_Vision_V5.1_fp16-no-ema.safetensors"

echo [3/5] Downloading ReaPony V9.0 (6.5GB)...
curl -L -o "%MODELS_DIR%\checkpoints\reapony_v90.safetensors" ^
    "https://civitai.com/api/download/models/1119970"

REM -----------------------------------------
REM NSFW Flux Diffusion Models
REM -----------------------------------------
echo [4/5] Downloading FluxedUp NSFW 5.1 FP8 (11GB)...
curl -L -o "%MODELS_DIR%\diffusion_models\fluxedUpFluxNSFW_51FP8.safetensors" ^
    "https://civitai.com/api/download/models/1203915"

echo [5/5] Downloading Persephone NSFW 1.1 FP8 (11GB)...
curl -L -o "%MODELS_DIR%\diffusion_models\persephoneFluxNSFWSFW_11FP8.safetensors" ^
    "https://civitai.com/api/download/models/1033850"

REM -----------------------------------------
REM NSFW WAN 2.2 Enhanced Models
REM -----------------------------------------
echo.
echo [OPTIONAL] Downloading WAN 2.2 Enhanced NSFW Models (24GB)...
echo These are specialized for NSFW video generation

curl -L -o "%MODELS_DIR%\unet\wan22EnhancedNSFW_V2_Q6K_HIGH.gguf" ^
    "https://huggingface.co/MonkeyWitATool/wan22EnhancedNSFW_V2/resolve/main/wan22EnhancedNSFW_V2_Q6K_HIGH.gguf"

curl -L -o "%MODELS_DIR%\unet\wan22EnhancedNSFW_V2_Q6K_LOW.gguf" ^
    "https://huggingface.co/MonkeyWitATool/wan22EnhancedNSFW_V2/resolve/main/wan22EnhancedNSFW_V2_Q6K_LOW.gguf"

echo.
echo =============================================
echo   NSFW Models Download Complete!
echo =============================================
echo.
echo Total downloaded: ~65GB
echo.
pause
