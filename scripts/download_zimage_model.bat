@echo off
REM ============================================
REM OELALA - Download Z-Image Turbo Model
REM ============================================
REM Fast single-step image generation

echo =============================================
echo   Downloading Z-Image Turbo (~12GB)
echo =============================================
echo.

set MODELS_DIR=ComfyUI_windows_portable\ComfyUI\models
if not exist "%MODELS_DIR%" set MODELS_DIR=ComfyUI\models

echo Downloading Z-Image Turbo BF16 (12GB)...
echo This is a fast single-step Flux-based model

curl -L -o "%MODELS_DIR%\diffusion_models\z_image_turbo_bf16.safetensors" ^
    "https://huggingface.co/Freepik/flux.1-lite-8B-alpha/resolve/main/z_image_turbo_bf16.safetensors"

echo.
echo =============================================
echo   Z-Image Turbo Download Complete!
echo =============================================
echo.
pause
