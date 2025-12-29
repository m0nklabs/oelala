@echo off
REM ============================================
REM OELALA - Download Base Models (Required)
REM ============================================
REM These are the core models needed for most workflows

echo =============================================
echo   Downloading Base Models (~30GB)
echo =============================================
echo.

set MODELS_DIR=ComfyUI_windows_portable\ComfyUI\models
if not exist "%MODELS_DIR%" set MODELS_DIR=ComfyUI\models

REM -----------------------------------------
REM CLIP Text Encoders
REM -----------------------------------------
echo [1/6] Downloading CLIP encoders...

curl -L -o "%MODELS_DIR%\clip\clip_l.safetensors" ^
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors"

curl -L -o "%MODELS_DIR%\clip\t5xxl_fp8_e4m3fn.safetensors" ^
    "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors"

echo [2/6] Downloading UMT5-XXL for WAN 2.2...
curl -L -o "%MODELS_DIR%\clip\umt5-xxl-enc-bf16.safetensors" ^
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5-xxl-enc-bf16.safetensors"

REM -----------------------------------------
REM CLIP Vision
REM -----------------------------------------
echo [3/6] Downloading CLIP Vision...
curl -L -o "%MODELS_DIR%\clip_vision\clip-vit-large.safetensors" ^
    "https://huggingface.co/openai/clip-vit-large-patch14/resolve/main/pytorch_model.bin"

REM -----------------------------------------
REM VAE
REM -----------------------------------------
echo [4/6] Downloading VAE models...
curl -L -o "%MODELS_DIR%\vae\ae.safetensors" ^
    "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/ae.safetensors"

curl -L -o "%MODELS_DIR%\vae\wan_2.1_vae.safetensors" ^
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors"

REM -----------------------------------------
REM Text Encoders for WAN
REM -----------------------------------------
echo [5/6] Downloading WAN Text Encoders...
curl -L -o "%MODELS_DIR%\text_encoders\umt5_xxl_fp8_e4m3fn_scaled.safetensors" ^
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"

curl -L -o "%MODELS_DIR%\text_encoders\qwen_3_4b.safetensors" ^
    "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/qwen_3_4b.safetensors"

REM -----------------------------------------
REM Flux Dev Checkpoint
REM -----------------------------------------
echo [6/6] Downloading Flux Dev FP8...
curl -L -o "%MODELS_DIR%\checkpoints\flux1-dev-fp8.safetensors" ^
    "https://huggingface.co/Comfy-Org/flux1-dev/resolve/main/flux1-dev-fp8.safetensors"

echo.
echo =============================================
echo   Base Models Download Complete!
echo =============================================
echo.
echo Total downloaded: ~30GB
echo.
pause
