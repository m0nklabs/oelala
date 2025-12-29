@echo off
REM ============================================
REM OELALA - Download WAN 2.2 Video Models
REM ============================================
REM WAN 2.2 14B models for I2V and T2V

echo =============================================
echo   Downloading WAN 2.2 Video Models (~72GB)
echo =============================================
echo.

set MODELS_DIR=ComfyUI_windows_portable\ComfyUI\models
if not exist "%MODELS_DIR%" set MODELS_DIR=ComfyUI\models

REM -----------------------------------------
REM WAN 2.2 GGUF Models (Q6_K quantized)
REM -----------------------------------------
echo [1/4] Downloading WAN 2.2 I2V High Noise Q6_K (12GB)...
curl -L -o "%MODELS_DIR%\unet\wan2.2_i2v_high_noise_14B_Q6_K.gguf" ^
    "https://huggingface.co/city96/Wan2.2-I2V-14B-480P-GGUF/resolve/main/wan2.2-i2v-14b-480p-Q6_K.gguf"

echo [2/4] Downloading WAN 2.2 I2V Low Noise Q6_K (12GB)...
curl -L -o "%MODELS_DIR%\unet\wan2.2_i2v_low_noise_14B_Q6_K.gguf" ^
    "https://huggingface.co/city96/Wan2.2-I2V-14B-480P-Low-GGUF/resolve/main/wan2.2-i2v-14b-480p-low-Q6_K.gguf"

echo [3/4] Downloading WAN 2.2 T2V High Noise FP8 (14GB)...
curl -L -o "%MODELS_DIR%\unet\wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors" ^
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_14b_high_noise_fp8_scaled.safetensors"

echo [4/4] Downloading WAN 2.2 T2V Low Noise FP8 (14GB)...
curl -L -o "%MODELS_DIR%\unet\wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors" ^
    "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.2_t2v_14b_low_noise_fp8_scaled.safetensors"

REM -----------------------------------------
REM Seko 4-Step LoRAs (Optional but recommended)
REM -----------------------------------------
echo.
echo [OPTIONAL] Downloading Seko 4-Step LoRAs (4.5GB)...
mkdir "%MODELS_DIR%\loras\Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1" 2>nul
mkdir "%MODELS_DIR%\loras\Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1" 2>nul

curl -L -o "%MODELS_DIR%\loras\Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1\high_noise_model.safetensors" ^
    "https://huggingface.co/Seko0815/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/resolve/main/high_noise_model.safetensors"

curl -L -o "%MODELS_DIR%\loras\Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1\low_noise_model.safetensors" ^
    "https://huggingface.co/Seko0815/Wan2.2-I2V-A14B-4steps-lora-rank64-Seko-V1/resolve/main/low_noise_model.safetensors"

curl -L -o "%MODELS_DIR%\loras\Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1\high_noise_model.safetensors" ^
    "https://huggingface.co/Seko0815/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/resolve/main/high_noise_model.safetensors"

curl -L -o "%MODELS_DIR%\loras\Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1\low_noise_model.safetensors" ^
    "https://huggingface.co/Seko0815/Wan2.2-T2V-A14B-4steps-lora-rank64-Seko-V1.1/resolve/main/low_noise_model.safetensors"

echo.
echo =============================================
echo   WAN 2.2 Models Download Complete!
echo =============================================
echo.
echo Total downloaded: ~72GB
echo.
pause
