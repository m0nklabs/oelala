@echo off
setlocal enabledelayedexpansion
REM MiniMax H3 download (Comfy-Org repack) - int8 pruned set voor 16GB-kaart
REM Herstartbaar: curl -C - hervat waar gebleven.
set "MR=C:\PROGRAMME\ComfyUI_windows_portable\ComfyUI\models"
set "BASE=https://huggingface.co/Comfy-Org/MiniMax-H3/resolve/main"
set "LOG=C:\PROGRAMME\ComfyUI_windows_portable\h3_download.log"

if not exist "%MR%\diffusion_models" mkdir "%MR%\diffusion_models"
if not exist "%MR%\text_encoders" mkdir "%MR%\text_encoders"
if not exist "%MR%\vae" mkdir "%MR%\vae"
if not exist "%MR%\loras" mkdir "%MR%\loras"

echo [%date% %time%] H3 download gestart >> "%LOG%"

call :dl "%MR%\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors" "%BASE%\diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors?download=1"
call :dl "%MR%\text_encoders\qwen3vl_32b_minimax_h3_int8_convrot.safetensors" "%BASE%\text_encoders\qwen3vl_32b_minimax_h3_int8_convrot.safetensors?download=1"
call :dl "%MR%\vae\minimax_h3_video_vae_fp16.safetensors" "%BASE%\vae\minimax_h3_video_vae_fp16.safetensors?download=1"
call :dl "%MR%\vae\minimax_h3_audio_vae_fp32.safetensors" "%BASE%\vae\minimax_h3_audio_vae_fp32.safetensors?download=1"
call :dl "%MR%\loras\minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors" "%BASE%\loras\minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors?download=1"

echo [%date% %time%] H3 download klaar >> "%LOG%"
exit /b 0

:dl
echo [%date% %time%] downloaden: %~n1 >> "%LOG%"
curl.exe -L -C - --retry 5 --retry-delay 5 -o "%~1" "%~2" >> "%LOG%" 2>&1
echo [%date% %time%] klaar: %~n1 (ERRORLEVEL=%ERRORLEVEL%) >> "%LOG%"
exit /b 0