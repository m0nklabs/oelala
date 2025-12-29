@echo off
REM ============================================
REM OELALA - Download LLM/Caption Models
REM ============================================
REM SmolLM2 for prompt generation + JoyCaption/QwenVL for captioning

echo =============================================
echo   Downloading LLM/Caption Models (~20GB)
echo =============================================
echo.

set MODELS_DIR=ComfyUI_windows_portable\ComfyUI\models
if not exist "%MODELS_DIR%" set MODELS_DIR=ComfyUI\models

REM -----------------------------------------
REM SmolLM2 for Prompt Generation
REM -----------------------------------------
echo [1/4] Downloading SmolLM2-1.7B-Instruct (3.2GB)...
echo       (Uses git lfs for large files)

cd "%MODELS_DIR%\smol"
if not exist "SmolLM2-1.7B-Instruct" (
    git lfs install
    git clone https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct
) else (
    echo      Already exists, skipping...
)
cd ..\..\..\..

REM -----------------------------------------
REM JoyCaption for Image Captioning (I2T)
REM -----------------------------------------
echo [2/4] Downloading JoyCaption GGUF (6.2GB)...
mkdir "%MODELS_DIR%\LLM\GGUF" 2>nul
curl -L -o "%MODELS_DIR%\LLM\GGUF\llama-joycaption-beta-one-hf-llava.Q6_K.gguf" ^
    "https://huggingface.co/Joseph717171/llama-joycaption-beta-one-hf-llava-GGUF/resolve/main/llama-joycaption-beta-one-hf-llava.Q6_K.gguf"

echo [3/4] Downloading JoyCaption mmproj (838MB)...
curl -L -o "%MODELS_DIR%\LLM\GGUF\llama-joycaption-beta-one-llava-mmproj-model-f16.gguf" ^
    "https://huggingface.co/Joseph717171/llama-joycaption-beta-one-hf-llava-GGUF/resolve/main/llama-joycaption-beta-one-llava-mmproj-model-f16.gguf"

REM -----------------------------------------
REM QwenVL for Video Captioning (V2T)
REM -----------------------------------------
echo [4/4] Downloading Qwen2.5-VL NSFW Caption (5.8GB)...
mkdir "%MODELS_DIR%\llm\GGUF" 2>nul
curl -L -o "%MODELS_DIR%\llm\GGUF\Qwen2.5-VL-7B-NSFW-Caption-V3-Q6_K.gguf" ^
    "https://huggingface.co/bartowski/thesby_Qwen2.5-VL-7B-NSFW-Caption-V3-GGUF/resolve/main/thesby_Qwen2.5-VL-7B-NSFW-Caption-V3-Q6_K.gguf"

echo.
echo =============================================
echo   LLM/Caption Models Download Complete!
echo =============================================
echo.
echo Models installed:
echo   - SmolLM2-1.7B-Instruct: AI prompt generation
echo   - JoyCaption GGUF: Uncensored image captioning
echo   - Qwen2.5-VL NSFW: Video + image captioning
echo.
echo Total downloaded: ~16GB
echo.
pause
