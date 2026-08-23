@echo off
setlocal
cd /d "C:\PROGRAMME\ComfyUI_windows_portable"
set "LOG=C:\PROGRAMME\ComfyUI_windows_portable\comfy_server.log"
echo [%date% %time%] starting ComfyUI server >> "%LOG%"
".\python_embeded\python.exe" -s ComfyUI\main.py --listen 0.0.0.0 --port 8188 --fast-disk >> "%LOG%" 2>&1
echo [%date% %time%] ComfyUI exited, ERRORLEVEL=%ERRORLEVEL% >> "%LOG%"
