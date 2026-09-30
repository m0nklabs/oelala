@echo off
rem OBSOLETE for the Windows H3 host: ComfyUI there is lazy-started by the
rem caretaker-llamacpp wake proxy (service env CARETAKER_COMFY_*, scheduled task
rem "ComfyUIServer", starter D:\start-comfy.ps1, internal port 8189 loopback).
rem Running this script manually would spawn a rogue ComfyUI that fights the
rem wake proxy for ports. Kept only as reference for other setups.
setlocal
cd /d "C:\PROGRAMME\ComfyUI_windows_portable"
set "LOG=C:\PROGRAMME\ComfyUI_windows_portable\comfy_server.log"
echo [%date% %time%] starting ComfyUI server >> "%LOG%"
".\python_embeded\python.exe" -s ComfyUI\main.py --listen 0.0.0.0 --port 8188 --fast-disk >> "%LOG%" 2>&1
echo [%date% %time%] ComfyUI exited, ERRORLEVEL=%ERRORLEVEL% >> "%LOG%"
