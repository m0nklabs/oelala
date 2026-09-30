### Changed
- RunPod MiniMax-H3 worker: ComfyUI pinned to release tag **v0.37.2** (new `COMFYUI_VERSION`
  build arg in `deploy/runpod-minimax-h3/Dockerfile`) instead of a 2026-08-16 master snapshot
  - Brings the same memory management the Windows-PC portable install (0.33.1) already
    benefits from: H3 peak-memory fix (Comfy-Org/ComfyUI#15486), H3 memory-estimation fix
    (#15983), comfy-aimdo 0.5.5 DynamicVRAM with auto `--fast-disk` (#16333), and
    int8_convrot VAE kernels via comfy-kitchen
  - Turbo-ready: the `MiniMaxH3SigmaShift` node required by the ModelTC/Minimax-H3-Turbo
    LoRAs needs ComfyUI 0.31+
  - `requirements.txt` does not pin torch, so the base image keeps torch 2.9.1/cu1300
    (comfy-kitchen requires CUDA >= 13 runtime)
