### Added
- MiniMax-H3 cloud video generation (t2v + i2v) via a new RunPod endpoint
  - New `deploy/runpod-minimax-h3/` Docker infrastructure (Dockerfile, handler.py, deploy.sh)
  - MiniMax-H3 FL2VA: joint video+audio DiT — every generation includes a synchronized soundtrack
  - Same checkpoint serves t2v and i2v (i2v anchors the input image as first keyframe)
  - `build_cloud_minimax_h3_t2v_workflow()` — text-to-video, mirrors the official Comfy-Org template
  - `build_cloud_minimax_h3_i2v_workflow()` — image-to-video (first-frame keyframe)
  - Programmatic workflow builders (no JSON templates needed)
- New adapters: `MiniMaxH3CloudT2VAdapter` + `MiniMaxH3CloudI2VAdapter`
- `RUNPOD_MINIMAX_H3_ENDPOINT_ID` / `RUNPOD_MINIMAX_H3_TEMPLATE_ID` environment variables
- `minimax_h3` profile in `src/backend/runpod_defaults.py` (80GB+ GPUs, 45 min execution timeout)
- Endpoint live: `5xuvnvyww4ujnc` (oelala-minimax-h3), template `fpfo4gmnrw`
  (serverless, image `ghcr.io/m0nklabs/oelala-minimax-h3-worker`)
- Frontend: MiniMax H3 modes in TextToVideo + ImageToVideo tools
  - "Kwaliteit (megapixels)" selector (0.4 / 0.6 / 0.98 native / 2.0 = 2K MP) —
    exacte canvas-dimensies via de officiële ResolutionSelector formule
  - Duur 3–15 s (getraind bereik 124–362 frames), frame count-snap op 17k+5 grid
  - Vaste 24 fps; geen negative prompt / CFG in de UI (model heeft dat niet)
  - Backend: `megapixels` veld op `GenerationRequest` + `_minimax_h3_canvas()`
    helper (native 768-short-edge canvas én MP-formule, beide ×32 afgerond)

### Technical Details
- Diffusion model: `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (20.97 GB, Comfy-Org/MiniMax-H3)
- Text encoder: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (15.69 GB, Qwen3-VL-32B, no Blackwell needed)
- VAEs: `minimax_h3_video_vae_fp16.safetensors` (5.21 GB) + `minimax_h3_audio_vae_fp32.safetensors` (0.61 GB)
- Total cold-start download: ~42.5 GB; container disk 100 GB
- ComfyUI cloned from official `Comfy-Org/ComfyUI` master (H3 core nodes are in `comfy_extras/nodes_minimax_h3.py`)
- Sampling: simple/20-step schedule, `res_multistep` sampler, BasicGuider (no negative prompt), 24 fps, 17k+5 frame grid
- Output: mp4 with muxed audio via VHS_VideoCombine
- GPU: 80GB+ tiers; the int8/nvfp4 quantizations may also fit 48GB tiers (untested)
