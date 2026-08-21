# Skill: ComfyUI workflows & generation modes

How to add, test, and document AI generation on oelala.

## Where things live
- API-format workflows: `workflows/` (organized by family: `ImageToVideo/`, `TextToImage/`, ...).
- ComfyUI install: `ComfyUI/` (models in `ComfyUI/models/`, custom nodes in `ComfyUI/custom_nodes/`).
- Workflow builders live in `src/backend/comfyui_client.py` (`_build_*_workflow()`).
- Backend orchestration: `src/backend/comfyui_client.py`, `runpod_client.py`, `job_queue.py`.

## Multi-GPU (DisTorch2)
- Allocation string: `cuda:0,10gb;cuda:1,15gb;cpu,*` with **RTX 3060 first** (cuda:0).
- Use DisTorch2 loader nodes for Wan2.2:
  `UnetLoaderGGUFAdvancedDisTorch2MultiGPU`, `VAELoaderDisTorch2MultiGPU`, `CLIPLoaderDisTorch2MultiGPU`.
- Include `expert_mode_allocations` on ALL loader nodes. Check `[MultiGPU DisTorch V2]` in ComfyUI logs.
- Full guide: `docs/DISTORCH2_MULTI_GPU_SETTINGS.md`. Model/VRAM inventory: `docs/COMFYUI_INVENTORY.md`.

## VRAM budget (WAN 2.2 14B, tested)
- 480×848 @ 321 frames ≈ 26GB (SAFE production max)
- 576×1024 @ 81–121 frames ≈ 24–27GB (standard)
- 720×1280 @ 41–61 frames ≈ 27GB (tight)
- Always use `PatchSageAttentionKJ` to save 15–20% VRAM.

## Cloud (RunPod)
- Cloud workers: Wan2.2, LTX-2.3, MiniMax-H3, Qwen I2I — see `.github/copilot-instructions.md` for endpoint/template IDs.
- Deploy with `deploy/runpod*/deploy.sh` (dated tags). NEVER `docker push :latest`.

## DOCUMENT EVERY SUCCESSFUL RUN (MANDATORY)
1. Update `docs/GENERATION_MODES_TREE.md` → "Tested Configurations Log".
2. Include: tool type, mode, resolution, frames, VRAM used, workflow file, result.
3. Add new modes to the correct tool section; flip status 🔨→✅ when production-ready.
4. Add detail to `docs/GENERATION_MODES.md` if a new mode.
- **Why**: prevents re-discovering working settings; single source of truth for "what works".

## Storage / naming
- Save to canonical dirs only: `media/generated/`, `ComfyUI/output/`. Never create new output dirs.
- Naming: `oelala_t2v_{YYYYMMDD}_{HHMMSS}_{index}.mp4`, `oelala_i2v_...`, `oelala_t2i_{index}_.png`, `ltx2_audio_{index}.mp4`.
