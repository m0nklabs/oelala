### Added
- `guardian_client.py`: Guardian LLM proxy VRAM management client (`GuardianVRAMClient`, `get_guardian()` singleton). Calls `POST /admin/unload` before queueing a ComfyUI workflow so all 28 GB VRAM is available for generation. Guardian auto-reloads the pinned model on the next inference request.
- `queue_prompt()` in `comfyui_client.py` now calls `get_guardian().unload_sync()` before every workflow submission (non-fatal if Guardian is unreachable).
- `GUARDIAN_BASE_URL` / `GUARDIAN_API_KEY` env vars documented in `.env.example`.
