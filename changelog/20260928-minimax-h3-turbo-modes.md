### Added
- MiniMax-H3 turbo quality modes (`draft` / `standard` / `full`) for cloud and
  local T2V/I2V generation, backed by the ModelTC/Minimax-H3-Turbo LoRAs
  (Comfy-Org mirrored files, downloaded by the worker at cold start)
  - `draft` → 4-step 768p LoRA + euler + `MiniMaxH3SigmaShift` 6/3 (trained at
    1344x768; the checkpoint-baked 12/3 must be overridden)
  - `standard` → 8-step LoRA + euler at the base 12/3 shift (official
    ComfyUI-template "turbo mode")
  - `full` → unchanged 20-step base behaviour (default)
  - `GenerationRequest.quality_mode` carries the selection; turbo LoRA chains
    in front of user LoRAs; unit tests in `tests/test_minimax_h3_turbo.py`
