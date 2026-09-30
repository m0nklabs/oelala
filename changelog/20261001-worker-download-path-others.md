### Fixed
- The Qwen I2I and LTX-2.3 RunPod workers no longer nest HF downloads one level
  too deep. `hf_hub_download` writes to `local_dir/<hf_path>`, and both handlers
  passed `local_dir=str(target.parent)` while the registry's `hf_path` already
  starts with a folder (`split_files/...`) — so e.g. the Qwen diffusion model
  landed in `models/diffusion_models/split_files/diffusion_models/` and was moved
  back on every cold start (`📂 Moved ...`), leaving empty directories in the
  ComfyUI tree. The same `_hf_local_dir()` / `_prune_empty_parents()` helpers as
  the MiniMax-H3 worker (b4c1b31) now download folder-carrying repo paths into
  the models root and prune the leftover nesting after a move. The Wan 2.2
  worker already downloads into a dedicated staging dir and needed no change.
