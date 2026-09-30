### Fixed
- MiniMax-H3 local (Windows-PC ComfyUI) LoRA generation failed prompt validation
  with `Value not in list: lora_name: 'minimax-h3/...'`:
  - Windows ComfyUI enumerates loras with **backslash** separators
    (`minimax-h3\X.safetensors`) while the workflow JSON sent forward-slash
    registry names; combo validation is an exact string match, so the prompt was
    rejected even with the files present.
  - `ComfyUIClient.build_local_minimax_h3_{t2v,i2v}_workflow()` now re-map each
    requested LoRA name onto the target server's own LoRA list
    (`get_lora_names()` via `/object_info/LoraLoaderModelOnly`): exact match,
    separator-swap, or unique basename. Names the server does not have are
    dropped with a warning instead of queueing a prompt that can never run; if
    the server list is unreadable the names pass through unchanged (fail-open).

### Changed
- Removed the dead `ComfyUIClient.upload_lora()` HTTP upload path: it targeted
  `/internal/models/upload`, which does not exist in ComfyUI core (404 on the
  v0.33.x portable installs), so the pre-dispatch upload never worked and only
  logged misleading failures. LoRA files must be staged on the target server's
  `models/loras/` out of band.
