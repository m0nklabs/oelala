### Fixed
- MiniMax-H3 local generation now works against the lazy-started Windows ComfyUI
  (behind the caretaker-llamacpp **wake proxy** on port 8188)
  - `ComfyUIClient.is_available()` gained a `timeout` parameter (default 5 s,
    unchanged for always-on backends); the local H3 T2V/I2V adapters wait up to
    300 s so the caretaker wake proxy can cold-start ComfyUI (~2 min boot after
    its 300 s idle-stop) before answering `/system_stats`
  - Connection-refused still fails fast, so a switched-off PC errors immediately
  - Blocking client calls in both local H3 adapters (preflight, image upload,
    queue) now run via `asyncio.to_thread` — a cold wake previously froze the
    backend's async event loop for the full wake duration, stalling all API
    traffic including `/health`
