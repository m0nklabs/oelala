### Fixed
- Cloud LoRA downloads from the RunPod worker failed with HTTP 422: URLs
  built by `build_lora_download_list` omitted the `?token=` query parameter
  that the `/loras/download` endpoint requires (HMAC-signed with
  RUNPOD_API_KEY). The URL builder now signs local-LoRA URLs with the shared
  `lora_download_token()` helper (single source of truth, also used by
  app.py's endpoint); the unused `lora_download_token_fn` parameter is
  removed. Affects all cloud adapters (MiniMax-H3, Wan2.2, LTX-2.3, cloud I2I).
  Regression tests in `tests/test_lora_download_token.py`.
