### Added
- MiniMax-H3 LoRA support (local / Windows-PC ComfyUI)
  - `LoraLoaderModelOnly` LoRA stacking for MiniMax-H3 T2V + I2V workflows:
    `_build_minimax_h3_workflow()` now chains single-stage LoRAs in front of the base
    `UNETLoader`, re-pointing the BasicGuider + BasicScheduler at the last loader.
    The `build_cloud_minimax_h3_*_workflow()` / `build_local_minimax_h3_*_workflow()`
    builders all accept an optional `lora_configs=[{name, strength}, ...]`.
  - `MiniMaxH3LocalT2VAdapter` / `MiniMaxH3LocalI2VAdapter` now pass `req.loras`
    (single-stage `{name, strength}`) into the workflow builders and upload the requested
    LoRA files from `/mnt/ssd/loras/minimax-h3` to the Windows-PC ComfyUI server before
    dispatch (new `ComfyUIClient.upload_lora()` via `/internal/models/upload`, `type=loras`).
    Missing/unfounded LoRAs are logged and skipped rather than failing the job.
  - `lora_scanner._derive_base_model()` recognises `minimax_h3` (subdir `minimax-h3/`,
    name markers `minimax` / `fl2va`) so MiniMax-H3 LoRAs are no longer mis-labelled as `wan2.2`.
  - LoRA registry entries (`docs/lora_registry.yaml`) for the three MiniMax-H3 LoRAs:
    `bounceV07_fl2va-000230_Intense`, `PenisV2_minimax-h3_epoch60`, `Vagina_minimax-h3_epoch20`
    (base_model `minimax_h3`, modes t2v+i2v).

### Changed
- Frontend: the TextToVideo LoRA panel is now available for the local MiniMax-H3 mode
  (`minimax_h3_local`); it stays hidden for the cloud worker (`minimax_h3`), which does
  not support LoRAs yet.
