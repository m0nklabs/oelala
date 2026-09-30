### Added
- Cloud workers now fetch LoRAs from HuggingFace first (fast CDN) and fall
  back to the signed self-hosted download: `build_lora_download_list` picks
  the primary source by priority — curated `LORA_HF_SOURCES` entries, then the
  public flat mirror repo (`LORA_HF_FLAT_MIRROR_REPO`, files stored by
  basename), then the private layout-preserving mirror
  (`LORA_HF_MIRROR_REPO`, token via `HF_LORA_TOKEN`) — and attaches the
  signed `/loras/download` URL as `fallback_url`. All four RunPod worker
  handlers (Wan2.2, I2I, LTX-2.3, MiniMax-H3) try `fallback_url` when the
  primary source fails and only send the HF token to huggingface.co hosts.
- `scripts/sync_loras_hf.py` syncs the local LoRA store to the private mirror
  repo (dry-run mode, refuses public repos) for files not already available
  on a flat mirror.

### Fixed
- `build_lora_download_list` signed URLs introduced earlier in the day are
  preserved as fallback when a HF source is configured.
