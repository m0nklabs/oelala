### Added
- MiniMax-H3 `model_variant` option, backed by `MINIMAX_H3_VARIANTS`:
  `"official"` (Comfy-Org weights, default), plus mirrored Civitai finetunes —
  `"eros"` (beta5 turbo-hybrid fp8/w4a8), `"eros_int8_turbo"`,
  `"eros_int8"` (non-turbo), `"dasiwa_turbo"` (DaSiWa Hybrid Turbo v2) and
  `"dasiwa"` (DaSiWa Hybrid v2, non-turbo). Turbo variants carry a baked
  distillation and run draft/standard at 4/8 euler steps with no turbo LoRA or
  sigma-shift nodes; non-turbo variants (and turbo variants asked for "full")
  keep the caller's step count and the base res_multistep sampler. A variant
  never silently swaps in a different checkpoint.
- `scripts/mirror_civitai_model.py`: mirrors a Civitai model file into the
  private HF model repo, resolving the filename from the Civitai API so the
  mirror keeps the upstream name, and printing a safetensors header summary
  (dtypes + quantization format) before uploading.
- Worker model downloads are now on demand: only the checkpoint/turbo-LoRA files
  a job's workflow actually references are fetched, instead of every entry at
  startup (an Eros job no longer pulls the 21 GB official DiT). Failures now
  return the worker log tail plus a per-model existence pre-check, and ComfyUI's
  validation detail is surfaced instead of a bare "Failed to queue workflow".
- Civitai checkpoints download through the HuggingFace mirror
  (`m0nk111/oelala-models`, private) with a direct-URL fallback using
  `CIVITAI_TOKEN`; tokens are never logged.

### Fixed
- `/user/media/{type}/{filename}/workflow` was unreachable: the generic
  `{filename:path}` route was registered first and swallowed the `/workflow`
  suffix, so "use in tool" always returned an empty workflow. A route shim now
  registers ahead of the generic route.
### Changed
- Checkpoint mirrors moved to the **public** HuggingFace repo `bomehika/oelala-models`
  (public repos have unlimited free storage; the private `m0nk111/oelala-models` hit
  HuggingFace's private-storage limit at ~77 GB). The worker registry,
  `scripts/mirror_civitai_model.py` and the model catalog all point at the public
  mirror; the Civitai URL stays as the download fallback.
