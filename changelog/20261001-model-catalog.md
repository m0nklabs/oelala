### Added
- **Model catalog** (`docs/model_catalog.yaml`, generated `docs/MODEL_CATALOG.md`):
  one source of truth for every component ComfyUI can load — role
  (diffusion_model/checkpoint/text_encoder/vae/lora/upscaler/other), target dir,
  family, quantization, size, sources (HF repo, Civitai url + fileId, public and
  private mirrors) and where the file actually lives (local ai-kvm2 ComfyUI,
  Windows-PC, RunPod worker, mirror, or Civitai-only). 249 entries across
  flux/flux2/krea2/ltx23/minimax_h3/qwen_image_edit/sdxl/wan21/wan22.
- `scripts/build_model_catalog.py` builds the catalog from the live registries
  (`deploy/runpod*/handler.py`, `MINIMAX_H3_VARIANTS`, `LORA_HF_SOURCES`,
  compute backends), the resolved local ComfyUI tree and the HuggingFace mirrors;
  `--check` fails when the committed catalog has drifted.
- `tests/test_model_catalog.py` keeps the catalog honest: every worker registry
  model and every H3 variant checkpoint must be present, and mirrored Civitai
  files must keep their upstream names.
- `scripts/mirror_civitai_model.py` mirrors a Civitai model file into the model
  repo (filename resolved from the Civitai API, safetensors header summarized
  before upload).
