### Added

- **Krea 2 LoRA support**: the `krea2-local-t2i` adapter now accepts one LoRA
  via a core `LoraLoaderModelOnly` node (the available Krea 2 LoRAs carry only
  `diffusion_model` weights; routing CLIP through a full LoraLoader is avoided
  on purpose). LoRA names from the UI are resolved server-side to the full
  relative path ComfyUI lists.
- Text-to-Image tool: LoRA picker now also renders for Krea 2 (1 slot,
  mirroring the adapter's `max_loras=1` constraint) next to SDXL (3 slots).
- `docs/lora_registry.yaml`: two Krea 2 NSFW LoRAs added to the catalog —
  `krea2/snofs_krea_v1_4.safetensors` (SNOFS v1.4 by Ashen3) and
  `krea2/KNP_000003000.safetensors` (Krea 2 NSFW V4 v4.3_EXP by
  19doorsside884).
- LoRA base-model detection: `krea2/` subdirectory (and `krea` filename
  marker) now derives `base_model: krea2` so the router's compat filter
  accepts them for krea2 jobs and rejects them for other model families.

### Changed

- `/loras` catalog: the `krea2/` directory is treated as NSFW as a whole
  (registry-documented content) even though its filenames carry no NSFW
  keywords.
- `/loras` response now includes `base_model` per LoRA (derived by the same
  scanner logic the router uses) so the frontend can filter the picker by
  model compatibility.

### Fixed

- Text-to-Image tool: the LoRA picker was nested inside the SDXL-only settings
  fragment (Steps/CFG/sampler/scheduler/seed block), so it never rendered for
  Krea 2 — only SDXL/Pony models showed it. The section is now a sibling
  conditional rendered for every model that accepts LoRAs.
- Text-to-Image tool: the LoRA picker filtered only on the NSFW toggle and
  showed every catalog entry for every model — Krea 2 listed SDXL LoRAs (and
  SDXL would list krea2 ones). The picker now filters by base-model family,
  resets incompatible selections on model switch, and the section stays
  visible for supported models even when every candidate is hidden by the
  NSFW filter (it shows a "hidden by NSFW filter" hint instead).
