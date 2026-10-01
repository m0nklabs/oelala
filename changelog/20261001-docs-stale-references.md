### Changed

- **Documentation sweep: stale local-LTX-2 references removed.** LTX is cloud-only
  (LTX-2.3 22B on RunPod `oelala-ltx23`, id `ctpoa610dva4ww`; LTX-2.5 planned in
  `docs/LTX25_MIGRATION.md`), and the local LTX-2 19B set was deleted on 2026-08-23
  (`changelog/minimax-h3-local-windows.md`), but several docs still listed local LTX
  models, a deleted workflow JSON and local DisTorch2 rows for a model that is gone.
- **Fixed active guidance that promised files that no longer exist:**
  - `docs/GENERATION_MODES.md`: the T2V workflow matrix no longer lists
    `ltx2_distorch2_multigpu_api.json` (replaced with the current cloud builders); the
    LTX-2 VAE row and the "LTX-2 19B Q4" allocation row are marked removed; the Gemma
    encoder rows read "LTX-2.3 (cloud)" instead of a local LTX-2 consumer; the two UMT5
    rows now say removed 2026-10-01 with Wan 2.2 instead of "parked".
  - `docs/COMFYUI_INVENTORY.md`: the LTX section is retitled and dated; both UMT5 rows
    now record removal instead of parking.
  - `docs/skills/comfyui-generation.md`: `ImageToVideo/` is documented as empty (the I2V
    graphs come from the adapters) and the `ltx2_audio_*` naming is marked as the legacy
    LTX-2 audio mode.
  - `.github/copilot-instructions.md`: the `ltx2_audio_{index}.mp4` naming convention now
    records that the local LTX-2 audio pipeline was removed on 2026-08-23.
  - `docs/GENERATION_MODES_TREE.md`: the I2V duration table's LTX rows are marked as
    removed local measurements, the Gemma family is labelled cloud-only, and
    `clip_vision_h.safetensors` no longer reads as a loadable model file.
- **Marked history instead of deleting it** — one dated note per document
  ("Removed 2026-08-23 …"), no table, section or measurement dropped:
  `docs/LTX2_STATUS.md`, `docs/LTX2_PERFORMANCE.md` (also
  `ComfyUI/ltx2_audio_test.py` → `scripts/ltx2_audio_test.py` and the node-pack name
  `ComfyUI-LTXVideo`), `docs/LTX2_HANDOVER.md`, `docs/LTX2_RESEARCH.md` (its
  "Recommended" model row now reads "Recommended then") and `docs/TODO_RESEARCH.md`
  (the open "download `ltx-2-19b-distilled-fp8`" items are marked superseded).
- No docs were deleted. LoRA inventories (`docs/lora_registry.yaml`,
  `docs/LORA_DOCUMENTATION.md`, `docs/MODEL_CATALOG.md`, `docs/model_catalog.yaml`) were
  left alone — the regenerated catalog already reports no local LTX model copies.
  `docs/LTX25_MIGRATION.md` was deliberately not touched: it describes the planned
  state and is owned by the operator.

### Fixed

- **Documented endpoints that do not exist in `src/`.** `docs/MEDIA_STORAGE.md` listed
  `/generate-sd15`, `/generate-video` and `/generate-text-video` under "Endpoints with
  Auto-Upload"; no such route exists anywhere in `src/`. The lists now name routes that do
  exist: `/generate-sdxl` (synchronous image, `sdxl-local-t2i` adapter),
  `/generate-ltx2-i2v-async` (LTX-2.3 cloud I2V), `/generate-text` (T2V) and
  `POST /v2/generate` (current unified video path).
- **`docs/SFW_CONTENT_PLAN.md` video settings retargeted to MiniMax-H3.** The "Video
  Settings (To Test)" block still carried Wan-era values (480p, 41 frames at 16 fps, local
  DisTorch2 `cuda:1+cuda:0`). It now uses the verified H3 configuration from
  `docs/GENERATION_MODES_TREE.md` and the H3 adapters: 768×1344 (0.98 MP; template default
  0.4 MP), 124 frames at 24 fps (~5 s), ~3.5-11.5 min per clip on the cloud worker, and it
  records that H3 does not use local DisTorch2. The "~3.5 hours (100 × 124s)" batch estimate
  became "~6-19 hours" from the same verified timings, and "Since we don't have pure T2V"
  now names the H3 T2V adapters. No figure was invented.
- **`docs/WORKFLOWS.md` re-synced with the repaired v1 routes.** The `/generate` row stated
  that the route was still pinned to the retired `wan22-local-i2v-q6` adapter and returned
  HTTP 400; `src/backend/app.py` now hints `minimax-h3-local-i2v`, so the row documents the
  live local MiniMax-H3 I2V path. The `/generate-text` row and its curl example now list the
  accepted `model_type` values (`minimax_h3`, `minimax_h3_local`, `ltx23`, `ltx2`; retired
  Wan 2.2 values are answered with 400), and the Wan-era `wan22-*` routes are documented as
  retired rather than missing.
- **`docs/LTX2_PERFORMANCE.md` no longer points at files the LTX cleanup deleted.** The
  same commit that carried the LTX sweep also removed `scripts/ltx2_audio_test.py`,
  `scripts/test_ltx2_umt5.py` and `ComfyUI/models/vae/ltx2_audio/`, which left three
  references dangling: the two script paths are now annotated "(removed 2026-08-23)", and
  the "Preparing Audio VAE Checkpoint" section carries a dated note saying its snippet
  consumes deleted artifacts and will not run as-is.