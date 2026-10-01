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