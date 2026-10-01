### Changed

- **Documentation sweep: removed stale Wan 2.2 references from `docs/` and
  `.github/`.** The family was retired from the product on 2026-10-01
  (`changelog/20261001-retire-wan22.md`, `docs/LEGACY.md`), but several docs
  still described it as the current video model. Video guidance now leads with
  **MiniMax-H3** and **LTX-2.3** as the second choice (LTX-2.5 planned,
  `docs/LTX25_MIGRATION.md`).
- **Fixed active guidance that had become wrong** — instructions, reference
  tables and endpoints that named deleted artifacts:
  `.github/copilot-instructions.md` (dropped the "always use DisTorch2 loader
  nodes for Wan2.2" rule, the deleted `oelala-wan22` endpoint/template/image
  block, the deleted workflow path, the Wan example log entry and the
  `deploy/runpod/` deploy path), `docs/skills/comfyui-generation.md`,
  `docs/LOCAL_GENERATION_DEFAULTS.md` (mode table now lists the live
  `minimax-h3-*` and `ltx23-*` adapters), `docs/HANDOFF.md`,
  `docs/GENERATION_MODES.md`, `docs/COMFYUI_INVENTORY.md`,
  `docs/MULTI_GPU_SETUP.md`, `docs/HARDWARE_LIMITS.md`, `docs/WAN2.md`,
  `docs/WORKFLOWS.md` (the legacy `/generate` route is documented as pinned to a
  retired adapter and returning HTTP 400; use `POST /v2/generate`),
  `docs/ADVANCED_VIDEO.md`, `workflows/ADVANCED_VIDEO_README.md`, `docs/API_v1.md`,
  `docs/MEDIA_STORAGE.md`, `docs/WEB_INTERFACE_README.md`,
  `docs/ARCHITECTURE.md`, `docs/AGENT_CONTEXT.md`,
  `docs/COMFYUI_INTEGRATION.md`, `docs/RUNPOD_GPU_TIERS.md`,
  `docs/PROJECT_OVERVIEW.md`, `docs/TEST_I2T_LLM_I2V_PIPELINE.md` and
  `docs/SFW_CONTENT_PLAN.md`.
- **Marked history instead of deleting it.** Pricing, plan, vision, research
  and changelog-style docs keep their text and gained a one-line dated note
  ("Retired 2026-10-01 ... see `docs/LEGACY.md`"): `docs/MONETIZATION.md`,
  `docs/UI_V2_PLAN.md`, `docs/STORY_WRITER_VISION.md`, `docs/PROJECT_PLAN.md`,
  `docs/ROADMAP.md`, `docs/LTX2_RESEARCH.md`, `docs/LEGAL_RESEARCH.md`,
  `docs/TODO_RESEARCH.md`, `docs/TODO_TOOLS.md`, `docs/TODO_LIST.md`,
  `docs/RUNPOD_NEW_ENDPOINT_CHECKLIST.md`, `docs/LTX25_MIGRATION.md` and
  `docs/DISTORCH2_MULTI_GPU_SETTINGS.md` (DisTorch2 technique kept; the
  Wan-specific measurements are marked as a historical record).
- **Kept the parked LoRA inventory.** The ~36 Wan 2.2 LoRAs stay parked for a
  possible open-weight "Wan 3", so their entries are unchanged in
  `docs/lora_registry.yaml`, `docs/LORA_DOCUMENTATION.md`,
  `docs/MODEL_CATALOG.md`, `docs/model_catalog.yaml` and
  `docs/.private/NSFW_LORAS.md`; a one-line parked-legacy marker was added at the
  top of the two inventory docs to match the registry's existing `legacy: true`
  entries.
- No docs were deleted. `docs/LEGACY.md`, `docs/GENERATION_MODES_TREE.md` and
  the generated catalogs were left to their owning changes.