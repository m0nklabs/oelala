### Added
- MiniMax-H3 **model variant + sampling selectors** in the frontend (Text-to-Video and
  Image-to-Video): `eros`, `eros_int8_turbo`, `eros_int8`, `dasiwa_turbo`, `dasiwa`,
  `official` plus `draft` (4 steps) / `standard` (8 steps) / `full` (20 steps). The choice is
  saved with the rest of the tool settings and sent as `model_variant` / `quality_mode`.
  The variant selector is cloud-only; the local Windows backend always runs the official
  weights (the finetunes live in the cloud mirror). Defaults are `eros` + `draft`.

### Fixed
- The legacy V1 request mapper silently dropped `quality_mode` and `model_variant`
  (they were missing from `_PASSTHROUGH_FIELDS`), so API clients could not select a
  checkpoint variant at all. The V2 JSON endpoint already accepted both.
