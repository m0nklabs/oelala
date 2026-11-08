### Added

- **Video → Video** (MiniMax-H3 Reference-to-Video / ref2va): a source video plus up to four
  reference images, a frame-grid picker, quality/MP presets, a LoRA stack and a local-vs-cloud
  compute target. Registered as `minimax-h3-local-r2v` and `minimax-h3-cloud-r2v`.

### Fixed

- The mode offered no way to pick its source video from My Creations or the gallery — only a local
  file upload. There is now a **"Select from My Creations"** picker (the reusable `MyMediaTool` in
  selection mode, as `PostProcessingTool` already does) and the mode is listed under **"Use in
  tool"**, so a video can be sent straight from My Creations or the Gallery. Both paths feed the
  existing upload handler, so the submit path is unchanged.
