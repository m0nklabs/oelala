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

- Object URLs for the source video and the reference previews are now revoked when they are
  replaced, removed or the tool unmounts; a large source video stayed in memory for the life of
  the page.
- Seed `0` stays `0` (`|| -1` turned it into the random sentinel) and a LoRA strength of `0`
  renders as `0` instead of `1.00`, so what the controls show is what gets submitted.
- **Cross-user upload collision:** the local R2V adapter uploaded its media under fixed names
  (`v2_minimax_r2v_source.mp4`, `v2_minimax_r2v_ref_{idx}.png`) while ComfyUI's `/upload/image`
  is called with `overwrite=true`. A second request therefore replaced the media of a workflow
  that was still queued, which then rendered somebody else's video. Upload names now carry a
  per-request id.
- The generate button is locked for the whole submit, not just the network call: reading a large
  source video to a data URL is asynchronous, so a double click used to submit the same input
  twice (and the second request could even start after the first finished).
- Picking a creation twice no longer races: each pick claims a ticket and only the newest one may
  write state, so a slow download of the first pick cannot overwrite the second.
- The reference-preview cleanup reads the live URL set at unmount time instead of the empty set
  captured on the first render, so switching tools actually frees those files.
- The pick guard no longer breaks under React Strict Mode (`main.jsx` enables it): the mount flag is
  restored on setup, because Strict Mode runs setup → cleanup → setup in development and a
  cleanup-only version dropped every later pick as if the tool had unmounted.
