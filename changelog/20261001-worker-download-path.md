### Fixed
- Worker downloads no longer nest the file one level too deep. `hf_hub_download`
  writes to `local_dir/<hf_path>`, and the MiniMax-H3 registry passes an `hf_path`
  that already starts with the target folder (`diffusion_models/...`) while
  `local_dir` was that same folder — so every download landed in
  `models/diffusion_models/diffusion_models/` and had to be moved back
  (`📂 Moved ...`). `_hf_local_dir()` now downloads into the models root when the
  repo path carries a folder, so all eleven registry entries land exactly on
  target and no move is needed; a leftover nesting from an older run is pruned
  afterwards. Not a caching defect, but it produced noise and left empty
  directories in the ComfyUI tree.
