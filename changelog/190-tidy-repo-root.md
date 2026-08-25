# Tidy repo root — move loose files to their proper directories

## Changed

- Moved loose top-level files to their proper directories (git renames, history preserved):
  - `MiniMax_H3_T2V_workflow.json` → `workflows/`
  - `README_MiniMax_H3_workflow.md` → `workflows/`
  - `download_minimax_h3.cmd` / `.ps1`, `start_comfy_server.bat`, `monitor_h3.sh` → `scripts/`
  - `STORAGE_MIGRATION.md`, `PR_IMPLEMENTATION_NOTES.md`, `PR_SUMMARY.md` → `docs/`
- Updated references to the moved files in `docs/GENERATION_MODES.md`,
  `docs/COMFYUI_INVENTORY.md`, `workflows/README_MiniMax_H3_workflow.md` and the
  local MiniMax-H3 T2V adapter docstring.
- Moved `.pytest_cache/` and `.ruff_cache/` out of the repo root into `scratch/caches/`.
