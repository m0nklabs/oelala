# Compute node types — generalized compute backend config

## Added

- `COMPUTE_NODE_{n}_*` env schema as the fresh-install fallback inventory:
  `TYPE` (`comfy` | `runpod`, extensible enum with `comfyui` accepted as an
  alias), `HOST` + optional `PORT` (default 8188) for `comfy` nodes,
  `NAME` display label, and `MODEL_FAMILIES` comma-list. Liked fallback nodes
  keep deterministic ids `node-{n}`; the scan stops at the first gap and
  invalid groups are skipped with a warning.

## Changed

- The Compute Backend Inventory (`compute_backends.py`, Admin panel → Compute)
  is now the single source of truth for which servers run which model families;
  env vars are only the fallback when `compute_backends.json` is missing.
- Removed the bespoke Windows client path:
  `get_windows_comfyui_client()` and the `COMFYUI_WINDOWS_HOST` /
  `COMFYUI_WINDOWS_PORT` env vars. A second server (e.g. a Windows PC running
  local MiniMax-H3) is now just another `comfyui` backend resolved through the
  inventory (`get_comfyui_client_for_backend`).
- Queue indicator / job-result resolution now poll the actual backend a local
  job is tagged with (by `backend_id`) instead of a hardcoded Windows branch;
  the frontend `server` label shows the backend name.
- Docs (`AGENTS.md`, `docs/COMFYUI_INVENTORY.md`, `docs/GENERATION_MODES.md`,
  `README_MiniMax_H3_workflow.md`) and adapter docstrings reworded to generic
  "compute node" language; `.env.example` documents the new schema.

## Notes

- No hardcoded addresses anywhere: `.env.example` and
  `compute_backends.json.example` use placeholders (`windows-pc.test.invalid`),
  and tests use dummy hosts.
