### Added
- **Compute Backend Inventory** — modular, configurable sources of compute for generation
  - New `src/backend/generation/compute_backends.py` + `compute_backends.json`: a typed
    `ComputeBackend` model (id, name, `type` = `comfyui` | `runpod`, `base_url`, `enabled`,
    `model_families`, `notes`), a JSON loader/saver, and resolvers
    (`list_backends`, `get_backend`, `enabled_backends`, `resolve_backend_for_model`).
  - Generic client factory `get_comfyui_client_for_backend()` in `comfyui_client.py`
    (cached by `(backend_id, base_url)`) so any configured ComfyUI server becomes usable
    with no code change — and a `base_url` edit takes effect on the next dispatch.
  - `client_fn_for_model()` / `client_fn_for_utility()` — adapters resolve their ComfyUI client
    lazily from the inventory each dispatch instead of a hardcoded client.
  - Admin API: `GET/POST /api/admin/backends`, `PUT/DELETE /api/admin/backends/{id}` (admin-only)
  - Admin UI: new **"Compute"** tab in the Admin panel — view, add, edit, enable/disable,
    delete backends, and set their model-family capabilities.

### Changed
- Generation adapter factory now routes **MiniMax-H3 local** and other local adapters through the
  inventory (`resolve_backend_for_model`) instead of hardcoded `get_windows_comfyui_client`.
- Local-job result resolution (`_resolve_local_job_result`) uses the `backend_id` attached at
  dispatch to pick the right ComfyUI server, falling back to the legacy adapter-name check.
- Adding a compute source is now a **configuration change** (Admin UI + JSON) rather than a code change,
  matching the "RunPod is just a container with a temporary ComfyUI" model.

### Notes
- Falls back to built-in defaults (ai-kvm2, Windows-PC, RunPod) if the JSON is missing/unreadable,
  so generation never breaks.
- RunPod endpoint IDs / API keys remain in `.env`; the inventory references them by backend id.
