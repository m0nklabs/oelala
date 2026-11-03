### Changed

- Checkout-relative paths now come from one place, `src/backend/project_paths.py`, instead of
  absolute `/home/<user>/oelala/...` literals. Converted in this slice: the frontend, ComfyUI
  output, LoRA, workflow-registry, `data/` and `logs/` paths in `app.py`, `admin_api.py` and
  `websocket_handler.py`, plus the `sys.path` root. On the production checkout every resolved
  value is identical to the literal it replaced; from any other checkout a path now points at
  that checkout, which is the point — a worktree previously read and wrote the production data
  directory.
- The remaining absolute paths (venv binaries, the MinIO data dir, filesystem-browser roots)
  are not checkout-relative and need configuration; they are tracked in #226.

### Added

- `tests/test_no_hardcoded_host_paths.py` is a ratchet: a new absolute host path in the
  backend fails the suite, and the allowlist of known remaining ones must shrink, never grow.
  It also pins that every converted constant still resolves inside the checkout.
