### Changed
- Removed all hardcoded private IPs / LAN hostnames from committed code, configs,
  docs and scripts (repo-wide cleanup). Real per-server addresses must now come
  from env/config (`.env`, `.env.example`) or the Admin panel:
  - `compute_backends.py`: the Windows-PC backend's `base_url` is no longer hardcoded;
    it is built from `COMFYUI_WINDOWS_HOST` / `COMFYUI_WINDOWS_PORT` and omitted
    entirely (not just disabled) when the host is unset — the inventory validator
    requires a `base_url` for comfyui backends, so an empty fallback would fail.
  - `compute_backends.json` is now gitignored as a machine-specific config with a
    placeholder template (`compute_backends.json.example`) committed instead, so the
    real address never lands in the public repo.
  - Dev/service URLs in `src/frontend/src/config.js`, `vite.config.js`,
    `storage_utils.py`, `gradio_app.py`, `tests/smoke_test_api.py` and dev scripts now
    use `localhost` instead of LAN IPs.
  - Docs, changelogs, `AGENTS.md`/`CLAUDE.md`/`.goosehints` and
    `.github/copilot-instructions.md` reference hosts by env var or generic labels.

### Added
- Critical rule in `AGENTS.md` (synced to `CLAUDE.md` / `.goosehints`): never commit
  private IPs, ports, hostnames or env-specific values — including in tests and docs;
  use env placeholders or `*.test.invalid` dummies.
- `compute_backends.json.example` template documenting the per-server config workflow.
