### Added

- `requirements-web.txt` — the web/API runtime (fastapi, uvicorn, python-dotenv,
  sentry-sdk, PyJWT[crypto], Pillow, python-multipart, httpx, minio, websocket-client,
  pyyaml, requests, cachetools, supabase), verified on Python 3.12 without torch.
  `requirements.txt` remains the GPU/ML stack; a full backend installs both.

### Changed

- CI installs `requirements-web.txt` and runs `import app` as a hard step, so a missing
  pin fails the backend job instead of being skipped by the `|| echo` fallbacks.
- `docs/HANDOFF.md` section 8 documents which manifest each host needs.
