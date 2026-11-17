### Removed

- `imageio` and `imageio-ffmpeg` are gone from `requirements.txt`. Nothing imports them: no module in
  `src/`, `scripts/`, `tests/` or `deploy/` has ever imported `imageio` or `imageio_ffmpeg` (checked
  the full git history), `pip show` reports an empty `Required-by` for both in the production venv,
  and the backend's ffmpeg work muxes audio through the `ffmpeg` binary directly (subprocess in
  `app.py` and `face_service.py`), not through imageio's shim. CI's backend-import drift check has
  been passing for weeks on a manifest that never contained imageio, and the full suite passes in an
  environment where the package was never installed. Removing an unused pin cannot break what nothing
  uses; this supersedes the 0.8.1 bump proposed in #248.
