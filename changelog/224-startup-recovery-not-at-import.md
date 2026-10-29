### Fixed

- Importing `src/backend/app.py` no longer rewrites training-job state. The module called
  `face_train_service.recover_stuck_jobs()` while it was being imported, and that function
  marks every job still in `running` as `failed` — correct on a real restart, wrong on an
  import, so a test run, a script or a REPL could fail live training jobs and later report
  them to the user as "backend restarted during training". The call now runs in the
  `startup` hook, where it belongs.
- The recovery can no longer prevent the backend from booting: a failure is logged as a
  warning instead of escaping the import block.

### Added

- `tests/test_startup_recovery_scope.py` proves the import path is inert in a fresh
  interpreter (a seeded `running` job survives `import app`) and that the startup hook
  still performs the recovery.

### Changed

- `tests/conftest.py` redirects `face_train_service.JOBS_DIR` / `JOBS_INDEX` to a temporary
  directory instead of stubbing the recovery function, so the suite exercises the real code
  path without reading or rewriting the checkout's job index.
