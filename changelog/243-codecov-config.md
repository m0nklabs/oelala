### Fixed

- Added `codecov.yml`. Without it Codecov's default patch status failed every documentation-only
  PR, because a diff with no coverable lines has no coverage data (measured 2026-11-06: #236
  docs-only = failure, #239 code = success). Both statuses are now informational and the paths
  that cannot be covered (`docs/`, `changelog/`, `tests/`, `scripts/`) are ignored, so a red
  codecov check means something again.
