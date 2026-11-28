### Added

- Behavioral suites for two hooks: `useObjectUrls` (a swapped or unmounted preview URL revokes
  its blob, dropped URLs of a tracked set are revoked on change, everything on unmount) and
  `useToolSettings` (defaults when empty, stored values merged over defaults, debounced
  persistence under `oelala_settings_<tool>`, reset clears and returns defaults, corrupt JSON
  falls back, per-tool key isolation). 14 tests; frontend suite 74 → 88 across 10 files.
