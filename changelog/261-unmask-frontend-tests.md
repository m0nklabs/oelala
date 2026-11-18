### Changed

- CI's frontend test step no longer swallows failures: `npm test -- --run || true` is now
  `npm test -- --run`. The mask was a no-op before #254 (no test script) and stayed only as a legacy
  habit; since the vitest harness landed it has been hiding real regressions. Proven locally: a
  deliberately broken assertion exits 1 (the step fails, the PR goes red) and the restored tree exits
  0. The frontend suite is 15 tests across 3 files on `main`.
