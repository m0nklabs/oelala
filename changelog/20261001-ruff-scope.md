### Fixed
- Ruff only linted `src/**/*.py`, so `deploy/`, `tests/` and `scripts/` were silently
  unchecked — a `ruff check deploy/ tests/` run reported "No Python files found" and
  passed without looking at anything. `ruff.toml` now covers all four trees (with
  `tests/manual/` excluded, since those are ad-hoc scratch scripts that pytest ignores)
  and the 65 findings that surfaced are fixed: 56 auto-fixes (unused imports, f-strings
  without placeholders) plus explicit `noqa` markers on two imports that exist purely as
  availability checks in `tests/verify_credits_implementation.py`.

### Changed
- Lint claims are now meaningful outside `src/`: run `/home/flip/venvs/gpu/bin/ruff check`
  without arguments to check everything that ships.
