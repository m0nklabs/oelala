# Skill: Changelog fragments

Every PR **must** add a changelog fragment in `changelog/`. This is non-negotiable.

## Why
- Prevents merge conflicts on the root `CHANGELOG.md`.
- Fragments are auto-merged into `CHANGELOG.md` on release by `scripts/merge_changelog.py`.

## Rules
1. **Never edit** root `CHANGELOG.md` directly.
2. Add a file: `changelog/{PR_NUMBER}-{short-description}.md`
   - e.g. `changelog/83-websocket-progress.md`
   - For PR-number-less local work, use a date-based name if needed (see existing files).
3. Use sections with `### ` headings: `### Added`, `### Fixed`, `### Changed`, etc.
4. One entry per PR, scoped to what the PR actually changes.

## Validation
- `.github/workflows/changelog-check.yml` enforces the presence of a fragment.
- Run `scripts/merge_changelog.py` (release time) to fold fragments into `CHANGELOG.md` — not part of normal dev commits.
