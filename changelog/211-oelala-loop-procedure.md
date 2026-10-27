### Added
- Added `docs/OELALA_LOOP.md`, the authoritative procedure for the autonomous nightly
  oelala loop (04:00 UTC): preflight in a dedicated worktree, open-PR remediation with
  PR-Piet suggestion handling, issue triage, pipeline seeding from unmerged branches,
  roadmap advance, and a per-run report on the tracking issue.
- Documented the loop's autonomy boundary (everything except merge) and its hard stops:
  no merges, no pushes to `main`, no service restarts, no access to the operator clone
  or other repositories.
