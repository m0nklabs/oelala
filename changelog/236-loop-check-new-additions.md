### Changed

- The loop procedure gains §4.1: every run starts by diffing what changed since the last
  run-log entry — new comments, reviews, CI state, and how far the base branch moved — and
  responds to that. Green checks from before a dependency merge prove nothing about the
  merged state, and a merged PR can invalidate the premise of an open one (a new helper this
  PR duplicates, a config knob that replaced a literal). "Nothing new" gets one line with a
  timestamp, not a padded report.

### Fixed

- The §4.2 merge command is bound to the verified commit with `--match-head-commit`: without it a
  commit pushed between verification and merge would be merged on the strength of checks and a
  review that never saw it.
- The `OELALA_ENV_FILE` warning carries a timestamp (rule 13). It arrived with #225 and was
  flagged on this head because this PR merged `main`; tracked as #240, fixed here because it
  blocked this PR.
