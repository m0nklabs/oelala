### Changed

- The loop procedure gains §4.1: every run starts by diffing what changed since the last
  run-log entry — new comments, reviews, CI state, and how far the base branch moved — and
  responds to that. Green checks from before a dependency merge prove nothing about the
  merged state, and a merged PR can invalidate the premise of an open one (a new helper this
  PR duplicates, a config knob that replaced a literal). "Nothing new" gets one line with a
  timestamp, not a padded report.
