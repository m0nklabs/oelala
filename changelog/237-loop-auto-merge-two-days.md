### Changed

- The loop may now merge a PR itself: §4.2 grants auto-merge for a PR that has been
  continuously mergeable for more than two days, after re-verifying on the current base in the
  same run (refresh, re-run the suite, fresh green checks and a fresh PR-Piet review on the
  head). Merge readiness itself is defined as "the latest review on the current head is
  answered — by a fix or by an evidence-backed reply", which is the user-level rule from
  `AGENTS.md`.
- Hard stop 1 no longer forbids merging outright; it now points at §4.2 and keeps the rest:
  never approve, never dismiss a review, never resolve a thread with an unanswered finding.
  The human-gated five (auth, migrations, secrets, deploy, dependency majors) stay human, and
  at most three merges happen per run.
