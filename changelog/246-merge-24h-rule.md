### Changed

- The merge policy moved from **two days** of mergeability to **24 hours**, matching the revised
  user-level rule. The clock starts when the last outstanding review is answered — or, with
  nothing to review, at the last push on the head — and operator silence for that window is
  standing permission.
- Reading the pull request's latest activity is now part of the merge, not a formality: a reply can
  carry a question, a suggestion or context, and a merge is a deliberate action on one pull request
  rather than a side effect of a sweep.
- The closing comment marks the point at which a pull request became merge-ready; it is no longer a
  precondition for merging when there was nothing to review.
- Re-verification no longer pushes: a base refresh is new work and new work restarts the clock, so
  a refresh-then-merge in one run would be impossible. The base is checked with a dry-run
  `git merge-tree` instead, and a real refresh only happens when the dry run conflicts.
- The merge-time verification now tests what will actually land: a clean merge can still break when
  `main` changed an interface the branch still calls, so the suite runs in a throwaway worktree
  holding the merged tree. And the clean-merge check reads `git merge-tree`'s own exit status —
  piping it through `head` reports success for the one case it must catch, because `merge-tree`
  writes a tree and exits 1 on a conflict.
- The merge-result check aborts on a red suite instead of continuing: the worktree cleanup used to
  run anyway and hand back a success status, which would have let the merge command go ahead on a
  failing merged tree. Cleanup now runs on every exit path through a trap, and a failed merge or
  test stops the sequence with a non-zero status.
