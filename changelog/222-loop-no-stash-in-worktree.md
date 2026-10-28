### Changed

- The loop procedure forbids `git stash` inside the loop worktree. The worktree shares
  `.git` with the operator clone, so `refs/stash` is a single stack for both checkouts —
  a `stash pop` there can pull the operator's WIP into the loop tree and a `stash drop`
  can destroy it. The rule records the recovery path (`git stash store <sha>`) with its
  limit: a dropped stash is unreachable and only recoverable until garbage collection
  prunes it, so recovery is a race rather than a guarantee.
