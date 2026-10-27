### Changed

- The loop procedure forbids `git stash` inside the loop worktree. The worktree shares
  `.git` with the operator clone, so `refs/stash` is a single stack for both checkouts —
  a `stash pop` there can pull the operator's WIP into the loop tree and a `stash drop`
  can destroy it. The rule records the recovery path (`git stash store <sha>`) because a
  dropped stash is still a reachable commit.
