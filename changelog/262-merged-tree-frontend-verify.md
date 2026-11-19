### Changed

- `docs/OELALA_LOOP.md` §4.2's merged-tree verification now covers the frontend suite: run the vitest
  tests on the merged tree for `src/frontend` changes, and provision `node_modules` first (symlink or
  `npm ci`) because a throwaway worktree has none — a vacuous test run counts as red, not green.
  Recorded after a merge abort on exactly that tooling gap, not on a red suite.
