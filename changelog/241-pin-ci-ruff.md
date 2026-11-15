### Fixed

- The lint gate no longer depends on which ruff the self-hosted runner happens to have cached:
  `ci.yml` pins `ruff==0.15.0`.

  Unpinned, `ruff check src/` resolved to the runner's cached 0.15.0, where the tree passes;
  ruff 0.17 reports **1610 findings** on the same tree (`UP045`, `UP006`, `BLE001`, `B008`, …).
  A runner image update would have turned every PR red at once, for reasons unrelated to any
  change — and a local `ruff check` disagreed with CI, which is how it was found.

### Known limitation

- `.pre-commit-config.yaml` stays on ruff `v0.8.6`. Aligning it with the CI pin was tried and
  fails on the runner: pre-commit runs under `/usr/bin/python3`, which has no `virtualenv`
  module, so it cannot build the environment for a hook whose rev changed. Both versions report
  the tree clean and formatted today; aligning them needs the runner's python repaired first
  (issue #238).
