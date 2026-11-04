### Fixed

- The lint gate no longer depends on which ruff the self-hosted runner happens to have cached.
  `ci.yml` pins `ruff==0.15.0`, and `.pre-commit-config.yaml` moves its ruff hook to the same
  `v0.15.0` so the two gates cannot disagree.

  Unpinned, `ruff check src/` resolved to the runner's cached 0.15.0, where the tree passes;
  ruff 0.17 reports **1610 findings** on the same tree (`UP045`, `UP006`, `BLE001`, `B008`, …).
  A runner image update would have turned every PR red at once, for reasons unrelated to any
  change — and a local `ruff check` disagreed with CI, which is how it was found.
