### Fixed

- `ruff.toml` pins its rule set (`select = ["E4", "E7", "E9", "F"]`) instead of relying on ruff's
  default. That default grows between releases, so the same tree was clean under 0.15.0 and reported
  **1610 errors under 0.17.3** and **1640 under 0.17.4** — none of them caused by a code change. With
  the rule set pinned, both versions now report `All checks passed!` on `src/` and on everything the
  config includes.
- This is the other half of #238. #241 pins the tool in `ci.yml` and `.pre-commit-config.yaml`; this
  pins the rules. Together the gate means the same thing on every machine and through a runner
  upgrade, and widening the rule set becomes a deliberate commit rather than a side effect.

Not included, deliberately: adopting the wider rules today. `--select ALL` reports 2070 findings, and
turning those on is a ratchet of its own (`UP` 480, `BLE001` 248, `B008` 191, `I001` 88, `RUF012` 53),
not a config change hidden in a lint-hygiene PR.
