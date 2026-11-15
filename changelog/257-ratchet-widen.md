### Changed

- The rule-6 ratchet in `tests/test_no_hardcoded_host_paths.py` widens its pattern from `/home/<user>`
  to every quoted absolute path under `/home`, `/mnt`, `/srv`, `/opt`, `/data`, `/var` and
  `/usr/local`. The old pattern is how `/mnt/ssd/loras` hid in five places for weeks while the config
  sweep (#226) converted 28 `/home/` literals.
- The allowlist freezes exactly the four remaining LoRA literals (`app.py` 2, `lora_scanner.py` 1,
  `generation/lora_utils.py` 1). Slice 2c (#256) removes all four, so after it lands the allowlist is
  empty and any new absolute host path anywhere in the backend fails CI. Mentions in prose (docstrings)
  are not violations: the rule targets code that *uses* a host path.
