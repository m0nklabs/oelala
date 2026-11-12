### Changed

- The admin panel's disk-usage mounts are configuration now (`ADMIN_DISK_ROOTS`, a comma-separated
  `name=path` list) instead of three literals — `/`, `/home/<user>` and `/mnt/ssd`. The default
  reports the root filesystem and the running user's home, which exist everywhere and need no
  literal; a deployment that keeps data on another mount sets it in `.env`.
- Malformed entries are skipped with a warning rather than raising: a typo in a monitoring setting
  must not take the health endpoint down. An all-malformed value falls back to the default.

This is slice 2b of #226 — the last of the absolute-path literals that live outside the checkout.
