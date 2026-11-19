### Changed

- CI's frontend job runs on Node 24 (was 20): vitest 5's bundled undici crashes at import time on
  Node 20.x (`Cannot read properties of undefined (reading 'bind')`), so the workers never started
  and the freshly unmasked test step failed. Proven with the runner's own binaries: the suite is
  0/3 files green on 20.20.2 (3 startup crashes) and 15/15 green on 24.22.3 — the version already in
  the runner tool cache. Local dev runs Node 26 and was unaffected.
