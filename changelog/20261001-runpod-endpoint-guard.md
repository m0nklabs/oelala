### Fixed
- Cloud video generation (MiniMax-H3 and LTX-2.3, both text-to-video and
  image-to-video) no longer fails with `503 Service Unavailable`. `_submit_to_runpod()`
  guarded on `_runpod.has_endpoint()` and ignored the `endpoint_id` argument the
  function itself accepts, so when the generic `RUNPOD_ENDPOINT_ID` was unset every
  submission was rejected even though the per-family endpoints were configured and
  the adapter had already resolved one. The guard now accepts an explicit
  `endpoint_id` as sufficient.
- Retiring the Wan 2.2 endpoint left `.env` without any generic default endpoint,
  so `RunPodClient` had no default at all. The generic default now resolves as
  explicit `RUNPOD_ENDPOINT_ID` → first id in `RUNPOD_ENDPOINT_IDS` → the platform
  preference order (`DEFAULT_ENDPOINT_PREFERENCE`: MiniMax-H3, LTX-2.3, i2i,
  resolved through the shared defaults registry). The per-family endpoint
  variables are deliberately kept OUT of `endpoint_ids`: that list feeds the
  health-based failover candidates, and seeding it with family endpoints would let
  a MiniMax-H3 job fail over to the LTX-2.3 worker.

### Changed
- `.env` sets `RUNPOD_ENDPOINT_ID` to the MiniMax-H3 endpoint. MiniMax-H3 is the
  leading video model, so it serves as the generic default; the per-family
  variables remain authoritative for each adapter.
