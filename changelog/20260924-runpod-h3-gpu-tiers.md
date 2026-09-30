### Changed
- MiniMax-H3 RunPod endpoint: GPU tiers lowered from 80GB+-only
  (`AMPERE_80, ADA_80_PRO, HOPPER_141, BLACKWELL_96, BLACKWELL_180`) via a
  24 GB experiment to **discrete >=48 GB cards only**: `AMPERE_48,AMPERE_80`
  (A6000/A40 at $1.22/hr floor, A100 80 GB fallback)
  - Removes accidental H100/H200/B200 placements ($4.79–8.64/hr)
  - 24 GB proved too small: a 768x1344 / 10 s / 20-step I2V job timed out
    against the handler's 40-min workflow limit on the RTX PRO 6000 MIG
    1g.24gb slice (measured 12.6 min for the 5 s variant, 2.3x slower than an
    L40S)
  - RunPod quirk: the scheduler places pool jobs on the MIG 1g.24gb slice even
    when ADA_48_PRO is selected (docs and the config validator deny it belongs
    there, and the "-" exclusion is rejected for the same reason) — dropping
    ADA_48_PRO is the only deterministic way to enforce a 48 GB floor
  - gpuIds changes propagate to the scheduler with a delay of minutes; test
    jobs submitted right after a change reflect the previous config
  - Verified live: 768x1344-class placement lands on NVIDIA A40; L40S runs the
    5 s / 20-step 768p workload in 5.6 min with `--fast-disk` (2.3x faster than
    the MIG slice)
- Worker now reports the CUDA device name in the job output (`gpu` field) for
  per-job GPU-type/cost attribution
