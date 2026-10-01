# Local Generation Defaults

This document is the product-facing source of truth for local generation mode
selection. The adapter constraints in `src/backend/generation/adapters/` remain
the code-level source of truth.

## Video Mode Policy

Use the mode selector as the compute decision. Do not let a separate compute
toggle override the selected mode.

| Tool | Mode | Adapter | Compute | Default | Use for |
|------|------|---------|---------|---------|---------|
| I2V | MiniMax-H3 — Cloud | `minimax-h3-cloud-i2v` | RunPod | 0.4 MP template default (0.98 MP recommended), 24fps, 20 steps, cfg 1.0 | Leading model: joint video+audio, 4-15s |
| I2V | MiniMax-H3 — Local (Windows PC) | `minimax-h3-local-i2v` | Local | Same H3 defaults, 24fps | Local generations with native stereo audio |
| I2V | LTX-2.3 | `ltx23-cloud-i2v` | RunPod | 576p, 5s, 25fps, 8 steps, cfg 1.0 | Fast cloud iterations (second choice) |
| T2V | MiniMax-H3 — Cloud | `minimax-h3-cloud-t2v` | RunPod | 0.4 MP template default (0.98 MP recommended), 24fps, 20 steps, cfg 1.0 | Leading text-to-video with audio |
| T2V | MiniMax-H3 — Local (Windows PC) | `minimax-h3-local-t2v` | Local | Same H3 defaults, 24fps | Local text-to-video with native stereo audio |
| T2V | LTX-2.3 | `ltx23-cloud-t2v` | RunPod | 576p, 5s, 25fps, 8 steps, cfg 1.0 | Fast cloud text-to-video (second choice) |

> **Retired 2026-10-01:** Wan 2.2 was retired from the product on 2026-10-01 together with its local (`wan22-local-*`) and cloud (`wan22-cloud-*`) modes and adapters; references to it below are historical. See `docs/LEGACY.md`.

## Simplification Rules

- MiniMax-H3 is the leading path in both the cloud and local (Windows-PC ComfyUI) selectors.
- LTX-2.3 is the second choice and cloud-only; LTX-2.5 is planned (`docs/LTX25_MIGRATION.md`).
- Wan 2.2-era saved profiles (`wan2.2`, `cloud_wan22`, `blockswap_q8`, `distorch2_q8`, `ultra_q8`) no longer have adapters; the frontend coerces them to MiniMax-H3 on restore.
- Cloud-only modes force `compute_target=cloud` in the request payload.
- Local modes force `compute_target=local` in the request payload.
- Backend validation clamps `frames` and `fps` against adapter constraints before queueing work in ComfyUI.

## Why

Local generation should fail less often and be easier to reason about. Users
should pick a mode, not combine an architecture choice with a contradictory
compute toggle. The frontend mode map, backend adapter constraints, and docs
must stay aligned whenever defaults change.
