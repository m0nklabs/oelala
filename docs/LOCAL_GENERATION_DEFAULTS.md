# Local Generation Defaults

This document is the product-facing source of truth for local generation mode
selection. The adapter constraints in `src/backend/generation/adapters/` remain
the code-level source of truth.

## Video Mode Policy

Use the mode selector as the compute decision. Do not let a separate compute
toggle override the selected mode.

| Tool | Mode | Adapter | Compute | Default | Use for |
|------|------|---------|---------|---------|---------|
| I2V | Stable Local — Wan2.2 Q6 | `wan22-local-i2v-q6` | Local | 480p, 8s, 16fps, 6 steps, cfg 1.0 | Reliable local generations |
| I2V | Quality Local — Wan2.2 Q8 | `wan22-local-i2v-distorch2` | Local | 480p, 10s, 16fps, 8 steps, cfg 1.0 | Higher quality after a good seed is found |
| I2V | Cloud Wan2.2 | `wan22-cloud-i2v` | RunPod | 720p, 8s, 15 steps, cfg 3.0 | Full precision quality or local VRAM avoidance |
| I2V | LTX-2.3 | `ltx23-cloud-i2v` | RunPod | 576p, 5s, 8 steps, cfg 1.0 | Fast cloud iterations |
| T2V | Wan2.2 Q6 | `wan22-local-t2v-q6` | Local | 480p, 5s, 16fps, 6 steps, cfg 1.0 | Reliable local text-to-video |
| T2V | Cloud Wan2.2 | `wan22-cloud-t2v` | RunPod | 720p, 5s, 15 steps, cfg 3.0 | Full precision quality |
| T2V | LTX-2.3 | `ltx23-cloud-t2v` | RunPod | 576p, 5s, 8 steps, cfg 1.0 | Fast cloud text-to-video |

## Simplification Rules

- Stable Local is the default local path.
- Quality Local is the only promoted local quality upgrade in the main I2V selector.
- BlockSwap and Ultra Q8 adapters stay available for saved profiles/backward compatibility, but they are not promoted in the main mode selector.
- Cloud-only modes force `compute_target=cloud` in the request payload.
- Local modes force `compute_target=local` in the request payload.
- Backend validation clamps `frames` and `fps` against adapter constraints before queueing work in ComfyUI.

## Why

Local generation should fail less often and be easier to reason about. Users
should pick a mode, not combine an architecture choice with a contradictory
compute toggle. The frontend mode map, backend adapter constraints, and docs
must stay aligned whenever defaults change.
