### Fixed

- **Legacy v1 routes that could no longer succeed after the Wan 2.2 retirement.**
  Commit `9b06398` removed the family and added the `RETIRED_MODEL_FAMILIES` guard in
  `generation/router.py`, which makes any request whose adapter hint names the retired
  family raise a `ValueError` that `v1_compat.py` maps to HTTP 400. Several v1 routes in
  `src/backend/app.py` were hard-pinned to Wan adapters or defaulted to a Wan
  `model_type`, so they returned 400 on **every** call. Each route was either repointed at
  a live family or reduced to an explicit retirement stub.

  | Route | Action | Why |
  | --- | --- | --- |
  | `POST /generate` | repointed to `minimax-h3-local-i2v` | documented public v1 I2V surface (listed by `GET /`) |
  | `POST /generate-pose` | repointed to `minimax-h3-local-i2v` | its docstring already said "uses the standard I2V workflow"; the hand-built Wan GGUF graph it used loaded deleted weights |
  | `POST /generate-text` | repointed; `model_type` default is now `minimax_h3` | accepts live types (`minimax_h3`, `minimax_h3_local`, `ltx23`, `ltx2`) |
  | `POST /generate-wan22-comfyui` | retired stub → 400 | inherently Wan-specific (local DisTorch2 dual-pass) |
  | `POST /generate-wan22-async` | retired stub → 400 | inherently Wan-specific |
  | `POST /generate-blockswap-q8-async` | retired stub → 400 | Wan-only VRAM strategy |
  | `POST /generate-distorch2-q8-async` | retired stub → 400 | Wan-only VRAM strategy |
  | `POST /generate-ultra-q8-async` | retired stub → 400 | Wan-only VRAM strategy |
  | `POST /generate-cloud-wan22-async` | retired stub → 400 | Wan-only RunPod bf16 route |

  The retired paths stay **registered** rather than removed, so a legacy caller gets the
  canonical `400 Model family 'Wan 2.2' has been retired … Leading video model:
  MiniMax-H3; alternative: LTX-2.3.` instead of a bare 404. A retired `model_type`
  (`wan22`, `wan2.2`, `cloud_wan22`, `WAN22`) is forwarded to the router verbatim, so the
  retirement guard — not a silent fallback — produces the error. An unknown `model_type`
  errors too, instead of being swapped for another family.

- **`GET /comfyui-status` no longer reports a deleted Wan checkpoint**
  (`wan2.2_i2v_low_noise_14B_Q5_K_S.gguf`) as the loaded model.

- **`/ai-suggest` offered parked Wan LoRAs and Wan tuning advice.** `BASE_MODEL_MAP` and
  `MODEL_CONSTRAINTS` now map the live model modes (`minimax_h3`, `minimax_h3_local`,
  `ltx2`, `ltx23`) to their base models and constraints, and the system prompt no longer
  asks the LLM for Wan 2.2 dual-noise high/low LoRA pairs.

### Changed

- **`src/backend/comfyui_client.py`: removed the Wan 2.2 workflow residue** (~2.8k lines).
  Deleted the `WAN22_ENHANCED_Q4KM_API_WORKFLOW`, `WAN22_I2V_Q6_API_WORKFLOW`,
  `WAN22_T2V_Q6_API_WORKFLOW`, `WAN22_I2V_Q5_API_WORKFLOW`, `WAN22_I2V_Q5_WORKFLOW` and
  `WAN22_I2V_DISTORCH2_API_WORKFLOW` graphs plus the builders that fed them
  (`build_api_workflow`, `build_t2v_q6_workflow`, `build_enhanced_workflow`,
  `build_q6_workflow`, `build_blockswap_q8_workflow`, `build_distorch2_q8_workflow`,
  `build_ultra_q8_workflow`, `build_distorch2_workflow`, `generate_distorch2_video`) and
  the Wan-only `WanVideo*` branches of the legacy node-format converter. Three of the
  builders loaded `ImageToVideo/wan22_i2v_*_q8_api.json` files that no longer exist; the
  rest referenced deleted `wan2.2_*` GGUF weights. No callers remained outside this
  module. The dual-stage LoRA plumbing is untouched — the parked Wan LoRAs stay on disk.

- **`src/backend/credits.py`: live video pricing tiers added, Wan tiers kept for history.**
  `credit_transactions.metadata` persists the `generation_type` string
  (`credits_api.deduct_credits`), so historical rows must keep resolving to the price they
  were charged. `WAN22_*` members are therefore kept and explicitly marked
  history-only, while new `VIDEO_I2V_*` / `VIDEO_T2V_*` members carry the same amounts for
  current generations. `/generate`, `/generate-text` and `/generate-pose` now bill through
  the live tiers; the legacy `wan22_i2v` / `wan22_t2v` keys still resolve identically.

- **`src/backend/api_v1.py`**: the public REST `POST /api/v1/generate` bills video through
  the live `video_i2v` tier, and the endpoint docstring names MiniMax-H3 / LTX-2.3
  instead of Wan 2.2.

- **Small residue fixes**: `GET /unet-models`'s docstring no longer presents Wan 2.2 I2V as
  its purpose, `GET /loras` documents the high/low grouping as a legacy dual-pass
  convention, generation stats default `model_mode` to `unknown` instead of `wan2.2`, cloud
  outputs without a recorded family are saved under `cloud-unknown/` instead of
  `cloud-wan22/`, and the RunPod workflow logging recognises the live
  `EmptyLTXVLatentVideo` / `LTXVImgToVideoConditionOnly` / `MiniMaxH3ImageToVideo` nodes.

### Added

- **`tests/test_wan22_route_cleanup.py`** (33 tests): every retired Wan route answers 400
  with a message naming MiniMax-H3 and LTX-2.3; `/generate`, `/generate-pose` and
  `/generate-text` reach the real (execution-stubbed) live adapter instances; `/v2/generate`
  still succeeds for a live hint and still rejects a retired one; and legacy/live credit
  tiers price identically.

### Notes

- **Deliberately kept:** the `GET /media/generated/cloud-wan22/{filename}` storage proxy
  (historical Wan videos are still stored under `cloud-wan22/`; now documented as
  read-only history) and `lora_scanner.py`'s `wan2.2` classification, which describes the
  parked LoRAs still present on disk. `scripts/` keeps its Wan-era download/test scripts —
  out of scope for this change.
- **Not touched:** `docs/` (a documentation sweep had just finished) and `deploy/`.
