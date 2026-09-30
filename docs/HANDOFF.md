# HANDOFF — Oelala

> Cold handoff: current state, what changed, what is verified, what comes next.
> Hot rules live in `AGENTS.md`; per-change detail lives in `changelog/`.

**Updated:** 2026-09-30 (UTC) · **Owner:** operator (flip) · **Branch:** `main`

---

## 1. Where the system stands

| Piece | State |
|---|---|
| Backend | `oelala-backend` (systemd, :7998) — healthy, restarted after the changes below |
| Frontend | `oelala-frontend` (systemd, vite preview :5174) — rebuilt and restarted |
| Local ComfyUI | `comfyui` (systemd, :8188), always-on, DisTorch2 multi-GPU |
| H3 cloud worker | RunPod endpoint `oelala-minimax-h3`, template `fpfo4gmnrw`, image `20260930-*`, gpuIds `AMPERE_48,AMPERE_80` |
| Other workers | `oelala-wan22`, `oelala-ltx23`, `oelala-i2i` — deployed with the same download-fallback loop |
| Model mirror (checkpoints) | **public** HF repo `bomehika/oelala-models` (5 H3 finetunes, 97.9 GB) |
| LoRA mirrors | private `m0nk111/oelala-loras`, flat public dump `Serenak/chilloutmix` |

Secrets stay in `.env` (never committed): `HF_LORA_TOKEN` (private mirrors),
`HF_PUBLIC_TOKEN` (public mirror uploads), `CIVITAI_TOKEN` (Civitai downloads),
`RUNPOD_API_KEY`. The RunPod **template** env carries `HF_TOKEN`, `CIVITAI_TOKEN`
and `COMFYUI_PATH` — the worker container does not read the repo `.env`.

## 2. What this session changed

**MiniMax-H3 model variants (cloud)**
- `MINIMAX_H3_VARIANTS` in `comfyui_client.py`: `official`, `eros` (beta5 turbo-hybrid
  fp8/w4a8), `eros_int8_turbo`, `eros_int8` (non-turbo), `dasiwa_turbo`, `dasiwa`.
  Turbo variants run draft/standard as 4/8 euler steps with **no** turbo LoRA and no
  sigma-shift node; non-turbo variants keep the caller's step count with
  `res_multistep`. A variant never silently swaps checkpoints.
- `quality_mode` + `model_variant` flow: frontend selectors → `/v2/generate`
  (`GenerationRequest`) → cloud adapters → builder. The legacy V1 form mapper was
  **dropping both fields** (`_PASSTHROUGH_FIELDS`) — fixed.
- Worker registry (`deploy/runpod-minimax-h3/handler.py`) downloads checkpoints
  **on demand** from the workflow instead of every entry at startup, so an Eros job no
  longer pulls the 21 GB official DiT. Failures now return the worker log tail plus a
  per-model existence pre-check, and ComfyUI's validation detail is surfaced.

**Mirrors and tooling**
- `scripts/mirror_civitai_model.py` — mirrors a Civitai file to the public model repo
  (filename resolved from the Civitai API, safetensors header summarized first).
- `scripts/extract_h3_scene_vocab.py` → `src/backend/h3_scene_vocab.json` (46
  dimensions, 772 options, 7 stage foley profiles, 15 style packs, 10 formats).
- `scripts/build_model_catalog.py` → `docs/model_catalog.yaml` + `docs/MODEL_CATALOG.md`
  (249 components with role, family, quant, sources and where each file lives).

**Prompt generation**
- `DEFAULT_H3_PROMPT_SYSTEM` gained **sound and body guards**: wet/close-mic sound must
  never read as chewing/crunch/eating; bodies keep two arms, two legs, one head, five
  fingers, no melt or face smear; continuity across cuts; one camera move per shot and a
  locked reference camera for image inputs.
- `src/backend/generation/prompt_scene.py` + `randomize` / `scene_seed` / `duration` on
  `/generate-prompt`: rolls a concrete scene (place, light, wardrobe, act arc with
  per-shot camera) and hands it to the prompt model as a constraint block; the rolled
  scene comes back in the response. 🎲 toggle in the Prompt Generator UI.

**Frontend**
- H3 model + sampling selectors in Text-to-Video and Image-to-Video (cloud only).
- "Use in tool" works again: `/user/media/{type}/{filename}/workflow` was unreachable
  (the generic `{filename:path}` route was registered first and swallowed `/workflow`);
  a route shim now registers ahead of it. Metadata extraction also reads the VHS
  `prompt` tag, not just `comment`.

## 3. Verified numbers (A40, 768×1344, 124 frames, same prompt/seed)

| Run | Steps | Wall time | Sharpness (Laplacian var) | Audio RMS |
|---|---|---|---|---|
| official full | 20 (res_multistep) | 688.6 s | 107.2 | −8.7 dB |
| official standard | 8 + turbo LoRA | 326.6 s | 21.9 | −29.4 dB (near silent) |
| eros draft (fp8 turbo) | 4 (euler) | 204.9 s | 25.4 | −7.7 dB |
| eros standard (fp8 turbo) | 8 (euler) | 325.2 s | 29.9 | −6.8 dB |
| eros int8 (non-turbo) at 4/8 | 4 / 8 | 210 / 337 s | 8.6 / 11.4 | ≈ −52 dB |

Reading: the Eros turbo files beat the official turbo-LoRA route at equal or lower step
counts and keep usable audio; official full remains the sharpness ceiling at ~3.4× the
cost of draft. The int8 Eros file behaves as the non-turbo variant (undercooked at 4–8
steps) — a 20-step run is still unmeasured.

## 4. Test state

`pytest tests/ --ignore=tests/gpu --ignore=tests/manual` → **632 passed, 7 failed,
8 skipped**. The 7 failures are **pre-existing**: verified on a clean worktree of
`85c7e25` (same 7). They are 4× `test_v2_api` (fixture passes `get_current_user=None`),
2× `test_storage_client` presigned URLs (MinIO env) and 1× `test_api_v1` credits (env).
`ruff check` is clean on everything touched. New coverage this session:
`test_minimax_h3_turbo.py`, `test_runpod_h3_ondemand.py`, `test_model_catalog.py`,
`test_prompt_scene.py`, `test_lora_download_token.py` (made hermetic — other modules
import `app.py`, which loads `.env` and set the mirror variables).

## 5. Open questions and carry-overs

1. **Provenance of the randomizer page** (`~/scratch/h3_porn_scene_randomizer (62).html`)
   — the file carries no author, generator marker or links, so it cannot be determined from
   the artifact. Vocabulary phrases were extracted (factual, low risk); porting its *code*
   would need the source.
2. **Windows-PC column in the catalog is `unverified`** for all 249 entries. The backend
   points at `windows-pc-comfyui` (`http://192.168.1.245:8188`, enabled) but the box did not
   answer within 55 s — re-run `scripts/build_model_catalog.py` when it is awake.
3. **Cold-start cost** — each cold worker still downloads the core set (text encoder +
   VAEs ≈ 21.5 GB) per job variant; a network volume would remove it (costs storage).
4. **AGENTS.md maintenance** (other repo): `guardian docs/HANDOFF.md` and
   `docs/AGENT_JOURNAL.md` are over budget — offer a batched archive-first pass.

## 5b. Legacy cleanup (in progress)

`docs/LEGACY.md` holds the inventory: which families are retired, what was deleted, what
is parked and where a parked file can be fetched again. Wan 2.2 is retired — the product
surface (adapters, UI, workflows, RunPod endpoint) is being removed and its 43.4 GB of
local runtime weights are deleted afterwards; the ~13 GB LoRA set stays parked for a
possible open-weight Wan 3. Local re-fetch sources are recorded there before deletion.

## 6. LLM options for the prompt generator — built

All four landed (see `changelog/20261001-prompt-llm-options.md`):

1. **Model choice for the H3 skill** — the pinned Huihui-Qwen3.5-9B is the default, but any
   model in the list can be selected for H3 targets (the backend already accepted `model`).
2. **Scene mode** — `scene_mode` on `/generate-prompt`: `random` (dice roll), `creative`
   (the model picks place/lighting/wardrobe/mood/style/format from shortlists first, then
   writes the shots; falls back to a dice roll when the picker answer is unusable) or off.
3. **Model comparison** — `POST /generate-prompt/compare` (max 3 models) submits one queued
   job per model and returns job ids; the Prompt Generator polls `/llm-job/{id}` and shows
   the prompts side by side. Queue-based on purpose: one H3 skill run can take minutes.
4. **Per-call controls** — `temperature` (default 1.2) and `llm_seed` are request fields,
   with a temperature slider plus reset in the UI; `scene_seed` re-rolls the same scene.

Not yet verified end-to-end:

- No live call has gone through the creative picker or the comparison endpoint. The queue
  path and the JSON parsing are unit-tested; watch the first real run (an unusable picker
  answer falls back to a dice roll by design, so a bad run is still a valid prompt).
- Provenance of the randomizer page is still open (vocabulary phrases were extracted;
  porting its *code* would need the source).
- Private mirror cleanup (`m0nk111/oelala-models`, 77 GB of the private quota) and the
  `unverified` Windows-PC column in the model catalog are still pending.

## 7. Key commands

```bash
# tests / lint
/home/flip/venvs/gpu/bin/python -m pytest tests/ -q --ignore=tests/gpu --ignore=tests/manual
/home/flip/venvs/gpu/bin/ruff check src/ scripts/ tests/

# worker deploy (image build + template update)
./deploy/runpod-minimax-h3/deploy.sh

# mirror a Civitai checkpoint (resolves the upstream filename itself)
/home/flip/venvs/gpu/bin/python scripts/mirror_civitai_model.py --version <id> --file-id <id>

# regenerate the model catalog / check drift
/home/flip/venvs/gpu/bin/python scripts/build_model_catalog.py [--check]

# re-extract the H3 scene vocabulary from the randomizer page
/home/flip/venvs/gpu/bin/python scripts/extract_h3_scene_vocab.py --source "<page.html>"

# services (systemd only)
sudo -n systemctl restart oelala-backend oelala-frontend
```
