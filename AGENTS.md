# AGENTS.md — OELALA

> Canonical AI-agent context for this repo. Read first.
> Claude Code: `CLAUDE.md` → here. Goose: `.goosehints` → here. Copilot: `.github/copilot-instructions.md` (Copilot-native, references this).
> Sync to tool-natives with `scripts/sync-agent-docs.sh`. This file is the source of truth.

## What this is

Hybrid AI-media platform (image/video/audio generation, NSFW-capable). **Actively developed**
despite root `README.md` claiming "Deprecated Repository" — that README is a **deliberate privacy
decoy** for the public GitHub repo. Do NOT "fix" or expand it. Internal `docs/*` are the real
source of truth.

## Stack

- **Backend** — Python 3 + FastAPI in `src/backend/` (entry `app.py`, port 7998). Orchestration in
  `comfyui_client.py`, `runpod_client.py`, `job_queue.py`. Deps pinned in `requirements.txt`
  (torch 2.12, transformers 5.8).
- **Frontend** — React 18 + Vite 7, **JSX only** (67 `.jsx`, 0 `.tsx`) in `src/frontend/src/`
  (port 5174). Supabase, Sentry, Recharts. Always use `apiFetch()` from `utils/api.ts`, never raw `fetch()`.
- **Execution / compute** — modular *compute backends*, each a configurable source that can run
  certain model families (see `src/backend/generation/compute_backends.py` + `compute_backends.json`):
  - **ai-kvm2 local ComfyUI** — `localhost:8188` (default client, workflows in `workflows/`,
    models in `ComfyUI/models/`), DisTorch2 multi-GPU alloc `cuda:0,10gb;cuda:1,15gb;cpu,*` with
    **RTX 3060 first**. Managed by systemd `comfyui`; **`always_on`** (monifuse must not idle-stop it).
  - **Windows-PC ComfyUI** — `192.168.1.245:8188` (second server, hosts local MiniMax-H3). Configured
    via `COMFYUI_WINDOWS_HOST`/`COMFYUI_WINDOWS_PORT` in `.env`; accessed through
    `get_windows_comfyui_client()`.
  - **RunPod cloud** — headless = a container with an ephemeral ComfyUI server (Wan2.2, LTX-2.3,
    MiniMax-H3, Qwen I2I). Submit via `submit_to_runpod_fn`.
  Adapters live in `src/backend/generation/adapters/{cloud,local}/`; the registry + router resolve
  the enabled backend per request by model-family capability.
- **Storage** — MinIO (S3). Canonical dirs: `media/generated/`, `ComfyUI/output/`, `uploads/`.
- **DB / monetization** — Supabase/PostgreSQL; Stripe credits.
- **Tests / lint** — pytest (`tests/`, GPU in `tests/gpu/` on self-hosted runner `oelala-gpu`); ruff on `src/`.

## Critical rules

1. **Changelog**: every PR adds a fragment in `changelog/{NNN}-{desc}.md` (`### Added/Fixed/Changed`).
   NEVER edit root `CHANGELOG.md` directly — `scripts/merge_changelog.py` merges on release.
2. **Services via systemd only** (`oelala-backend`, `oelala-frontend`, `comfyui`, `minio`). NEVER run
   `npm run dev`, `uvicorn`, or `python main.py` manually.
3. **Frigate HANDS OFF** — security cameras, 24/7. Read-only (`/home/flip/frigate/config/`). Never
   "fix" its ffmpeg GPU usage.
4. **Media storage**: use only canonical dirs (see Stack). Never create new output dirs. Follow
   naming conventions (`oelala_t2v_{date}_{time}_{idx}.mp4`, etc.).
5. **Secrets**: never commit keys/credentials; use env vars + `.env.example`. Don't log sensitive values.
6. **Language**: artifacts (docs/comments/commits) in **English**; chat with the user in **Dutch**.
7. **Log integrity**: logs are historical — never rewrite/delete/trim old entries; only append.
8. **Docs**: don't delete/prune docs. Keep `docs/GENERATION_MODES_TREE.md` current after successful
   generation runs. LoRA metadata lives in `docs/lora_registry.yaml`.
9. **Tests**: run relevant lint/tests before claiming fixed; add/extend tests for changed behavior.
   GPU-affecting work adds tests under `tests/gpu/`.
10. **GPU venv**: use canonical `/home/flip/venvs/gpu` (symlink); don't create heavy project `.venv`s.
11. **Scope**: smallest correct change; follow existing patterns; don't reformat unrelated files.
12. **Loop device / debug code**: add debug logging with emoji prefixes (🐛🔍⚠️❌✅) and timestamps
    on every log line.

## Skills

When touching an area, read its detail-skill:
- Changelog fragments → `@docs/skills/changelog.md`
- ComfyUI workflows / generation modes → `@docs/skills/comfyui-generation.md`

## Available agents

- No repo-local Copilot agents — removed by design (see `.github/COPILOT_WORKFLOW.md` for the
  issue/PR automation flow and `.github/instructions/*.instructions.md` for domain scoping).

## References

- Detailed operational rules: `@.github/copilot-instructions.md` (Copilot-native, source of the
  critical rules above; also covers MinIO, Cloudflare/CORS, RunPod endpoints, DisTorch2).
- Architecture: `docs/ARCHITECTURE.md`; model inventory: `docs/COMFYUI_INVENTORY.md`.
- Compute backends inventory: `src/backend/generation/compute_backends.py` + `compute_backends.json`; admin UI under the Admin panel → "Compute".
- Todo list: `docs/TODO_LIST.md`.

## Maintenance

- This **AGENTS.md** is the single source of truth. `CLAUDE.md` and `.goosehints` are **synced
  copies** (not symlinks — repo may be cloned on Windows). After changing here, run
  `scripts/sync-agent-docs.sh`. A pre-commit hook asserts they stay in sync.
