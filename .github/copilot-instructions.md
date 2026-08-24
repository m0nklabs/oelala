<!-- See AGENTS.md (repo root) for the canonical source of truth. This file is Copilot-native. -->
# Repository custom instructions (Copilot)

These instructions apply to GitHub Copilot in the context of this repository. This file is the
detailed Copilot-native ruleset; `AGENTS.md` (repo root) is the canonical overview and is
referenced by `CLAUDE.md` and `.goosehints` via `scripts/sync-agent-docs.sh`.

## Primary goals

- Make the smallest correct change that satisfies the request.
- Keep the repo buildable/testable; don’t break CI.
- Prefer clarity and correctness over cleverness.

## General Work Methodology & Agent Behavior

- **Execute, don't ask**: If you can run a command, create a file, or perform an action — do it immediately. Never ask the user to run something you can execute yourself.
- **Minimize back-and-forth**: Complete tasks in one pass when possible. Don't stop to ask for confirmation on routine operations.
- **Fix errors yourself**: If a command fails, debug and retry before asking the user for help.
- **Reuse-First Rule**: ALWAYS search online first (GitHub/web) to see if the idea/feature is already implemented somewhere we can reuse or adapt, then search the local repo/workspace before writing new code.
- **Autonomous Testing Rule**: Do not ask the user to perform tests or run commands that the agent can execute autonomously with available tools. Only involve the user when a physical action or inaccessible secret is strictly required.
- **Best Option Rule**: The agent always chooses the best option and executes it directly, without waiting for permission, input, or confirmation.
- **NEVER do manual workarounds when automating**: If we are building automation for something, NEVER fall back to doing it manually "just this once". Fix the automation instead.
- **NEVER approve PRs manually**: Do not run `gh pr review --approve` or any approval command unless the user explicitly requests it.
- **NEVER rebase Copilot branches manually**: Unless explicitly requested. Let the automated workflows or Copilot handle rebases.
- **Use ALL available resources**: The agent MUST actively use every resource the user has made available (SSH access, API endpoints, CLI tools, tunnels, services, MCP tools, etc.). Never leave a capability unused when it could solve the task faster or better. If unsure whether a resource exists, check first — don't assume it's unavailable.
- **Maintain resource inventory**: The agent MUST keep an up-to-date inventory of all available resources (machines, SSH access, tunnels, API keys, services, tools) in the relevant instruction files. When a new resource is provisioned or discovered, document it immediately.
- **Context refresh**: Periodically re-read this instruction file (`.github/copilot-instructions.md`) and the active agent file (`.github/agents/MARK1.md`) during long sessions. Context drifts — re-reading prevents mistakes. Do this at minimum: (1) at session start, (2) before starting a new major task, (3) whenever unsure about conventions or rules. Also check `/memories/` for persistent context from previous sessions.
- **Memory sync**: Local Copilot memory is convenience context only and is not GitHub-synced. Durable repo rules or lessons that should survive machines/workspaces must be mirrored into tracked files here and into `/home/flip/github-copilot-config/` when they are cross-project.

## Communication Rules

- **Language**: Communicate with users in **Dutch** when appropriate.
- **Artifacts**: Keep all project artifacts (documentation, code comments, commits) in **English**.
- **Personality**: A bit of "beidehand" (cheeky/witty) humor and enthusiasm is allowed and appreciated. Don't be a boring robot.
- **Conciseness**: Keep responses SHORT and TO THE POINT. No excessive emojis, no verbose explanations natively.
- 🚨 **MANDATORY TOOL USAGE**: NEVER use `cat`, `sed`, `grep`, `echo`, or any terminal redirections (`>`/`>>`) to read or edit files in the workspace. YOU MUST ONLY use the built-in MCP editor tools (`read_file`, `create_file`, `replace_string_in_file`, etc). Terminal-based text reading and editing is STRICTLY FORBIDDEN.
- 🚨 **TEMPORARY FILES**: NEVER create temporary files, scripts, or experimental code outside the active workspace directory (like directly in `~/` or `/home/flip/`). Keep the file system clean. All work must happen within the project.

## User preferences (skeleton)

- When the user asks for a "skelet" (scaffolding), prefer a **as complete as practical** skeleton (types + interfaces + DB schema) over a minimal one, as long as it stays within the project scope and does not introduce risky or production-ready features by default without safeguards.

## Project assumptions

- **Detect Context**: Infer the project name, domain, and tech stack from the current codebase (e.g., `package.json`, `requirements.txt`, `README.md`).
- **Documentation**: If the repo is missing documentation (README, build steps), ask the user for the intended stack before introducing major scaffolding.

## Engineering rules

- **Consistency**: Follow existing patterns in the repo. If a pattern exists, reuse it.
- **Dependencies**: Avoid adding dependencies unless they are clearly justified; mention any new dependency explicitly.
- **Scope**: Don’t introduce new features beyond what is requested.
- **Focus**: Keep changes focused; do not reformat unrelated files.
- **Documentation**: Don't delete or prune documentation files/directories unless the user explicitly requests it.
- **Log Integrity**: NEVER delete or modify existing log entries (event logs, grow logs, deadline logs, snooze records, etc.). Logs are historical records — they happened. You may MOVE logs to a better centralized location, but the original text must be preserved verbatim. Errors, outdated entries, and entries written by other agents/LLMs stay exactly as written. Only ADD new entries; never rewrite, summarize, or remove old ones.
- **Scratchpad**: Treat directories like `research/` or `scratch/` as local-only scratch space and keep them out of git via `.gitignore`.
- **Canonical Requirements**: Canonical requirements must be written into `docs/*`.
- **CHANGELOG Required**: Every PR MUST add a changelog fragment file in `changelog/` directory (NOT direct CHANGELOG.md edits). This prevents merge conflicts.
  - File naming: `changelog/{PR_NUMBER}-{short-description}.md` (e.g., `changelog/83-websocket-progress.md`)
  - Use sections: `### Added`, `### Fixed`, `### Changed`, etc.
  - Fragments are auto-merged into CHANGELOG.md on release via `scripts/merge_changelog.py`

## Project Directory Boundaries & Structure

- **External Projects**: Never commit or push inside external projects (e.g. submodules, cloned dependencies); only within first-party projects.
- **Read-Only**: External project directories are always read-only for git actions.
- **Scope**: All commit/push actions are limited to the current project directory and repository.
- **Root Directory Rule**: Project root should contain only README.md and CHANGELOG.md plus standard tool manifests/config.
- **Subdirectories**: All other files must be organized in subdirectories with a narrow and deep tree structure.
- **Todo Location**: Store persistent todos in `docs/TODO_LIST.md`.

## Frigate (HANDS OFF — CRITICAL)

- **NEVER modify Frigate configuration files** (`/home/flip/frigate/config/config.yml`, `/home/flip/frigate/config/docker/docker-compose.gpu.yml`, or any other Frigate config).
- **NEVER restart, stop, or reconfigure the Frigate Docker container** unless the user EXPLICITLY requests it with full awareness.
- **Frigate runs 24/7 security cameras** — breaking it means cameras go down. This is unacceptable.
- **ffmpeg GPU usage is expected** — Frigate's ffmpeg processes use ~1GB on the GPU for hardware-accelerated video decoding. This is normal and must be accounted for in VRAM budgets, NOT "fixed" by modifying Frigate.
- **If VRAM is tight**: Adjust the ComfyUI/oelala side (model allocations, resolution caps, frame limits), NEVER touch Frigate.
- **Frigate config location**: `/home/flip/frigate/config/` — treat this entire directory as READ-ONLY.

## Platform Support Policy

- **Supported Platforms**: Windows and Linux only.
- **macOS/iOS/Android**: NOT supported. Do not suggest, implement, or test platform-specific code for these. Do not mention macOS, iOS, or Android in documentation.
- **Cross-platform code**: When writing cross-platform code, only consider Windows and Linux. Use appropriate platform checks.
- **Go builds**: Cross-compile for `GOOS=windows` and `GOOS=linux` only.

## Storage: MinIO (replaced oelala-storage as of 2026-04-15)

### Node 1 — ai-kvm2
- **MinIO S3 API**: `http://localhost:9000` (systemd: `minio.service`)
- **MinIO Console**: `http://localhost:9001`
- **Data directory**: `/home/flip/minio-data`
- **Cloudflare**: `storage.oelala.xyz` → `:9000`, `storage-main.oelala.xyz` → `:9001`
- **mc alias**: `mc alias set oelala http://localhost:9000 $ACCESS $SECRET`

### Node 2 — ubuntu-oelalastorage2
- **MinIO S3 API**: `http://localhost:9000` (systemd: `minio.service`)
- **Cloudflare**: `storage2.oelala.xyz` → `:9000`
- **mc binary**: `mcli` (not `mc`), alias `local`

### Common
- **Buckets**: `oelala-generated`, `oelala-comfyui` (public read), `oelala-avatars`, `oelala-users` (private)
- **Backend access**: via `minio` Python SDK in `src/backend/storage_client.py`
- **CDN distribution**: `STORAGE_PUBLIC_NODES=https://storage.oelala.xyz,https://storage2.oelala.xyz` in `.env` — public buckets served via round-robin across both nodes
- **Env file**: `/etc/default/minio` (credentials + data dir, both nodes)
- **Override**: `/etc/systemd/system/minio.service.d/override.conf` (runs as `flip`, both nodes)

### Backup
- **Node 2 mirror**: `scripts/minio-backup-mirror.sh` runs every 15 min via cron (`mc mirror`)
- **B2 offsite backup**: Same script mirrors all 4 buckets to Backblaze B2 `oelala-media-eu`
- **mc alias**: `mc alias set b2 https://s3.eu-central-003.backblazeb2.com $KEY_ID $APP_KEY`

### Legacy (DEPRECATED — DO NOT USE)
- `oelala-storage` Go service: stopped + disabled (`oelala-storage.service`, `oelala-node-01.service`)
- The Go-based storage service and its multi-node CDN architecture is no longer in use
- Data was migrated to MinIO on 2026-04-15 (1.1 GiB, 1,059 objects)

## Media Storage Locations (CRITICAL)

**DO NOT create new output directories.** Use ONLY these canonical locations:

| Purpose | Path | Served via |
|---------|------|------------|
| Generated videos/images (primary) | `/home/flip/oelala/media/generated/` | `/media/generated/{filename}` |
| ComfyUI direct output | `/home/flip/oelala/ComfyUI/output/` | `/comfyui/output/{filename}` |
| User uploads (temporary) | `/home/flip/oelala/uploads/` | N/A (processed only) |
| Example/demo media | `/home/flip/oelala/examples/` | Static files |

### Rules

1. **NEVER create new output directories** like `/mnt/ssd/comfyui_output/`, `/tmp/output/`, or any other location.
2. **ComfyUI workflows** must use `filename_prefix` that writes to `ComfyUI/output/` (default) or configure VHS_VideoCombine to use `media/generated/`.
3. **Backend API** saves generated media to `media/generated/` with consistent naming: `{type}_{timestamp}_{index}.{ext}`.
4. **Legacy files** on `/mnt/ssd/comfyui_output/` should be migrated to `media/generated/`, not used as a new standard location.
5. When in doubt, use `media/generated/` - it's the canonical location for all AI-generated content.

### Naming Conventions

- Text-to-Video: `oelala_t2v_{YYYYMMDD}_{HHMMSS}_{index}.mp4`
- Image-to-Video: `oelala_i2v_{YYYYMMDD}_{HHMMSS}_{index}.mp4`
- Text-to-Image: `oelala_t2i_{index}_.png`
- LTX-2 Audio: `ltx2_audio_{index}.mp4` (with `-audio` suffix for audio version)
- Benchmarks: `bench_{resolution}_{duration}_{vram}_{index}.mp4`

## External Storage (SSD) - Models Only

The SSD at `/mnt/ssd/` is used **exclusively for large model files** due to disk space constraints. Do NOT use it for generated output.

| Purpose | Path |
|---------|------|
| LoRA models | `/mnt/ssd/loras/` |
| Large checkpoints (overflow) | `/mnt/ssd/checkpoints/` |
| HuggingFace cache | `/mnt/ssd/huggingface/` |

**Primary model locations remain in workspace:**
- ComfyUI models: `/home/flip/oelala/ComfyUI/models/`
- Checkpoints: `ComfyUI/models/checkpoints/`
- VAEs: `ComfyUI/models/vae/`
- Text encoders: `ComfyUI/models/text_encoders/`
- GGUF models: `ComfyUI/models/diffusion_models/`

## Key File Locations Reference

| What | Where |
|------|-------|
| Workflows (API format) | `/home/flip/oelala/workflows/` |
| Backend API | `/home/flip/oelala/src/backend/` |
| Frontend | `/home/flip/oelala/src/frontend/` |
| ComfyUI custom nodes | `/home/flip/oelala/ComfyUI/custom_nodes/` |
| Test files | `/home/flip/oelala/tests/` |
| Documentation | `/home/flip/oelala/docs/` |
| GPU venv | `/home/flip/venvs/gpu` (symlink) |
| ComfyUI logs | `journalctl -u comfyui` |
| Backend logs | `journalctl -u oelala-api` |

## Python / GPU Virtual Environments

- **Canonical GPU venv**: Use `/home/flip/venvs/gpu` (a symlink) as the single canonical environment for GPU/ML work on this server.
- **Do not create per-project GPU venvs**: Avoid new heavyweight `.venv` folders for GPU stacks inside projects; prefer the canonical venv.
- **Archiving old venvs**: When deprecating GPU venvs, move them to `/home/flip/venvs/_archive/YYYY-MM-DD/` and replace the original path with a symlink so scripts keep working.

## GPU Configuration & DisTorch2

- **Hardware**: RTX 5060 Ti 16GB (cuda:1) + RTX 3060 12GB (cuda:0) = 28GB total VRAM
- **GPU ordering pinned**: `CUDA_DEVICE_ORDER=PCI_BUS_ID` set system-wide (`/etc/environment`), nvidia-smi and PyTorch indices are identical
- **DisTorch2 Allocation**: `cuda:0,10gb;cuda:1,15gb;cpu,*` (OPTIMAL - puts 3060 first!)
- **Model Inventory**: See `docs/COMFYUI_INVENTORY.md` for complete list of available models and VRAM limits
- **DisTorch2 Guide**: See `docs/DISTORCH2_MULTI_GPU_SETTINGS.md` for comprehensive configuration
- **DisTorch2 Nodes** for multi-GPU video generation:
  - `UnetLoaderGGUFAdvancedDisTorch2MultiGPU` - GGUF model loading with GPU distribution
  - `VAELoaderDisTorch2MultiGPU` - VAE loading with GPU distribution
  - `CLIPLoaderDisTorch2MultiGPU` - T5 text encoder loading with GPU distribution
- **SageAttention**: Always use `PathchSageAttentionKJ` node to reduce VRAM by 15-20%
- **VRAM Limits (Tested 2026-01-16)**:
  - 480×848 @ 321 frames: ~26GB (SAFE MAX, ~20 sec video)
  - 576×1024 @ 81 frames: ~24GB (GPU-only, safe)
  - 720×1280 @ 41 frames: ~27GB (tight)

**Key Discovery**: Allocation order matters! `cuda:0` first makes 3060 hold 97% of model,
leaving 5060 Ti with 15GB free for activations. 5060 Ti runs at 100% utilization.

## Ports

- Prefer the server-wide port inventory in `/home/flip/caramba/docs/PORTS.md` to avoid conflicts.
- Oelala defaults:
    - Frontend dev server: 5174
    - Backend API: 7998
    - ComfyUI: 8188

## Services

All oelala services run as **systemd services**. NEVER start/stop manually with npm/uvicorn/go commands - always use systemctl!

### Oelala Services

| Service | systemd unit | Port | Restart Command |
|---------|--------------|------|-----------------|
| Backend API | `oelala-backend.service` | 7998 | `sudo systemctl restart oelala-backend` |
| Frontend | `oelala-frontend.service` | 5174 | `sudo systemctl restart oelala-frontend` |
| ComfyUI | `comfyui.service` | 8188 | `sudo systemctl restart comfyui` |

### Service Commands Reference

```bash
# Status check
sudo systemctl status oelala-backend oelala-frontend comfyui minio

# View logs (follow mode)
journalctl -u oelala-backend -f
journalctl -u oelala-frontend -f
journalctl -u comfyui -f
journalctl -u minio -f

# Restart all oelala services
sudo systemctl restart oelala-backend oelala-frontend

# Check if services are enabled at boot
systemctl is-enabled oelala-backend oelala-frontend comfyui minio
```

### CRITICAL Rules
- **NEVER run `npm run dev` manually** - use `sudo systemctl restart oelala-frontend`
- **NEVER run `uvicorn` manually** - use `sudo systemctl restart oelala-backend`
- **NEVER run `python main.py` for ComfyUI** - use `sudo systemctl restart comfyui`
## Cloudflare & CORS Configuration (CRITICAL)

### Tunnel Inventory

| Tunnel | ID | Machine | Config |
|--------|----|---------|--------|
| `oelala-main` | `b34ce27b-e9b1-4926-b5fe-ebbaf42d506a` | ai-kvm2 (LAN) | `/etc/cloudflared/config.yml` |
| `oelala-storage-node2` | `83d253c4-24eb-4643-b36f-174a2fc3f10b` | ubuntu-oelalastorage2 (LAN) | `/etc/cloudflared/config.yml` |

### DNS → Tunnel Routing

| Hostname | Tunnel | Target Service |
|----------|--------|----------------|
| `oelala.xyz` | oelala-main | `http://localhost:5174` (frontend) |
| `api.oelala.xyz` | oelala-main | `http://localhost:7998` (backend) |
| `storage.oelala.xyz` | oelala-main | `http://localhost:9000` (MinIO S3 API) |
| `storage-main.oelala.xyz` | oelala-main | `http://localhost:9001` (MinIO Console) |
| `storage2.oelala.xyz` | oelala-storage-node2 | `http://localhost:9000` (MinIO S3 API) |
| `pgdb.oelala.xyz` | oelala-main | `tcp://localhost:5432` (PostgreSQL) |

### Tunnel Management

- **Cert location (ai-kvm2)**: `/home/flip/.cloudflared/cert.pem`
- **Create DNS record**: `TUNNEL_ORIGIN_CERT=/home/flip/.cloudflared/cert.pem cloudflared tunnel route dns <tunnel-id> <hostname>`
- **Each node runs its own tunnel** — independent, decentralized. No node depends on another for Cloudflare connectivity.
- **Node 2 is fully autonomous** — has its own cloudflared install, tunnel, credentials, and systemd service.

### CORS Configuration
- **CORS Origins**: Explicit list in `src/backend/app.py` — NEVER use `allow_origins=["*"]` with `allow_credentials=True` (violates CORS spec, browsers reject it)
- **Allowed Origins**:
  ```python
  ["https://oelala.xyz", "http://oelala.xyz", "http://localhost:5174",
   "http://localhost:5173", "http://localhost:3000", "http://localhost:5174"]
  ```
- **Cloudflare Cache Gotcha**: CF caches responses WITHOUT `Vary: Origin`, so the first CORS response gets served to all origins. Fix: add `Vary: Origin` header to all CORS-sensitive endpoints.
- **Static File CORS**: The `/comfyui/output/` endpoint has explicit CORS headers + `Vary: Origin` because CF caches aggressively.
- **Cache Busting**: Frontend uses `?_cors=1` query param to bypass stale CF cache when fetching images cross-origin.
- **Full Details**: See `docs/CLOUDFLARE_SETUP.md` for comprehensive Cloudflare + CORS documentation.

## I2I Face Processing Pipeline

- **Dynamic Workflow**: `_build_i2i_workflow()` in `comfyui_client.py` builds I2I ComfyUI workflows dynamically based on enabled features.
- **Face Features** (all optional, toggled per-request):
  - **IP-Adapter FaceID Plus V2**: Transfers face identity from source to generation (strength 0.0-1.0)
  - **FaceDetailer**: Detects + refines faces using face_yolov8m + SAM (denoise 0.0-1.0)
  - **GFPGAN Face Restore**: Restores/enhances facial details post-generation (via mtb nodes)
- **Presets**: `/i2i/presets` endpoint returns named presets (e.g., "Portrait", "Character", "Stylized")
- **Models Required** (see `docs/COMFYUI_INVENTORY.md`):
  - `ip-adapter-faceid-plusv2_sdxl.bin`
  - `ip-adapter-faceid-plusv2_sdxl_lora.safetensors`
  - `CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors`
  - `buffalo_l/` (InsightFace analyzer)
  - `face_yolov8m.pt`, `sam_vit_b_01ec64.pth`
  - `GFPGANv1.4.pth`
## Safety & secrets

- **Secrets**: Never commit secrets (API keys, credentials, private keys). Use environment variables and `.env.example` only.
- **Logging**: Don’t log sensitive values.
- **Local Config**: Don’t delete or rewrite existing local secret files unless explicitly requested; prefer hardening via `.gitignore` and templates.
- **Safety Defaults**: If adding logic with side effects (e.g., API calls, money movement, deletions), default to **dry-run / safe-mode** unless the user explicitly requests live execution.

## Validation & Testing

- Always run the most relevant tests/lint/build checks that exist in the repo.
- If no tests exist for changed behavior and the repo has a test framework, add/extend tests.
- Prefer fast, targeted test runs first; then broader checks if available.
- **Testing Requirements**:
    1. Add unit tests matching the module path.
    2. Use the existing testing framework (e.g., `pytest`, `jest`, `vitest`).
    3. Mock external services (APIs, databases) in unit tests.
    4. Minimum coverage: Aim for high coverage (e.g., 80%) for new code.

## GPU Integration Testing (Self-Hosted Runner)

This repository has a **self-hosted GPU runner** (`oelala-gpu`) with direct access to:
- ComfyUI API at `localhost:8188`
- Backend API at `localhost:7998`
- GPU (RTX 5060 Ti 16GB + RTX 3060 12GB)
- All installed models in `/home/flip/oelala/ComfyUI/models/`

**When working on video/ComfyUI related code:**

1. **Add tests to `tests/gpu/`** - These run on the self-hosted runner with real GPU access.
2. **Trigger GPU tests via workflow** - The `gpu-tests.yml` workflow runs on PRs touching:
   - `src/backend/comfyui_client.py`
   - `src/backend/app.py`
   - `workflows/**`
   - `tests/gpu/**`
3. **Test actual workflow execution** - You can queue real ComfyUI workflows:
   ```python
   # In tests/gpu/test_*.py
   resp = requests.post("http://localhost:8188/prompt", json={"prompt": workflow})
   ```
4. **Use available models only** - Check `/api/models/checkpoints` for available models.
   Do NOT use `realvisxlV50_v50Bakedvae.safetensors` (removed).
5. **Python environment** - Use `/home/flip/venvs/gpu` for GPU-enabled Python.

## Multi-GPU & DisTorch2 Configuration

The server uses **DisTorch2** for multi-GPU model distribution across both GPUs:

### Hardware
> **GPU ordering pinned** via `CUDA_DEVICE_ORDER=PCI_BUS_ID` (system-wide in `/etc/environment`).
> nvidia-smi index = PyTorch cuda index. Stable across reboots.

| Index | GPU | VRAM | Role |
|-------|-----|------|------|
| cuda:0 | RTX 3060 | 12GB | Secondary (donor) |
| cuda:1 | RTX 5060 Ti | 16GB | Primary (compute) |
| Total | - | **28GB** | |

### DisTorch2 Allocation String
```
cuda:0,10gb;cuda:1,15gb;cpu,*
```
- cuda:0 = 3060 (FIRST - receives 97% of model weights)
- cuda:1 = 5060 Ti (compute device, 15GB free for activations)
- cpu,* = safety fallback (rarely needed)
- Used in UnetLoaderGGUFAdvancedDisTorch2MultiGPU, VAELoaderDisTorch2MultiGPU, CLIPLoaderDisTorch2MultiGPU

**⚠️ ORDER MATTERS!** First device in string gets model first. Put 3060 first!

### Video Generation Limits (WAN 2.2 14B Q6_K, tested 2026-01-16)
| Resolution | Max Frames | Video Length | VRAM | Notes |
|------------|------------|--------------|------|-------|
| 480×848 | **321** | ~20 sec | ~26GB | SAFE production max |
| 480×848 | 341-350 | ~21 sec | ~27GB | Tight, works |
| 576×1024 | 81-121 | ~5-8 sec | ~24-27GB | Standard quality |
| 720×1280 | 41-61 | ~2.5-4 sec | ~27GB | High quality |

### Generation Times (6 steps, uni_pc sampler)
| Frames | Time/Step | Total Time |
|--------|-----------|------------|
| 81 | ~50-60s | ~5-6 min |
| 161 | ~110-120s | ~12 min |
| 321 | ~227s | ~23 min |

### Key Files
- `docs/DISTORCH2_MULTI_GPU_SETTINGS.md` - **COMPREHENSIVE DisTorch2 guide** ⭐
- `docs/COMFYUI_INVENTORY.md` - Complete inventory of all models, LoRAs, custom nodes
- `docs/GENERATION_MODES_TREE.md` - **🌳 HOLY TREE - Visual tree of ALL tested generation modes**
- `ComfyUI/custom_nodes/ComfyUI-MultiGPU/distorch_2.py` - DisTorch2 with local fixes
- `workflows/ImageToVideo/WAN22-I2V-DISTORCH2-LATEST-api.json` - Optimal DisTorch2 API workflow

### When modifying video workflows:
1. Always use DisTorch2 loader nodes for Wan2.2
2. Use allocation: `cuda:0,10gb;cuda:1,15gb;cpu,*` (3060 FIRST!)
3. Include `expert_mode_allocations` on ALL loader nodes
4. Test with target resolution before production
5. Check ComfyUI logs for distribution: `[MultiGPU DisTorch V2]`
6. **Consult `docs/DISTORCH2_MULTI_GPU_SETTINGS.md`** for optimal settings

## Generation Mode Documentation (MANDATORY)

> **⚠️ CRITICAL: Document ALL successful generation tests!**

After ANY successful ComfyUI generation run or new model/workflow test:

1. **Update `docs/GENERATION_MODES_TREE.md`** - Add to "Tested Configurations Log"
2. **Include**: Tool type, mode, resolution, frames, VRAM used, workflow file, result
3. **Add new modes** to appropriate tool section with full sub-model tree
4. **Update status** (🔨→✅) when a mode becomes production-ready

**Why this matters:**
- Research/testing results MUST be preserved
- Prevents re-discovering what already works
- Single source of truth for "what works with what settings"
- Future agents/sessions can rely on documented configurations

Example log entry:
```
│ 2026-01-12 | I2V | wan22 standard                                   │
│   Resolution: 576x1024 | Frames: 81 | VRAM: ~24GB                   │
│   Workflow: wan22_i2v_distorch2_api.json                            │
│   Result: ✅ SUCCESS                                                 │
```

## Debug Code Requirements

When implementing any feature or component:
1. **Always Include Debug Logging**: Add comprehensive debug output throughout all code.
2. **Global Debug Control**: Implement a DEBUG flag that controls debug output.
3. **Clear Formatting**: Use emoji prefixes for easy scanning (🐛, 🔍, ⚠️, ❌, ✅).
4. **Timestamps Required**: ALL log files MUST include timestamps on every line. Use `%(asctime)s` in Python logging formatters or ISO 8601 format. Never write logs without timestamps — logfiles without timestamps are useless for debugging.

## Git Workflow & Commit Standards

- **Direct Push**: If the user explicitly asks to commit and push changes to GitHub, push directly to the default branch in this repository (no PR/feature branch) unless the user asks otherwise.
- **Per-File Commit Comments**: When making changes to individual files, always create specific git commit messages that describe the exact changes made to that file.
- **Granular Commits**: Prefer smaller, focused commits with clear descriptions over large commits with generic messages.
- **Descriptive Messages**: Each commit message should explain what was changed, why it was changed, and the impact of the change.

## GitHub Issue/PR Work Policy

- **Claim First**: Before starting work on any GitHub issue or pull request, ALWAYS claim it first (self-assign, add comment).
- **Copilot Agent**: To activate the Copilot Coding Agent on an issue or PR, you **must** mention `@copilot` in a comment.
- **Workflow Approval**: DO NOT suggest changing GitHub Actions settings for first-time contributor approval.
- **TODO Checkbox Updates**: When completing PR work, ALWAYS check off completed items in the issue's TODO checklist. This is a priority action.

## Task Management with Todo Lists

All Copilot-style agents **MUST** use structured todo lists for planning, tracking, and executing complex multi-step tasks.

### Workflow
1.  **Check `CHANGELOG.md`**: Understand what has already been implemented.
2.  **Plan Tasks**: Write a complete todo list with specific, actionable items before starting.
3.  **Mark In-Progress**: Set **ONE** todo to `in-progress` before working on it.
4.  **Execute**: Complete the work for that specific todo.
5.  **Mark Completed**: **IMMEDIATELY** mark the todo as `completed`.
6.  **Repeat**: Move to the next todo and repeat the process.

### Tool Usage
- **`manage_todo_list` Tool**: This tool is **MANDATORY** for managing tasks. It must be updated immediately upon any status change.

## Frontend API Calls (apiFetch)

- **NEVER use raw `fetch()` for backend API calls** — always use `apiFetch()` from `src/frontend/src/utils/api.ts`.
- `apiFetch` automatically adds credentials, auth headers, and CORS cache-busting (`?_cors=1`).
- All 7 tool files (`T2I`, `I2V`, `T2V`, `V2V`, `FaceSwap`, `Upscale`, `I2I`) must use `apiFetch` for image/video fetching.
- When adding a new tool, import and use `apiFetch` — never raw `fetch`.

## CreationsPickerModal (Inline Panel)

- The "From My Creations" picker is an **inline panel** (not a modal overlay) embedded in each tool's form.
- Uses `CreationsPickerModal` component but renders inline with `position: relative` styling.
- Each tool manages its own picker state via `showCreationsPicker` / `setShowCreationsPicker`.

## Reusable Frontend Components

- **InfoTooltip**: `src/frontend/src/components/InfoTooltip.jsx` — hover "?" icon with auto-flip tooltip. Use for ALL parameter explanations across tools. Don't inline help text or create per-tool tooltip implementations.
- **AISuggestPanel**: `src/frontend/src/components/AISuggestPanel.jsx` — sends prompt + current settings to `/ai-suggest` endpoint, returns optimized parameters. Integrate in tool forms where LLM-guided optimization makes sense.
- When adding a new reusable component, place it in `src/frontend/src/components/` and document it here.

## LoRA Registry

- **Canonical metadata**: `docs/lora_registry.yaml` — YAML file with trigger words, recommended strengths, source URLs, tags, base model compatibility.
- **Scanner enrichment**: `src/backend/lora_scanner.py` merges file-system discovery with registry metadata via `LoRARegistry` dataclass.
- **API endpoints**: `/loras/registry` (full registry), `/loras/validate` (check consistency).
- When adding a new LoRA: (1) place the file in `ComfyUI/models/loras/` or `/mnt/ssd/loras/`, (2) add metadata to `docs/lora_registry.yaml`, (3) scanner auto-discovers and enriches.
- **RunPod private LoRAs**: Workers accept `HF_TOKEN` env var to download private LoRAs from Hugging Face during cold start.

## Technical Stack Reference

**Infer from codebase.**
- Check `requirements.txt`, `pyproject.toml`, `package.json`, `CMakeLists.txt`, etc.
- Follow the versions and libraries specified in the configuration files.

## RunPod Serverless (Cloud Wan22 — Wan 2.2)

- **Endpoint**: `x2x496ymkidl3m` ("oelala-wan22")
- **Template**: `tkpy0pi8gt` ("oelala-comfyui-worker"), containerDisk=100GB
- **Image**: `ghcr.io/m0nklabs/oelala-comfyui-worker` (dated tags, e.g. `20260408-135917`)
- **RunPod LoRA Volume**: `ochebt0xbq` (`oelala-runpod-lora-eu-cz`), `EU-CZ-1`, `50GB`
- **GPU Tiers**: `AMPERE_48,ADA_48_PRO,AMPERE_80,ADA_80_PRO,BLACKWELL_96,HOPPER_141,BLACKWELL_180` (48GB+ only)

## RunPod Serverless (LTX-2.3 22B)

- **Endpoint**: `ctpoa610dva4ww` ("oelala-ltx23")
- **Template**: `c1fz26l07d` ("oelala-ltx23-worker"), containerDisk=100GB
- **Image**: `ghcr.io/m0nklabs/oelala-ltx23-worker` (dated tags, e.g. `20260412-102222`)
- **GPU Tiers**: `AMPERE_80,ADA_80_PRO,HOPPER_141,BLACKWELL_96,BLACKWELL_180` (80GB+ only — 22B model needs ~60GB VRAM)
- **Deploy**: `deploy/runpod-ltx23/deploy.sh` (same pattern as Wan worker)
- **Env var**: `RUNPOD_LTX23_ENDPOINT_ID=ctpoa610dva4ww` in `.env`
- **⚠️ CRITICAL**: RunPod `gpuIds` expects architecture-tier IDs (e.g., `AMPERE_48`), NOT model names (e.g., `"NVIDIA RTX 4090"`). API silently accepts wrong names but scheduler never matches them.
- **Full tier reference**: See `docs/RUNPOD_GPU_TIERS.md` for all 11 valid tier IDs.
- **Config**: `workersMin=0`, `workersMax=1`, `idleTimeout=120` (keeps burst traffic warm without pinning a permanent worker)
- **🚨 DEPLOY RULE**: ALWAYS use `deploy/runpod/deploy.sh` to deploy new worker images. NEVER manually `docker push :latest` — RunPod templates use explicit dated tags, not `:latest`. Pushing `:latest` alone means RunPod keeps pulling the old tag and your changes never reach production. This mistake wasted 3 deploys on 2026-04-08.
- **Storage policy**: RunPod Network Volume is for LoRAs and hard-to-replace private/custom assets only. NEVER store general Hugging Face models, general model caches, or broad cold-start optimization payloads there.
- **Population policy**: Upload local rare/private assets to the LoRA volume on demand. Do not prefill it with broad model libraries.
- **Attachment policy**: Keep the LoRA volume detached by default. Attaching it to a serverless endpoint constrains scheduling to `EU-CZ-1`.
- **Account**: `mark.op.mobiel@gmail.com`, spend limit $80

## RunPod Serverless (Cloud I2I — SDXL/Pony/Qwen)

- **Endpoint**: `8djiexluyybooj` ("oelala-i2i")
- **Template**: `ed2614hd8k` ("oelala-i2i-worker"), containerDisk=100GB
- **Image**: `ghcr.io/m0nklabs/oelala-i2i-worker` (dated tags)
- **GPU Tiers**: `AMPERE_48,ADA_48_PRO,AMPERE_80,ADA_80_PRO,BLACKWELL_96,HOPPER_141,BLACKWELL_180` (48GB+)
- **Deploy**: `deploy/runpod-i2i/deploy.sh` (same pattern as Wan/LTX workers)
- **Env vars**: `RUNPOD_I2I_ENDPOINT_ID=8djiexluyybooj`, `RUNPOD_I2I_TEMPLATE_ID=ed2614hd8k` in `.env`
- **Models** (fp8mixed/safetensors downloaded at runtime):
  - UNET: `qwen_image_edit_2511_fp8mixed.safetensors` (19.1 GB)
  - CLIP: `qwen_2.5_vl_7b_fp8_scaled.safetensors` (8.8 GB)
  - VAE: `qwen_image_vae.safetensors` (243 MB)
  - Lightning LoRA: `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors` (850 MB)

## RunPod Serverless (MiniMax-H3 video+audio — t2v / i2v)

- **Endpoint**: `5xuvnvyww4ujnc` ("oelala-minimax-h3")
- **Template**: `fpfo4gmnrw` ("oelala-minimax-h3-worker"), containerDisk=100GB, serverless
- **Image**: `ghcr.io/m0nklabs/oelala-minimax-h3-worker` (dated tags)
- **GPU Tiers**: `AMPERE_80,ADA_80_PRO,HOPPER_141,BLACKWELL_96,BLACKWELL_180` (80GB+; int8/nvfp4 quants may fit 48GB, untested)
- **Deploy**: `deploy/runpod-minimax-h3/deploy.sh` (same pattern as Wan/LTX workers)
- **Models** (~42.5 GB downloaded at runtime from `Comfy-Org/MiniMax-H3`):
  - Diffusion: `minimax_h3_fl2va_pruned_int8_convrot.safetensors` (20.97 GB) — same checkpoint for t2v AND i2v (i2v anchors first-frame keyframe)
  - Text encoder: `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` (15.69 GB, Qwen3-VL-32B, no Blackwell needed)
  - VAEs: `minimax_h3_video_vae_fp16.safetensors` (5.21 GB) + `minimax_h3_audio_vae_fp32.safetensors` (0.61 GB)
- **Audio**: MiniMax-H3 is a joint video+audio DiT — every generation includes a synchronized soundtrack (no separate audio step)
- **ComfyUI**: cloned from official `Comfy-Org/ComfyUI` master (H3 core nodes in `comfy_extras/nodes_minimax_h3.py`); bump `CACHE_DATE` in the Dockerfile to refresh
- **Workflow builders**: `build_cloud_minimax_h3_t2v_workflow` / `build_cloud_minimax_h3_i2v_workflow` in `src/backend/comfyui_client.py` (mirror official template: simple/20-step, res_multistep, BasicGuider — no negative prompt, 24 fps, 17k+5 frame grid)
- **Adapters**: `MiniMaxH3CloudT2VAdapter` / `MiniMaxH3CloudI2VAdapter` (`src/backend/generation/adapters/cloud/minimax_h3_{t2v,i2v}.py`)
- **Config**: `workersMin=0`, `workersMax=2`, `idleTimeout=120`, `QUEUE_DELAY:1`, `executionTimeout=45min`, `ttl=2h` (`minimax_h3` profile in `src/backend/runpod_defaults.py`)

- **Config**: `workersMin=0`, `workersMax=1`, `idleTimeout=120`

## Node Architecture Updates (2026-03-06)
- **oelala-storage architecture**:
  - `storage-main.oelala.xyz` = Coordinator (`oelala-storage.service` on 7990) -> Data: `/home/flip/oelala-main-data`
  - `storage-node-01.oelala.xyz` = Node 01 (`oelala-node-01.service` on 7993) -> Data: `/home/flip/oelala-storage-data`
  - `storage-node-02.oelala.xyz` = Node 02 (on ubuntu-oelalastorage2)
- **Local media directories**: DO NOT USE permanent local folders like `/home/flip/oelala/media/generated/`. Temporary processing uses `/tmp/oelala_generated/` and is immediately unlinked.
