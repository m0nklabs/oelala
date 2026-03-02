# Repository custom instructions (Copilot)

These instructions apply to GitHub Copilot in the context of this repository.

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

## Communication Rules

- **Language**: Communicate with users in **Dutch** when appropriate.
- **Artifacts**: Keep all project artifacts (documentation, code comments, commits) in **English**.
- **Personality**: A bit of "beidehand" (cheeky/witty) humor and enthusiasm is allowed and appreciated. Don't be a boring robot.
- **Conciseness**: Keep responses SHORT and TO THE POINT. No excessive emojis, no verbose explanations with cat/echo commands.
- **No Long Scripts**: NEVER paste long Python scripts directly in terminal with `<< 'EOF'`. Always create a proper `.py` file instead.

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
- **Documentation**: Don’t delete or prune documentation files/directories unless the user explicitly requests it.
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

## Platform Support Policy

- **Supported Platforms**: Windows and Linux only.
- **macOS/iOS/Android**: NOT supported. Do not suggest, implement, or test platform-specific code for these. Do not mention macOS, iOS, or Android in documentation.
- **Cross-platform code**: When writing cross-platform code, only consider Windows and Linux. Use appropriate platform checks.
- **Go builds**: Cross-compile for `GOOS=windows` and `GOOS=linux` only.

## Related Repositories

- **oelala-storage**: Separate Go-based storage service at `/home/flip/oelala-storage/`.
  - **Canonical Docs**: `/home/flip/oelala-storage/docs/VISION.md` (architecture)
  - **Standalone product** - not just an oelala microservice, can be used by any project
  - **Ports**: HTTP API (7990), gRPC Sync (7991), Metrics (7992)
  - **S3-compatible API**: PUT/GET/DELETE/HEAD/LIST objects
  - **Config**: `oelala-storage.yaml`
  - **Build**: `go build -o bin/oelala-storage ./cmd/oelala-storage`
  - **Run**: `./bin/oelala-storage serve`
- When implementing storage features, defer to oelala-storage rather than building in Python.

### oelala-storage Architecture (CLIENT/SERVER/CDN)

```
┌───────────────────────────────────────────────────────────────────┐
│ oelala-backend (Python/FastAPI) - THE BRAIN                       │
│ • User auth, access control, retention policies, tier logic       │
│ • Sends X-Expires-At header for retention                        │
│ • Decides who can see what                                        │
└───────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌───────────────────────────────────────────────────────────────────┐
│ oelala-storage Coordinator - CDN ENTRY POINT                      │
│ • Routes requests to correct node                                 │
│ • Manages replication and deduplication                           │
│ • Database-based blob references (NO SYMLINKS)                    │
└───────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐    ┌──────────┐    ┌──────────┐
        │  Node 1  │    │  Node 2  │    │  Node 3  │
        │  Linux   │    │  Windows │    │  Linux   │
        │  500GB   │    │  200GB   │    │  1TB     │
        └──────────┘    └──────────┘    └──────────┘
```

**Key Principles:**
1. **Storage is "dumb"** - backend tells storage what to do
2. **Deduplication via database** - hash → node locations, NOT symlinks
3. **Retention via headers** - backend sends `X-Expires-At`, storage executes
4. **EU retention** - 6 months minimum (GDPR)

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

- **Hardware**: RTX 5060 Ti 16GB (cuda:0) + RTX 3060 12GB (cuda:1) = 28GB total VRAM
- **⚠️ CRITICAL**: PyTorch indices differ from nvidia-smi! nvidia-smi GPU 0 = PyTorch cuda:1
- **DisTorch2 Allocation**: `cuda:1,11gb;cuda:0,15gb;cpu,*` (OPTIMAL - puts 3060 first!)
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

**Key Discovery**: Allocation order matters! `cuda:1` first makes 3060 hold 97% of model,
leaving 5060 Ti with 15GB free for activations. 5060 Ti runs at 100% utilization.

## Ports

- Prefer the server-wide port inventory in `/home/flip/caramba/docs/PORTS.md` to avoid conflicts.
- Oelala defaults:
    - Frontend dev server: 5174
    - Backend API: 7998
    - ComfyUI: 8188
- oelala-storage:
    - HTTP API: 7990
    - gRPC Sync: 7991
    - Metrics: 7992

## Services

All oelala services run as **systemd services**. NEVER start/stop manually with npm/uvicorn/go commands - always use systemctl!

### Oelala Services

| Service | systemd unit | Port | Restart Command |
|---------|--------------|------|-----------------|
| Backend API | `oelala-backend.service` | 7998 | `sudo systemctl restart oelala-backend` |
| Frontend | `oelala-frontend.service` | 5174 | `sudo systemctl restart oelala-frontend` |
| ComfyUI | `comfyui.service` | 8188 | `sudo systemctl restart comfyui` |

### Oelala-Storage Services

| Service | systemd unit | Port | Restart Command |
|---------|--------------|------|-----------------|
| Storage API | `oelala-storage.service` | 7990 | `sudo systemctl restart oelala-storage` |

### Service Commands Reference

```bash
# Status check
sudo systemctl status oelala-backend oelala-frontend comfyui oelala-storage

# View logs (follow mode)
journalctl -u oelala-backend -f
journalctl -u oelala-frontend -f
journalctl -u comfyui -f
journalctl -u oelala-storage -f

# Restart all oelala services
sudo systemctl restart oelala-backend oelala-frontend

# Check if services are enabled at boot
systemctl is-enabled oelala-backend oelala-frontend comfyui oelala-storage
```

### CRITICAL Rules
- **NEVER run `npm run dev` manually** - use `sudo systemctl restart oelala-frontend`
- **NEVER run `uvicorn` manually** - use `sudo systemctl restart oelala-backend`
- **NEVER run `python main.py` for ComfyUI** - use `sudo systemctl restart comfyui`
- **Dev mode exception**: Only for oelala-storage during active development: `cd /home/flip/oelala-storage && ./bin/oelala-storage serve`
## Cloudflare & CORS Configuration (CRITICAL)

- **Cloudflare Tunnels**: `api.oelala.xyz` → `localhost:7998`, `oelala.xyz` → `localhost:5174`
- **CORS Origins**: Explicit list in `src/backend/app.py` — NEVER use `allow_origins=["*"]` with `allow_credentials=True` (violates CORS spec, browsers reject it)
- **Allowed Origins**:
  ```python
  ["https://oelala.xyz", "http://oelala.xyz", "http://localhost:5174",
   "http://localhost:5173", "http://localhost:3000", "http://192.168.1.26:5174"]
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
> **⚠️ CRITICAL**: PyTorch CUDA indices differ from nvidia-smi! Use PyTorch mapping:

| nvidia-smi | PyTorch | GPU | VRAM | Role |
|------------|---------|-----|------|------|
| 1 | **cuda:0** | RTX 5060 Ti | 16GB | Primary (compute) |
| 0 | **cuda:1** | RTX 3060 | 12GB | Secondary (donor) |
| - | **Total** | - | **28GB** | |

### DisTorch2 Allocation String
```
cuda:1,11gb;cuda:0,15gb;cpu,*
```
- cuda:1 = 3060 (FIRST - receives 97% of model weights)
- cuda:0 = 5060 Ti (compute device, 15GB free for activations)
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
2. Use allocation: `cuda:1,11gb;cuda:0,15gb;cpu,*` (3060 FIRST!)
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

## Technical Stack Reference

**Infer from codebase.**
- Check `requirements.txt`, `pyproject.toml`, `package.json`, `CMakeLists.txt`, etc.
- Follow the versions and libraries specified in the configuration files.
