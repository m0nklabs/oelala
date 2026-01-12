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

- **oelala-storage**: Separate Go-based storage service. See `docs/MEDIA_STORAGE.md` for architecture.
- When implementing storage features, defer to oelala-storage rather than building in Python.

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
- **DisTorch2 Allocation**: `cuda:0,11gb;cuda:1,15gb;cpu,2gb` (recommended for all video workflows)
- **Model Inventory**: See `docs/COMFYUI_INVENTORY.md` for complete list of available models and VRAM limits
- **DisTorch2 Nodes** for multi-GPU video generation:
  - `UnetLoaderGGUFAdvancedDisTorch2MultiGPU` - GGUF model loading with GPU distribution
  - `VAELoaderDisTorch2MultiGPU` - VAE loading with GPU distribution
  - `CLIPLoaderDisTorch2MultiGPU` - T5 text encoder loading with GPU distribution
- **SageAttention**: Always use `PathchSageAttentionKJ` node to reduce VRAM by 15-20%
- **VRAM Limits**:
  - 576×1024 @ 81 frames: ~24GB (GPU-only, safe)
  - 720×1280 @ 81 frames: ~27GB (CPU offload required)
  - 720×400 @ 241 frames: ~22GB (GPU-only, 15s video)

## Ports

- Prefer the server-wide port inventory in `/home/flip/caramba/docs/PORTS.md` to avoid conflicts.
- Oelala defaults:
    - Frontend dev server: 5174
    - Backend API: 7998
    - ComfyUI: 8188

## Services

- **Backend API**: Runs as a systemd service `oelala-api.service`. Do NOT start/stop manually with uvicorn commands.
  - Restart: `sudo systemctl restart oelala-api`
  - Logs: `journalctl -u oelala-api -f`
- **ComfyUI**: Runs as a systemd service `comfyui.service`.
  - Restart: `sudo systemctl restart comfyui`
  - Logs: `journalctl -u comfyui -f`
- **Frontend**: Runs via `npm run dev` in development, or as static build in production.

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
| GPU | VRAM | CUDA Device |
|-----|------|-------------|
| RTX 5060 Ti | 16GB | `cuda:1` |
| RTX 3060 | 12GB | `cuda:0` |
| **Total** | **28GB** | |

### DisTorch2 Allocation String
```
cuda:0,12gb;cuda:1,16gb
```
- GPU-only mode (no CPU fallback for model weights)
- Used in UnetLoaderGGUFAdvancedDisTorch2MultiGPU, VAELoaderDisTorch2MultiGPU, CLIPLoaderDisTorch2MultiGPU

### Video Generation Limits (WAN 2.2 14B Q6_K)
| Resolution | Max Frames | Notes |
|------------|------------|-------|
| 480p (848x480) | 81 | Standard quality |
| 720p (1280x720) | 41 | High quality |
| 1080p | 17-25 | May need lower frames |

### Key Files
- `docs/COMFYUI_INVENTORY.md` - **Complete inventory of all models, LoRAs, custom nodes**
- `docs/MULTI_GPU_SETUP.md` - Full multi-GPU configuration guide
- `docs/GENERATION_MODES.md` - Detailed generation mode specifications
- `docs/GENERATION_MODES_TREE.md` - **🌳 HOLY TREE - Visual tree of ALL tested generation modes**
- `ComfyUI/custom_nodes/ComfyUI-MultiGPU/distorch_2.py` - DisTorch2 with local fixes
- `workflows/ImageToVideo/wan22_i2v_distorch2_api.json` - DisTorch2 API workflow

### When modifying video workflows:
1. Always use DisTorch2 loader nodes for Wan2.2
2. Include `expert_mode_allocations` on ALL loader nodes
3. Test with both low (17 frames) and high (81 frames) settings
4. Check ComfyUI logs for distribution: `[MultiGPU DisTorch V2]`
5. **Consult `docs/COMFYUI_INVENTORY.md`** for available models/LoRAs

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

## Technical Stack Reference

**Infer from codebase.**
- Check `requirements.txt`, `pyproject.toml`, `package.json`, `CMakeLists.txt`, etc.
- Follow the versions and libraries specified in the configuration files.
