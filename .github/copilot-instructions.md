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

## Project Directory Boundaries & Structure

- **External Projects**: Never commit or push inside external projects (e.g. submodules, cloned dependencies); only within first-party projects.
- **Read-Only**: External project directories are always read-only for git actions.
- **Scope**: All commit/push actions are limited to the current project directory and repository.
- **Root Directory Rule**: Project root should contain only README.md and CHANGELOG.md plus standard tool manifests/config.
- **Subdirectories**: All other files must be organized in subdirectories with a narrow and deep tree structure.
- **Todo Location**: Store persistent todos in `docs/TODO_LIST.md`.

## Python / GPU Virtual Environments

- **Canonical GPU venv**: Use `/home/flip/venvs/gpu` (a symlink) as the single canonical environment for GPU/ML work on this server.
- **Do not create per-project GPU venvs**: Avoid new heavyweight `.venv` folders for GPU stacks inside projects; prefer the canonical venv.
- **Archiving old venvs**: When deprecating GPU venvs, move them to `/home/flip/venvs/_archive/YYYY-MM-DD/` and replace the original path with a symlink so scripts keep working.

## Ports

- Prefer the server-wide port inventory in `/home/flip/caramba/docs/PORTS.md` to avoid conflicts.
- Oelala defaults:
    - Frontend dev server: 5174
    - Backend API: 7998

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
