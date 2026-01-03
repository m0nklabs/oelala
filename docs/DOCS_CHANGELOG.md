Oelala Documentation Change Log
================================

All documentation edits and provenance for the Oelala project.

2026-01-03 | GitHub Copilot
- Major documentation update for dashboard UI v2 and Prompts feature
- Updated ARCHITECTURE.md with current system overview, metadata extraction, frontend navigation
- Updated WEB_INTERFACE_README.md with new features, API endpoints, metadata response format
- Rewrote UI_V2_PLAN.md to reflect implemented status
- Updated PROJECT_OVERVIEW.md with current project status and structure

Why:
- Documentation was outdated after implementing Prompts section and extended metadata extraction.

Files changed:
- `docs/ARCHITECTURE.md` - System diagram, endpoint table, frontend components
- `docs/WEB_INTERFACE_README.md` - Features, API docs, file structure
- `docs/UI_V2_PLAN.md` - Implementation status, completed features
- `docs/PROJECT_OVERVIEW.md` - Project status, directory structure

New features documented:
- Prompts section in My Media with 💬 bubble and popup modal
- Extended metadata extraction (LoRAs, sampler, scheduler, model, resolution)
- Preset selector component
- Video duration in prompt popup

2025-12-27 | GitHub Copilot
- Added UI reference folder for the new dashboard direction.
- Captured the Grok Imagine feature set as a screenshot manifest + filename convention so we can align navigation and panel layout.

Why:
- Keep a stable UX target while we rework the Oelala UI.

Files changed:
- `docs/ui-reference/grok-imagine/README.md`
- `docs/ui-reference/grok-imagine/manifest.json`

Notes:
- Raw images should live in `docs/ui-reference/grok-imagine/raw/` using the documented filenames.

2025-12-27 | GitHub Copilot
- Added a UI v2 plan (feature matrix + IA + MVP scope) to drive the new dashboard implementation.

Why:
- Align UI work with existing backend endpoints and clearly mark missing capabilities.

Files changed:
- `docs/UI_V2_PLAN.md`

2025-11-27 | GitHub Copilot
- Replaced every LAN reference (README files, quick-reference guides, landing page) with the current host `192.168.1.2` so WAN2.2 docs stay accurate after the infrastructure remap.
- Updated example curl commands, service checklists, and contact sheets to keep operators from pointing to the retired `192.168.1.28` address.

Why:
- Maintain a single authoritative IP across runtime docs before wiring in HiDream.

Files changed:
- `WEB_INTERFACE_README.md`, `WAN2_README.md`, `WORKFLOW_QUICK_REFERENCE.md`, `docs/PROJECT_OVERVIEW.md`, `index.html`, and related helper files.

Notes:
- Documentation-only sweep; runtime configs already consumed the updated host constants.

2025-09-07 | GitHub Copilot
- Updated multiple README files to change the project LAN IP from 192.168.1.27 -> 192.168.1.2.
- added 'Network and LAN notes' to `WAN2_README.md` and a short network convention entry to `PROJECT_PLAN.md`.
- Rewrote example API usage URLs in `WEB_INTERFACE_README.md` to use 192.168.1.2.
- added a `KEYWORDS.md` network tag (`oelala-lan:192.168.1.2`) for discoverability.

Why:
- Align local documentation with the project's LAN rules (use 192.168.1.2).
- Ensure examples, quickstart steps, and network notes are consistent and actionable.

Files changed:
- `WEB_INTERFACE_README.md` (URLs, network section, recent edits note)
- `WAN2_README.md` (added network notes)
- `PROJECT_PLAN.md` (added network convention)
- `KEYWORDS.md` (added network tag)

Agent / Editor:
- GitHub Copilot (agent tag as requested)

Notes:
- These edits are documentation-only. No code behavior was changed.
- If you want these IPs propagated into runtime config files (frontend `src/config.js` or backend start scripts), I can update those next and run a small smoke-test.
