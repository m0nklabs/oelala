# Oelala Product Roadmap

> **Last Updated**: 2026-03-06
> **Version**: 0.11.x (Alpha)

## Vision

Oelala is an AI media platform for creators who want one place for prompt generation, image creation, video generation, audio workflows, gallery publishing, and hybrid local/cloud execution.

---

## Current Snapshot

### ✅ Recently Landed

- Storage migration phase 1-5 completed and documented
- User/source/generated media now centered on oelala-storage rather than permanent local paths
- RunPod Cloud Max support added for Wan 2.2 text-to-video and image-to-video
- Cloud job persistence and queue semantics improved for backend restarts and stale queue handling
- I2I face pipeline expanded with FaceID, FaceDetailer, and GFPGAN
- Prompt and image-to-text workflows expanded with camera motion and two-step motion prompting
- Admin dashboard now has storage node/cluster visibility
- Frontend API calls standardized on `apiFetch()` for auth/CORS correctness

### 🔄 Active Focus

| Area | Current Focus | Why It Matters |
|------|---------------|----------------|
| Cloud reliability | RunPod worker provisioning, queue timeout behavior, clearer cloud failure states | Cloud generation must fail honestly instead of looking alive forever |
| Storage cluster | Coordinator + node rollout, public URLs, node heartbeats | Multi-node storage is the base for resilience and future scale |
| Media UX | Move/organize media, quota/retention visibility, polished gallery sorting | Users need sane lifecycle management, not just generation buttons |
| Legacy cleanup | Remove remaining fallback/local-path assumptions | Prevent split-brain between backend disk and storage service |

---

## Completed Product Areas

### Generation Stack
- [x] Text-to-Image
- [x] Image-to-Image
- [x] Inpainting and reframing
- [x] Image-to-Video
- [x] Text-to-Video
- [x] Video-to-Video
- [x] Image and video upscaling
- [x] Frame interpolation
- [x] Prompt generation and image captioning

### Face / Character Tooling
- [x] Face swap
- [x] I2I FaceID identity transfer
- [x] FaceDetailer refinement
- [x] GFPGAN face restoration
- [x] Face LoRA training queue integration

### User / Monetization
- [x] Supabase auth (Google + GitHub)
- [x] Credits and Stripe checkout
- [x] User media history and gallery publishing
- [x] Profiles and social links
- [x] NSFW gating and guest restrictions

### Platform / Ops
- [x] Multi-GPU local execution with DisTorch2
- [x] WebSocket progress and queue polling
- [x] Storage migration to oelala-storage
- [x] Cloudflare tunnel/CORS hardening
- [x] systemd-based deployment for core services

---

## Near-Term Roadmap

### 1. Cloud Execution Hardening

| Task | Status | Priority |
|------|--------|----------|
| Ensure at least one viable RunPod worker profile for Cloud Max | 🔄 In progress | Critical |
| Improve worker warm-up, queue visibility, and timeout reporting | 🔄 In progress | Critical |
| Finish EU-centric storage access for cloud workers | 🔄 In progress | High |
| Reduce cold-start friction for LoRA/model downloads | 🔄 In progress | High |

### 2. Storage as the Real Source of Truth

| Task | Status | Priority |
|------|--------|----------|
| Roll out coordinator + additional local node configs | 🔄 In progress | Critical |
| Remove more legacy direct-disk fallback code | 🔄 In progress | High |
| Expose quota, retention, and expiration to users | ⏳ Planned | High |
| Decide when backend proxy routes can be retired in favor of direct storage hostnames | ⏳ Planned | Medium |

### 3. Media Management Polish

| Task | Status | Priority |
|------|--------|----------|
| Folder/move/rename workflows in My Media | 🔄 In progress | High |
| Better storage/admin observability | 🔄 In progress | High |
| More consistent workflow metadata and import/export UX | ⏳ Planned | Medium |

### 4. Safety and Governance

| Task | Status | Priority |
|------|--------|----------|
| Improve auditability of admin and storage actions | ⏳ Planned | High |
| Add stronger moderation workflows beyond keyword gates | ⏳ Planned | High |
| Tighten public/private media access strategy | 🔄 In progress | High |

---

## Mid-Term Roadmap

### Distributed Storage Network

| Milestone | Status | Notes |
|-----------|--------|-------|
| Coordinator node | 🔄 In progress | Main storage entrypoint and node registry |
| Local node 01 | 🔄 In progress | Secondary local node on distinct ports |
| Remote node 02 | 🔄 In progress | Autonomous tunnel, independent host |
| Replication and observability | ⏳ Planned | Node health, lag, placement, metrics |

### Advanced Creation Workflows

| Milestone | Status | Notes |
|-----------|--------|-------|
| ControlNet-style guidance | ⏳ Planned | Better controllability |
| Stronger avatar consistency | ⏳ Planned | Character continuity across runs |
| More direct tool chaining | 🔄 In progress | "Use in tool", prompt/image/video reuse |
| Better preset system | 🔄 In progress | Factory presets + safe defaults |

### Commercial Readiness

| Milestone | Status | Notes |
|-----------|--------|-------|
| Credit-based monetization | ✅ Done | Already in production path |
| Subscription tiers | ⏳ Planned | Beyond one-off credit packs |
| API product surface | ⏳ Planned | External integrations |
| Team/org workflows | ⏳ Planned | Shared workspaces and governance |

---

## Longer-Term Direction

- Production-grade hybrid routing between local GPUs and cloud GPUs
- Richer provenance, audit, and retention tooling
- More composable media pipelines instead of isolated tool tabs
- Marketplace or packaged model/preset distribution once platform stability justifies it

---

## Technical Debt to Keep Burning Down

- Remove stale docs and code that still describe local media folders as canonical storage
- Keep frontend/backend/storage hostnames aligned across config and docs
- Keep cloud queue behavior honest: pending is pending, not fake running
- Keep auth behavior consistent across REST, media URLs, and WebSocket-style flows

### Performance
- [ ] Response caching (Redis)
- [ ] Database optimization (if added)
- [ ] Lazy loading in frontend
- [ ] Image/video compression pipeline
- [ ] WebP/AVIF support

### DevOps
- [ ] Docker containerization
- [ ] Kubernetes deployment configs
- [ ] Terraform infrastructure
- [ ] Monitoring (Prometheus/Grafana)
- [ ] Log aggregation (Loki/ELK)
- [ ] Alerting system

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | Nov 2025 | Initial release, T2V/I2V working |
| 0.1.5 | Dec 2025 | Web interface, ComfyUI integration |
| 0.2.0 | Jan 2026 | Audio pipeline, voice cloning, lip sync |
| 0.10.0 | Mar 2026 | I2I face processing, CORS fix, CreationsPickerModal inline |
| 0.11.0 | Jul 2026 | Auth hardening, fetch→apiFetch migration (14 files), face system E2E tests |

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on contributing to Oelala.

## Related Documents

- [PROJECT_PLAN.md](./PROJECT_PLAN.md) - Detailed task breakdown
- [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture
- [MEDIA_STORAGE.md](./MEDIA_STORAGE.md) - Storage architecture
- [TODO_TOOLS.md](./TODO_TOOLS.md) - Tool implementation status
