# Oelala Product Roadmap

> **Last Updated**: 2026-01-03  
> **Version**: 0.2.0 (Alpha)

## Vision

Oelala is an AI-powered video generation platform that enables creators to produce professional-quality video content using state-of-the-art generative AI models. The platform combines text-to-video, image-to-video, voice synthesis, lip-sync, and avatar generation into a unified, user-friendly interface.

---

## Current Status: Alpha (v0.2.x)

### ✅ Completed Features

#### Core Generation Pipeline
- [x] Text-to-Image (T2I) via ComfyUI/Flux
- [x] Image-to-Video (I2V) via Wan2.2
- [x] Text-to-Video (T2V) via Wan2.2
- [x] Video-to-Video style transfer
- [x] Image upscaling (multiple models)
- [x] Video reframing/aspect ratio conversion

#### Audio Pipeline (Phase 3) - January 2026
- [x] YouTube import (yt-dlp integration)
- [x] Text-to-Speech (ChatterBox TTS)
- [x] Voice Cloning (F5-TTS with multi-language support)
- [x] Lip Sync (LatentSyncNode)

#### Infrastructure
- [x] React/Vite frontend with tool-based UI
- [x] FastAPI backend with ComfyUI integration
- [x] WebSocket progress streaming
- [x] Multi-GPU support (basic)
- [x] Workflow metadata embedding in outputs

---

## Short-Term Roadmap (Q1 2026)

### Phase 4: Media Management 🔄 In Progress

**Goal**: Unified media storage and improved developer experience

| Task | Status | Priority |
|------|--------|----------|
| Unify storage locations (generated + ComfyUI/output) | 🔄 Planned | High |
| ComfyUI symlink integration | 🔄 Planned | High |
| Improved My Media browser | 🔄 Planned | Medium |
| Media metadata indexing | ⏳ Todo | Medium |
| Batch operations (delete, download, share) | ⏳ Todo | Low |

### Phase 4.5: oelala-storage Integration 💰 NEW

**Goal**: Monetization-ready storage backend (Go service)

> **Repository**: [oelala-storage](https://github.com/m0nklabs/oelala-storage)

| Task | Status | Priority | Monetization Impact |
|------|--------|----------|---------------------|
| User/tenant isolation | ⏳ Todo | Critical | Enables multi-user |
| Storage quota tracking | ⏳ Todo | Critical | Upsell trigger |
| Usage metering (bytes, bandwidth) | ⏳ Todo | Critical | Billing foundation |
| Tier-based limits enforcement | ⏳ Todo | High | Plan differentiation |
| Retention policies (30d free) | ⏳ Todo | High | Free tier cleanup |
| Signed URL generation | ⏳ Todo | Medium | Premium sharing |
| Watermark injection hook | ⏳ Todo | Medium | Conversion driver |
| Content deduplication | ⏳ Todo | Low | Cost optimization |

**Why separate service?**
- Go = single binary, no Python venv issues
- Better I/O performance (goroutines vs GIL)
- Can run on edge devices (Android, low-resource servers)
- S3-compatible = easy integration with existing tools

### Phase 5: Advanced Generation

**Goal**: More control over generation process

| Task | Status | Priority |
|------|--------|----------|
| ControlNet integration (pose, depth, canny) | ⏳ Todo | High |
| LoRA model browser and loading | ⏳ Todo | High |
| Custom LoRA training interface | ⏳ Todo | Medium |
| Inpainting/outpainting tools | ⏳ Todo | Medium |
| Frame interpolation (FI) | ⏳ Todo | Low |

### Phase 6: Avatar System

**Goal**: Consistent character generation

| Task | Status | Priority |
|------|--------|----------|
| Character/avatar profile system | ⏳ Todo | High |
| Face consistency across generations | ⏳ Todo | High |
| Expression control | ⏳ Todo | Medium |
| Full-body pose library | ⏳ Todo | Medium |
| Avatar-to-video pipeline | ⏳ Todo | High |

---

## Mid-Term Roadmap (Q2-Q3 2026)

### Phase 7: User System & Multi-tenancy

**Goal**: Production-ready user management

#### Authentication Providers
| Provider | Status | Notes |
|----------|--------|-------|
| Email/Password | ⏳ Todo | Basic auth |
| Google OAuth | ⏳ Todo | Priority |
| GitHub OAuth | ⏳ Todo | Developer-friendly |
| Discord OAuth | ⏳ Todo | Community integration |
| Facebook OAuth | ⏳ Todo | Mass market |
| Steam OAuth | ⏳ Todo | Gaming audience |
| Apple Sign-In | ⏳ Todo | iOS requirement |

#### User Features
- [ ] User registration & profiles
- [ ] Project/workspace management
- [ ] Generation history & favorites
- [ ] Usage quotas & tracking
- [ ] API key management
- [ ] Team/organization support

### Phase 8: Local-First Distributed Storage

**Goal**: Self-hosted, cross-platform storage nodes before cloud integration

#### Storage Node Architecture
| Feature | Status | Priority |
|---------|--------|----------|
| Cross-platform node daemon (Win/Linux/Mac) | ⏳ Todo | Critical |
| SQLite metadata database | ⏳ Todo | Critical |
| REST API for node communication | ⏳ Todo | High |
| Content-addressed storage (SHA-256) | ⏳ Todo | High |
| P2P sync engine | ⏳ Todo | High |
| File chunking & resumable transfers | ⏳ Todo | Medium |
| Conflict resolution (LWW/manual) | ⏳ Todo | Medium |
| Node discovery (mDNS/manual) | ⏳ Todo | Medium |
| At-rest encryption (optional) | ⏳ Todo | Low |
| LZ4/ZSTD compression (optional) | ⏳ Todo | Low |

#### Node Types
| Type | Description | Typical Hardware |
|------|-------------|------------------|
| **Primary** | Main node, always online | Home server, VPS |
| **Replica** | Mirror/failover, read-heavy | Secondary server |
| **Edge** | Local cache, partial sync | Desktop, laptop |
| **Archive** | Cold storage, infrequent access | NAS, external drive |

### Phase 9: Cloud Integration (Post-Local)

**Goal**: Add cloud backends once local distribution is stable

| Feature | Status | Priority |
|---------|--------|----------|
| S3-compatible backend (MinIO, AWS, etc.) | ⏳ Todo | High |
| CDN integration (CloudFlare/Bunny) | ⏳ Todo | High |
| Multi-region redundancy | ⏳ Todo | Medium |
| Automatic backup/archival | ⏳ Todo | Medium |
| Content deduplication | ⏳ Todo | Low |
| IPFS/decentralized option | ⏳ Todo | Low |

### Phase 10: Security & Compliance

**Goal**: Enterprise-ready security

| Feature | Status | Priority |
|---------|--------|----------|
| Content moderation (NSFW filtering) | ⏳ Todo | Critical |
| DMCA takedown system | ⏳ Todo | Critical |
| Watermarking (optional/invisible) | ⏳ Todo | High |
| Audit logging | ⏳ Todo | High |
| GDPR compliance tools | ⏳ Todo | High |
| Content provenance (C2PA) | ⏳ Todo | Medium |
| Rate limiting & abuse prevention | ⏳ Todo | High |

---

## Long-Term Roadmap (Q4 2026+)

### Phase 10: Commercial Platform

**Goal**: Monetization and scaling

#### Pricing Tiers
| Tier | Target | Features |
|------|--------|----------|
| Free | Hobbyists | Limited generations, watermark, queue |
| Creator ($19/mo) | Content creators | More generations, no watermark, priority |
| Pro ($49/mo) | Professionals | Unlimited, API access, custom models |
| Enterprise | Studios | Self-hosted, SLA, dedicated support |

#### Monetization Features
- [ ] Subscription billing (Stripe)
- [ ] Credit-based usage system
- [ ] Marketplace for custom LoRAs/models
- [ ] API metering & billing
- [ ] White-label/reseller program

### Phase 11: Advanced AI Features

| Feature | Description |
|---------|-------------|
| Real-time generation | Stream generation progress |
| Multi-shot video editing | Storyboard-based generation |
| Audio-reactive video | Music visualization |
| 3D avatar generation | Full 3D character creation |
| Virtual try-on | Fashion/product visualization |
| AI video editing | Smart cuts, transitions |

### Phase 12: Platform Ecosystem

| Feature | Description |
|---------|-------------|
| Plugin/extension system | Third-party integrations |
| Public API | Developer access |
| Mobile apps | iOS/Android clients |
| Desktop app | Electron-based client |
| Browser extension | Quick generation from any page |
| Zapier/n8n integration | Workflow automation |

---

## Technical Debt & Improvements

### Code Quality
- [ ] Comprehensive test suite (pytest, vitest)
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Code coverage > 80%
- [ ] API documentation (OpenAPI/Swagger)
- [ ] Type hints throughout backend

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

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for guidelines on contributing to Oelala.

## Related Documents

- [PROJECT_PLAN.md](./PROJECT_PLAN.md) - Detailed task breakdown
- [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture
- [MEDIA_STORAGE.md](./MEDIA_STORAGE.md) - Storage architecture
- [TODO_TOOLS.md](./TODO_TOOLS.md) - Tool implementation status
