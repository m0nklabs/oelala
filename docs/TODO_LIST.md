# Oelala TODO List

> Active development tasks. Updated: 2026-01-15

---

## ✅ Current Status: Core Features Complete

The main functionality is complete! All MEGA issues for core features are done:
- ✅ Credit System & Payments (MEGA #78)
- ✅ Real-Time Progress Tracking (MEGA #8)
- ✅ Auto-Upload to Storage (MEGA #7)
- ✅ Admin Panel (MEGA #84)
- ✅ Supabase Database (EPIC #91)
- ✅ API Keys & Webhooks (#62, #63)
- ✅ Public REST API v1 (#65)
- ✅ Retention Policy (#71)

---

## 📋 Open GitHub Issues (by priority)

### 🔴 High Priority
| Issue | Title | Labels | Notes |
|-------|-------|--------|-------|
| #51 | Video upscaling (480p → 4K) | frontend, backend | **BLOCKED**: No ESRGAN models installed |

### 🟡 Medium Priority - Ready to Work
| Issue | Title | Labels |
|-------|-------|--------|
| #33 | Storage quota tracking and display | frontend, backend, storage |
| #36 | Age verification modal for NSFW | frontend, ux |
| #47 | Audio generation API endpoints | backend |
| #48 | MMAudio model integration | backend |
| #49 | Audio generation UI panel | frontend |
| #55 | User profile API and database | backend, database |
| #56 | User profile page component | frontend |
| #58 | Admin dashboard main page | frontend |
| #60 | Analytics and metrics dashboard | frontend |
| #61 | Content moderation queue | backend |
| #67 | Database query optimization | backend, database |
| #68 | Code splitting and lazy loading | frontend |
| #69 | CDN and caching optimization | infrastructure |
| #70 | Monitoring and observability | infrastructure |

### 🟢 Low Priority
| Issue | Title | Labels |
|-------|-------|--------|
| #29 | ControlNet integration | frontend, backend |
| #30 | Canvas-based inpainting | frontend |
| #31 | Email notification on job completion | backend |
| #32 | Remix button - copy settings | frontend |
| #35 | NSFW default prompt pool | frontend |
| #37 | Share link generation | frontend, backend |
| #54 | Avatar upload and management | frontend |
| #57 | Following system | backend, database |
| #64 | SDKs and developer examples | documentation |

### 🚀 MEGA Issues (Epics)
| Issue | Title | Priority |
|-------|-------|----------|
| #41 | Audio Generation & Sound Design | medium |
| #42 | Advanced Video Workflows | medium |
| #43 | User Profiles & Social Features | medium |
| #44 | Admin Dashboard & Content Moderation | medium |
| #45 | Public API & Developer Integrations | medium |
| #46 | Performance & Infrastructure Optimization | medium |

---

## 📊 oelala-storage Status

| Issue | Title | Priority | Status |
|-------|-------|----------|--------|
| #12 | Prometheus metrics | done | ✅ Closed |
| #13 | Admin CLI | done | ✅ Closed (stats command added) |
| #10 | Webhook notifications | medium | 🔄 Open |
| #7 | Windows installer | low | 🔄 Open |
| #19 | MEGA: Operations & Observability | medium | 🔄 Open |
| #20 | MEGA: Platform & Deployment | low | 🔄 Open |
| #24 | MEGA: Distributed Storage Network | high | 🔄 Open |

---

## ✅ Recently Completed

### 2026-01-15
- [x] Retention policy business logic (#71)
  - Tier-based expiration: free=30d, pro=90d, vip=365d
  - X-Expires-At header sent to storage
  - Metadata includes tier, retention_days, expires_at

### 2026-01-13
- [x] Video-to-Video style transfer (#50)

### 2026-01-12
- [x] Supabase Database Implementation (EPIC #91)
- [x] API key management system (#63)
- [x] Webhook delivery system (#62)
- [x] Admin user management tools (#59)
- [x] Auto-upload service (#15)
- [x] Stripe integration (#12)

### 2026-01-11
- [x] Real-Time Queue & Progress (MEGA #8)
- [x] Public REST API v1 (#65)

### 2026-01-10
- [x] Admin Panel (MEGA #84)
- [x] Auto-Upload to Storage (MEGA #7)
- [x] WebSocket progress events (#17)
- [x] Real-time progress indicator (#16)

---

## 🎯 Suggested Next Steps

Based on project priorities:

1. **#33 Storage quota tracking** - Completes storage integration with UI
2. **#55 + #56 User profiles** - Enables social features
3. **#47 + #48 + #49 Audio generation** - New product feature
4. **#67 Database optimization** - Performance improvement
