# Oelala TODO List

> Active development tasks. Updated: 2026-01-10

---

## 🔥 Current Priority: Admin Panel

**MEGA Issue**: [#84](https://github.com/m0nklabs/oelala/issues/84)

**Goal**: Full admin dashboard for user/credits/content management

### P0 - Critical (Admin Access)
| Task | Status | Issue |
|------|--------|-------|
| Admin route protection (isAdmin check) | ⏳ Todo | #84 |
| Admin panel page (`/admin`) | ⏳ Todo | #84 |
| Admin navigation in sidebar | ⏳ Todo | #84 |

### P1 - User Management
| Task | Status | Issue |
|------|--------|-------|
| Users list with search/filter | ⏳ Todo | #59 |
| View user details | ⏳ Todo | #59 |
| Edit user credits | ⏳ Todo | #59 |
| Ban/suspend user | ⏳ Todo | #59 |

### P2 - Credits & Content
| Task | Status | Issue |
|------|--------|-------|
| Analytics dashboard | ⏳ Todo | #60 |
| Content moderation queue | ⏳ Todo | #61 |

### Infrastructure Required
| Task | Status | Notes |
|------|--------|-------|
| Supabase `user_credits` table | ✅ Complete | Migration 001 |
| Supabase `published_media` table | ✅ Complete | Migration 002 |
| Supabase `profiles` table | ✅ Complete | Migration 005 |
| Supabase `user_media` table | ✅ Complete | Migration 006 |
| Supabase `gallery` table | ✅ Complete | Migration 006 |
| Backend admin API endpoints | ✅ Complete | `/api/admin/*` |
| Backend profile API endpoints | ✅ Complete | `/api/profile/*` |
| Migration documentation | ✅ Complete | `docs/MIGRATION_GUIDE.md` |

---

## 📋 Open GitHub Issues (by priority)

### High Priority
| Issue | Title | Labels |
|-------|-------|--------|
| #65 | Backend: Public REST API v1 | backend, priority:high |
| #63 | API key management system | backend, priority:high |
| #62 | Backend: Webhook delivery system | backend, priority:high |
| #51 | Video upscaling (480p → 4K) | frontend, backend, priority:high |
| #50 | Video-to-Video style transfer | frontend, backend, priority:high |

### Medium Priority
| Issue | Title | Labels |
|-------|-------|--------|
| #71 | Retention policy (tier-based expiration) | priority:medium |
| #70 | Monitoring and observability | infrastructure, priority:medium |
| #69 | CDN and caching optimization | priority:medium |
| #68 | Frontend: Code splitting | frontend, priority:medium |
| #67 | Backend: Database optimization | backend, database, priority:medium |
| #61 | Content moderation queue | backend, priority:medium |
| #60 | Analytics dashboard | frontend, priority:medium |
| #59 | Admin: User management tools | backend, priority:medium |
| #58 | Admin dashboard main page | frontend, priority:medium |
| #56 | User profile page | frontend, priority:medium |
| #55 | User profile API | backend, database, priority:medium |

### Low Priority
| Issue | Title | Labels |
|-------|-------|--------|
| #64 | SDKs and developer examples | documentation, priority:low |
| #57 | Following system | database, backend, priority:low |
| #54 | Avatar upload | frontend, priority:low |

---

## ✅ Recently Completed

### 2026-01-10
- [x] Mobile responsive layout (collapsible parameters)
- [x] README modernization
- [x] Repo description update

### 2026-01-09
- [x] ComfyUI upstream sync (13 commits)
- [x] Audio VAE fix (committed to fork)
- [x] Fiber security fix in oelala-storage
- [x] Credits bypass toggle (CREDITS_ENABLED)

### 2026-01-07
- [x] CHANGELOG enforcement on PRs
- [x] MEGA issue template
- [x] Vite 7.3.0 security upgrade
- [x] Rick & Morty agent personas

### 2026-01-06
- [x] Guest access (view-only dashboard)
- [x] LoginModal for auth-required actions
- [x] NSFW forced off for guests
- [x] LogViewer admin-only

### 2026-01-05
- [x] Credit system (Stripe, packages)
- [x] Gallery system (publish, likes, views)
- [x] SFW/NSFW content tagging

---

## 🔄 Backlog

### Storage & Quota
- [ ] Auto-upload generated content (#7, #15)
- [ ] Storage quota tracking (#33)
- [ ] Tier-based limits (free: 1GB)

### Generation
- [ ] ControlNet integration
- [ ] Video extension (loop/extend)
- [ ] Queue position indicator

### UX
- [ ] Age verification modal
- [ ] "Remix" button on gallery
- [ ] Share link generation

---

*Sync with [GitHub Issues](https://github.com/m0nklabs/oelala/issues)*
