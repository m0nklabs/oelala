# Oelala TODO List

> Active development tasks. Updated: 2026-01-04

---

## 🔥 Priority: New User Experience (NUX)

**Goal**: First-time visitor clicks one button and sees AI magic within 60 seconds

### Authentication & Content Control
| Task | Status | Priority | Notes |
|------|--------|----------|-------|
| Google OAuth login | ✅ Done | Critical | Supabase integration |
| User-scoped storage | ✅ Done | Critical | `/users/{user_id}/media/` |
| Logout dropdown menu | ✅ Done | High | Click avatar → dropdown |
| NSFW toggle requires login | ✅ Done | High | `isAdult` check in AuthContext |
| Age verification on first NSFW | ⏳ Todo | Medium | "I am 18+" checkbox modal |

### Default Settings & Auto-Prompt
| Task | Status | Priority | Notes |
|------|--------|----------|-------|
| AI-generated default prompt | ✅ Done | Critical | Random creative prompt on load |
| SFW default prompt pool | ✅ Done | Critical | 25+ safe, interesting prompts |
| NSFW default prompt pool | ⏳ Todo | High | Only shown when logged in + toggle |
| Optimal default parameters | ✅ Done | High | 480p, 6s, 16fps, 6 steps |
| One-click "Create" experience | ✅ Done | Critical | Prompt pre-filled, just upload image |
| ✨ Random prompt button | ✅ Done | Medium | Sparkles icon in prompt header |

### Queue & Wait Time UX
| Task | Status | Priority | Notes |
|------|--------|----------|-------|
| Estimated wait time display | ✅ Done | High | "~1m 30s" shown before Create |
| Queue position indicator | ⏳ Todo | Medium | "You're #3 in queue" |
| Progress bar with ETA | ⏳ Todo | Medium | Real-time during generation |
| Email notification on complete | ⏳ Todo | Low | Optional for long jobs |

---

## 🔄 In Progress: Storage Integration

| Task | Status | Priority | Notes |
|------|--------|----------|-------|
| oelala-storage running | ✅ Done | Critical | Port 7990 |
| Backend auth (JWT) | ✅ Done | Critical | Supabase tokens |
| User storage endpoints | ✅ Done | High | `/user/media` CRUD |
| Admin sees ComfyUI output | ✅ Done | Medium | Whitelist by email |
| Handle empty bucket (404) | ✅ Done | High | Return empty list |
| Upload generated content | ⏳ Todo | Critical | Auto-save to user storage |
| Storage quota tracking | ⏳ Todo | High | Free tier limits |

---

## 🌟 Publish System (Community Gallery)

**Goal**: Users publish their best work → SFW content feeds the frontpage

### Core Publish Flow
| Task | Status | Priority | Notes |
|------|--------|----------|-------|
| "Publish" button on media items | ⏳ Todo | High | Like favorites, but public |
| Publish modal with details | ⏳ Todo | High | Title, description, tags |
| SFW/NSFW flag on publish | ⏳ Todo | Critical | Auto-detect + manual override |
| Published media metadata | ⏳ Todo | High | Store in DB with user_id |
| Unpublish option | ⏳ Todo | Medium | Remove from gallery |

### Frontpage Gallery
| Task | Status | Priority | Notes |
|------|--------|----------|-------|
| Public gallery page | ⏳ Todo | Critical | `/gallery` or frontpage |
| SFW-only for anonymous | ⏳ Todo | Critical | Filter NSFW unless logged in |
| Reuse MyMedia gallery component | ⏳ Todo | High | Same grid, different source |
| Pagination/infinite scroll | ⏳ Todo | Medium | Performance |
| Sort by: newest, popular, random | ⏳ Todo | Medium | Discovery |

### Media Details Page
| Task | Status | Priority | Notes |
|------|--------|----------|-------|
| `/media/{id}` detail page | ⏳ Todo | High | Full-size view |
| Show prompt/settings used | ⏳ Todo | Medium | Educational for users |
| Creator attribution | ⏳ Todo | Medium | Username/avatar |
| Like/reaction system | ⏳ Todo | Low | Social engagement |
| "Remix" button | ⏳ Todo | Low | Copy settings to generator |

---

## 📋 Backlog

### Media Management
- [ ] Batch delete/download
- [ ] Share link generation
- [ ] Favorites sync to storage

### Generation
- [ ] LoRA browser with thumbnails
- [ ] ControlNet integration
- [ ] Inpainting tools

### Advanced Video Workflows
- [x] **Video Upscaling** (2026-01-05)
  - VideoUpscalerTool.jsx component
  - `/upscale-video` backend endpoint
  - Resolution presets: 480p → 720p → 1080p → 4K
  - Quality vs speed presets (fast, balanced, quality)
  - Workflow: `workflows/VideoUpscale/video_upscale_realesrgan.json`
- [x] **Frame Interpolation** (2026-01-05)
  - FrameInterpolationTool.jsx component
  - `/interpolate-video` backend endpoint
  - FPS conversion: 15fps → 30fps → 60fps
  - Slow motion modes: 2x, 4x, 8x
  - RIFE/FILM model support
  - Workflow: `workflows/FrameInterpolation/rife_interpolation.json`
- [x] **Image-to-Video enhancements** (already existed)
  - Camera motion presets (16 options)
  - Multiple duration options (3-15s)
  - CameraMotionSelector component
- [ ] **Video Extension** (planned)
  - Extend video forwards/backwards
  - Seamless loop creation
  - Workflow: `workflows/VideoExtension/extend_video_wan22.json`

### Monetization
- [x] Credit system backend (CreditManager, API endpoints)
- [x] Supabase migrations for credits tables
- [x] Credit cost calculation per generation type
- [x] Frontend credits UI (balance, purchase modal)
- [ ] Stripe integration (checkout, webhooks) - needs STRIPE_SECRET_KEY
- [ ] Credit deduction in generate endpoints
- [ ] Run Supabase migration (001_credits_system.sql)

---

## ✅ Recently Completed

- [x] **Frontend credits UI** (2026-01-04)
  - CreditsContext: balance, packages, purchase flow
  - PurchaseCreditsModal: shows packages with pricing
  - UserMenu: credits display + buy button
- [x] **Credit-based monetization model** (2026-01-04)
  - Backend: `credits.py`, `credits_api.py`
  - DB: `migrations/001_credits_system.sql`
  - Endpoints: `/api/credits/*`
- [x] Cloudflare Tunnel setup (oelala.xyz, api.oelala.xyz)
- [x] Google OAuth via Supabase
- [x] User menu with logout dropdown
- [x] JWT authentication in backend
- [x] User-scoped storage paths
- [x] Admin email whitelist for dev access
