# Oelala TODO List

> Active development tasks. Updated: 2026-01-10

---

## 🔥 Priority: Admin Panel & User Management

**MEGA Issue**: #84 (Admin Panel)

**Goal**: Full admin dashboard for user/credits/content management

### P0 - Critical (Admin Access)
| Task | Status | Priority |
|------|--------|----------|
| Admin route protection (isAdmin check) | ⏳ Todo | Critical |
| Admin panel page (`/admin`) | ⏳ Todo | Critical |
| Admin navigation in sidebar | ⏳ Todo | Critical |

### P1 - User Management
| Task | Status | Priority |
|------|--------|----------|
| Users list with search/filter | ⏳ Todo | High |
| View user details (email, created, last login) | ⏳ Todo | High |
| Edit user credits (add/remove) | ⏳ Todo | High |
| Set user tier (free/pro/vip) | ⏳ Todo | High |
| Set admin/VIP status flags | ⏳ Todo | High |
| Ban/suspend user | ⏳ Todo | Medium |
| User activity log | ⏳ Todo | Low |

### P2 - Credits Administration
| Task | Status | Priority |
|------|--------|----------|
| Credits overview (total in system) | ⏳ Todo | Medium |
| Grant bonus credits to user | ⏳ Todo | Medium |
| View credit transactions | ⏳ Todo | Medium |
| Bulk credit operations | ⏳ Todo | Low |

### P3 - Content Moderation
| Task | Status | Priority |
|------|--------|----------|
| Review flagged content | ⏳ Todo | Medium |
| Remove published gallery items | ⏳ Todo | Medium |
| NSFW override controls | ⏳ Todo | Low |

### Infrastructure Required
| Task | Status | Notes |
|------|--------|-------|
| Supabase `user_credits` table | ❌ Missing | Causes 404 errors |
| Supabase `published_media` table | ❌ Missing | Gallery broken |
| Supabase `credit_transactions` table | ❌ Missing | For audit log |
| Backend admin API endpoints | ⏳ Todo | `/api/admin/*` |

---

## ✅ Recently Completed

### CHANGELOG Enforcement & SFW Content (2026-01-07)
- [x] GitHub Action for CHANGELOG enforcement on PRs
- [x] MEGA issue template with P0/P1/P2 requirements
- [x] PR template updated with CHANGELOG section
- [x] copilot-instructions.md CHANGELOG requirement
- [x] Vite 7.3.0 upgrade (Dependabot #14 security fix)
- [x] SFW content generation plan documented
- [x] SFW batch generator script (100 videos)
- [x] Rick & Morty agents for Copilot

### Credit System Docs (2026-01-07) - PR #79 Merged
- [x] Verification script (9 automated checks)
- [x] Deployment checklist document
- [x] Final summary document
- [x] pytest.ini configuration
- [x] GPU test markers for CI

### Guest Access & NSFW Protection (2026-01-06)
- [x] Dashboard accessible without login (view-only)
- [x] LoginModal component for on-demand auth
- [x] All generation tools require login
- [x] NSFW forced off for guests (context-level)
- [x] MyMediaTool hidden from guests
- [x] Gallery filters NSFW for non-authenticated
- [x] LogViewer admin-only (mark.op.mobiel@gmail.com)
- [x] Duplicate requirements.txt removed

### Credit System (2026-01-05) - PR #77 Merged
- [x] Supabase credit tables and RLS policies
- [x] Stripe checkout integration
- [x] Credit packages (€5-€500)
- [x] Welcome bonus (100 credits)
- [x] Credit costs per generation type
- [x] CreditsContext in frontend
- [x] Purchase success/cancel handling

### Gallery System (2026-01-05) - PR #78 Merged
- [x] Publish to Gallery from MyMedia
- [x] PublishModal with title/description/tags
- [x] SFW/NSFW tagging
- [x] Public gallery page
- [x] Like system
- [x] View count tracking
- [x] MediaDetailModal

---

## 🔥 Priority: Auto-Upload to User Storage

**Goal**: Generated content automatically saves to user's storage bucket

| Task | Status | Priority | Issue |
|------|--------|----------|-------|
| Hook into ComfyUI job completion | 🚧 In PR #81 | Critical | #7 |
| Upload to user storage bucket | 🚧 In PR #81 | Critical | #7 |
| Store metadata (prompt, settings) | 🚧 In PR #81 | High | #7 |
| Frontend refresh after upload | ⏳ Todo | High | #14 |
| Retry logic for failed uploads | ⏳ Todo | Medium | #15 |

---

## 🎬 In Progress: SFW Content Generation

**Goal**: 100 diverse SFW videos for frontpage gallery

| Task | Status | Notes |
|------|--------|-------|
| T2I + I2V workflow | ✅ Done | DisTorch2 multi-GPU |
| Batch script | ✅ Done | 100 prompts, 10 categories |
| Video generation | 🔄 Running | ~6 videos done, ~8 min each |

**Categories**: nature, animals, urban, abstract, space, weather, water, fire/light, plants, technology

---

## 🔄 Backlog: Storage Quota

| Task | Status | Priority | Issue |
|------|--------|----------|-------|
| Calculate storage used per user | ⏳ Todo | High | #33 |
| API endpoint for quota info | ⏳ Todo | High | #33 |
| Storage usage bar in user menu | ⏳ Todo | Medium | #33 |
| Warning when approaching limit | ⏳ Todo | Medium | #33 |
| Tier-based limits (free: 1GB) | ⏳ Todo | Medium | #71 |

---

## 🎵 Backlog: Audio Generation

**MEGA Issue**: #41

| Task | Status | Priority | Issue |
|------|--------|----------|-------|
| MMAudio model integration | ⏳ Todo | Medium | #48 |
| Audio generation API endpoints | ⏳ Todo | Medium | #47 |
| Audio generation UI component | ⏳ Todo | Medium | #49 |
| Audio sync with video | ⏳ Todo | Low | #41 |

---

## 🔄 Backlog: Advanced Video

**MEGA Issue**: #42

| Task | Status | Priority | Issue |
|------|--------|----------|-------|
| Video-to-Video style transfer | ⏳ Todo | Medium | #50 |
| Video extension (loop/extend) | ⏳ Todo | Low | - |

---

## 📋 Backlog

### Media Management
- [ ] Batch delete/download
- [ ] Share link generation
- [ ] Favorites sync to storage

### Generation Improvements
- [ ] ControlNet integration (#53 I2V enhancements)
- [ ] Inpainting tools
- [ ] Queue position indicator (#8, #16, #17)
- [ ] Email notification on complete

### UX Polish
- [ ] Age verification modal on first NSFW
- [ ] NSFW prompt pool (logged in only)
- [ ] "Remix" button on gallery items
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
