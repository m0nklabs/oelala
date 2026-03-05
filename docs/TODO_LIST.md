# Oelala TODO List

> Active development tasks. Updated: 2026-03-05

---

## 🎭 Face System (Active Sprint)

See [docs/FACE_SYSTEM.md](FACE_SYSTEM.md) for full architecture.

### Implementation Status
| Component | Status | Notes |
|-----------|--------|-------|
| Image face swap (direct) | ✅ Done | `/face-swap` endpoint |
| Image face swap (profile) | ✅ Done | `/face-swap/profile` endpoint |
| Video face swap (direct) | ✅ Done | `/face-swap-video` endpoint |
| Video face swap (profile) | ✅ Done | `/face-swap-video/profile` endpoint |
| Face profiles (create/list/delete) | ✅ Done | `/api/face-profiles` CRUD |
| Face LoRA training UI | ✅ Done | Tab 3 in FaceSwapTool |
| FaceSwapTool.jsx (image + video UI) | ✅ Done | 3-tab component |
| insightface buffalo_l pre-loaded | ✅ Done | CUDA warm |

### Still TODO — Face System
| Priority | Task | Notes |
|----------|------|-------|
| ✅ DONE | Test image swap end-to-end (API) | 5 faces detected, swap produces 768x1344 PNG |
| ✅ DONE | Test video swap end-to-end (API) | 480x480 41-frame video swapped correctly |
| ✅ DONE | Test face profile create + swap (API) | Profile CRUD + profile-based swap working |
| ✅ DONE | Fix auth bug (JWKS exception + duplicate decode) | auth.py fixed, backend restarted |
| ✅ DONE | Fix FaceSwapTool.jsx fetch→apiFetch | 10 raw fetch() calls migrated |
| MED | Test LoRA training (200 steps, 2 photos) | OOM when ComfyUI loaded; needs free VRAM |
| MED | GFPGAN face enhancement (image quality post-swap) | Model present, not wired up |
| MED | Use trained face LoRA in ComfyUI T2I/T2V workflow | Integration missing |
| LOW | Face swap in generated video (auto after I2V job) | UX feature |

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
| #51 | Video upscaling (480p → 4K) | frontend, backend | ✅ Closed — All 3 presets working (lanczos/ESRGAN/SeedVR2) |

> **Note**: All GitHub issues (#29-70, #51) have been closed. No open issues remain in m0nklabs/oelala.

---

## 📊 oelala-storage Status

| Issue | Title | Priority | Status |
|-------|-------|----------|--------|
| #12 | Prometheus metrics | done | ✅ Closed |
| #13 | Admin CLI | done | ✅ Closed (stats command added) |
| #24 | MEGA: Distributed Storage Network | high | 🔄 Open |
| #19 | MEGA: Operations & Observability | done | ✅ Closed (all sub-issues resolved) |
| #10 | Webhook notifications | done | ✅ Closed (implemented 2026-03-05) |
| #20 | MEGA: Platform & Deployment | done | ✅ Closed |
| #7 | Windows installer | low | 🔄 Open |

---

## ✅ Recently Completed

### 2026-03-06
- [x] Face system E2E testing: image swap, video swap, profiles CRUD — all working
- [x] fix(auth): JWKS exception handling + duplicate JWT decode removal
- [x] fix(frontend): FaceSwapTool.jsx 10x fetch()→apiFetch() for auth
- [x] Upscale E2E testing: all 3 video presets + 4 image models verified
- [x] SeedVR2 OOM fix (resolution cap + memory management)

### 2026-03-05
- [x] oelala-storage: Webhook notification system (#10) — async dispatcher, HMAC signing, retry
- [x] oelala-storage: Fiber v2.52.10 → v2.52.12 security fix (DoS + predictable UUID CVEs)
- [x] oelala-storage: Issues #19, #20 closed (already resolved)
- [x] oelala: PR #109 (opencv bump) resolved, PR #107 (video upscaling dead code) closed + cleaned
- [x] oelala: rollup CVE fixed (4.53.3 → 4.59.0)
- [x] oelala: Dead upscaler files removed (-671 lines), upscale wired to send-to menu

### 2026-03-04
- [x] UI refactor: inline styles → CSS classes across 14 tool components
- [x] Added .form-range CSS for range sliders
- [x] Storage fallback fix (eager exists check before lazy streaming)
- [x] LoRA Stack dropdown fix (object access)
- [x] Motion prompt thinking strip improvement
- [x] httpx import fix in /generate-motion-prompt

### 2026-03-03
- [x] I2T UI restructure: removed Caption Mode buttons, added Generate Motion button
- [x] Two-step I2T pipeline (vision LLM + T2T motion model)

### 2026-03-02
- [x] I2I Face Processing Pipeline (IP-Adapter FaceID, FaceDetailer, GFPGAN)
- [x] CreationsPickerModal refactored from overlay to inline panel (all 7 tools)
- [x] CORS fix: explicit origins + Vary:Origin + CF cache bust
- [x] apiFetch migration for image loading in all 7 tools
- [x] ComfyUI Impact-Subpack + face_yolov8m.pt installed

### 2026-01-15
- [x] Retention policy business logic (#71)

### 2026-01-13
- [x] Video-to-Video style transfer (#50)

### 2026-01-12
- [x] Supabase Database Implementation (EPIC #91)
- [x] API key management, Webhooks, Admin tools, Auto-upload, Stripe

### 2026-01-11
- [x] Real-Time Queue & Progress (MEGA #8)
- [x] Public REST API v1 (#65)

### 2026-01-10
- [x] Admin Panel (MEGA #84), Auto-Upload (MEGA #7), WebSocket progress

---

## 🎯 Suggested Next Steps

Based on project priorities:

1. **Face system testing** - All code is built, needs end-to-end validation
2. **UI polish** - Continue CSS class migration for remaining inline styles
3. **oelala-storage #24** - Distributed storage network (MEGA)
