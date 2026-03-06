# User System - Supabase Auth Integration

> **Status**: ✅ Implemented
> **Last Updated**: 2026-01-06

---

## Overview

User authentication via Supabase Auth met OAuth providers.

### Waarom Supabase?
- Self-hostable (Docker) voor productie
- Cloud gratis tier (50k MAU) voor development
- Native passkeys support (future)
- OAuth: Google, GitHub
- JWT tokens → compatible met oelala-storage
- Python + JS SDKs

---

## Current Implementation

### What's Working ✅
- **Google OAuth** - Primary login method
- **GitHub OAuth** - Secondary login method
- **JWT validation** - Backend validates Supabase tokens
- **User-scoped storage** - Each user has own bucket
- **Credit system** - Per-user credits with Stripe payments
- **NSFW toggle** - Requires login, forced off for guests
- **Guest access** - Dashboard viewable without login
- **Login modal** - On-demand auth for protected actions
- **Admin whitelist** - `mark.op.mobiel@gmail.com` for dev features

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Frontend (React)                        │
│  ┌─────────────────────────────────────────────────┐    │
│  │  AuthContext.jsx - Global auth state            │    │
│  │  LoginModal.jsx - On-demand login popup         │    │
│  │  useAuth() hook - Access user/session           │    │
│  │  requestLogin() - Trigger login modal           │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────┘
                      │ JWT in Authorization header
                      ▼
┌─────────────────────────────────────────────────────────┐
│                Backend (FastAPI)                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  auth.py - JWT validation middleware            │    │
│  │  get_current_user() dependency                  │    │
│  │  get_optional_user() for public endpoints       │    │
│  │  Protected endpoints require valid JWT          │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────┘
                      │ JWT forwarded
                      ▼
┌─────────────────────────────────────────────────────────┐
│              Supabase Auth                               │
│  ┌─────────────────────────────────────────────────┐    │
│  │  User management, sessions, OAuth               │    │
│  │  JWT signing & verification                     │    │
│  │  Cloud: nsbjwhxdkxnyggtuxjjp.supabase.co       │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## User Access Levels

| Feature | Guest | Logged In | Admin |
|---------|-------|-----------|-------|
| View Dashboard | ✅ | ✅ | ✅ |
| View Gallery (SFW) | ✅ | ✅ | ✅ |
| View Gallery (NSFW) | ❌ | ✅ | ✅ |
| Generate Content | ❌ | ✅ | ✅ |
| My Media | ❌ | ✅ | ✅ |
| Publish to Gallery | ❌ | ✅ | ✅ |
| NSFW Toggle | ❌ | ✅ | ✅ |
| LogViewer | ❌ | ❌ | ✅ |
| ComfyUI Dev Content | ❌ | ❌ | ✅ |

**Admin emails**: `mark.op.mobiel@gmail.com`

---

## OAuth Providers

| Provider | Status | Notes |
|----------|--------|-------|
| Google | ✅ Active | Primary login |
| GitHub | ✅ Active | Developer audience |
| Discord | ⏳ Future | Gaming/creator community |
| Passkey | ⏳ Future | Modern, passwordless |

---

## Frontend Components

### AuthContext.jsx
```javascript
// Provides:
- user          // Current user or null
- session       // Supabase session
- loading       // Auth loading state
- signInWithGoogle()
- signInWithGithub()
- signOut()
- isAdult       // true if logged in (for NSFW)
- showLoginModal
- requestLogin(message)  // Trigger login popup
- closeLoginModal()
```

### LoginModal.jsx
On-demand login popup shown when guest tries to:
- Generate content
- View My Media
- Enable NSFW toggle
- Publish to gallery

### NSFWContext.jsx
- NSFW state persisted in localStorage
- `effectiveNsfwEnabled` always false for guests
- Prevents enabling NSFW without login

### Phase 2: Enhanced Auth
- [ ] GitHub OAuth
- [ ] Discord OAuth
- [ ] Passkey support
- [ ] User profile page
- [ ] Avatar upload

### Phase 3: Integration
- [ ] oelala-storage: user_id in paths
- [ ] Media ownership (migrate existing to dev account)
- [ ] Per-user quotas
- [ ] Generation history per user

### Phase 4: Self-Hosting (Production)
- [ ] Self-host Supabase via Docker
- [ ] Custom domain (auth.oelala.xyz)
- [ ] Backup & security hardening

---

## Configuration

### Environment Variables

```bash
# .env (DO NOT COMMIT)
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_ANON_KEY=eyJhbG...
SUPABASE_SERVICE_KEY=eyJhbG...  # Backend only, never expose
SUPABASE_JWT_SECRET=your-jwt-secret

# OAuth (configured in Supabase dashboard)
GOOGLE_CLIENT_ID=xxx.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=xxx
GITHUB_CLIENT_ID=xxx
GITHUB_CLIENT_SECRET=xxx
```

### .env.example

```bash
# Supabase Auth
SUPABASE_URL=
SUPABASE_ANON_KEY=
SUPABASE_SERVICE_KEY=
SUPABASE_JWT_SECRET=
```

---

## File Structure

```
src/
├── frontend/
│   └── src/
│       ├── contexts/
│       │   ├── NSFWContext.jsx      # Existing
│       │   └── AuthContext.jsx      # NEW - Auth state
│       ├── pages/
│       │   ├── LoginPage.jsx        # NEW - OAuth buttons
│       │   └── ProfilePage.jsx      # NEW - User settings
│       ├── components/
│       │   ├── AuthGuard.jsx        # NEW - Protected route wrapper
│       │   └── UserMenu.jsx         # NEW - Avatar dropdown
│       └── lib/
│           └── supabase.js          # NEW - Supabase client
│
└── backend/
    ├── app.py                       # Add auth middleware
    ├── auth.py                      # NEW - JWT validation
    └── models/
        └── user.py                  # NEW - User profile model
```

---

## Frontend Flow

### Login Page
```jsx
// OAuth buttons
<button onClick={() => supabase.auth.signInWithOAuth({ provider: 'google' })}>
  Continue with Google
</button>
<button onClick={() => supabase.auth.signInWithOAuth({ provider: 'github' })}>
  Continue with GitHub
</button>

// Passkey
<button onClick={handlePasskeyLogin}>
  Login with Passkey
</button>
```

### Protected Routes
```jsx
// Wrap protected content
<AuthGuard requireAdult={true}>
  <NSFWToggle />  {/* Only visible to verified adults */}
</AuthGuard>
```

---

## Backend Flow

### JWT Middleware
```python
from fastapi import Depends, HTTPException
from supabase import create_client

async def get_current_user(authorization: str = Header(None)):
    if not authorization:
        return None  # Anonymous user

    token = authorization.replace("Bearer ", "")
    user = supabase.auth.get_user(token)

    if not user:
        raise HTTPException(401, "Invalid token")

    return user

# Usage
@app.get("/my-media")
async def get_my_media(user = Depends(get_current_user)):
    if not user:
        raise HTTPException(401, "Login required")
    return get_media_for_user(user.id)
```

---

## Age Verification for NSFW

### Options
1. **Checkbox** - "I confirm I am 18+" (weakest)
2. **Date of Birth** - Calculate age (medium)
3. **ID Verification** - Third party service (strongest, $$)

### Recommended: DOB + Checkbox
```jsx
<input type="date" onChange={handleDOB} />
<label>
  <input type="checkbox" required />
  I confirm I am 18 years or older
</label>
```

Store `is_adult: true` in profile after verification.

---

## Migration Plan

### Existing Media
1. Create "dev" user account
2. Assign all existing media to dev user
3. Set `user_id` in media metadata going forward

### Anonymous Users
- Can browse SFW content
- Cannot generate (or limited free generations?)
- Prompted to login to save/access history

---

## Security Considerations

- [ ] JWT validation on every protected endpoint
- [ ] HTTPS only for auth endpoints
- [ ] Rate limiting on login attempts
- [ ] Secure cookie settings (httpOnly, sameSite)
- [ ] CORS properly configured
- [ ] No secrets in frontend code
- [ ] Audit logging for auth events

---

## Next Steps

1. **Create Supabase project** (cloud.supabase.com)
2. **Configure Google OAuth** in Supabase dashboard
3. **Install dependencies**:
   - Frontend: `npm install @supabase/supabase-js`
   - Backend: `pip install supabase`
4. **Implement AuthContext.jsx**
5. **Implement LoginPage.jsx**
6. **Add JWT middleware to backend**
7. **Test login flow**
