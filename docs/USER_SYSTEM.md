# User System - Supabase Auth Integration

> **Status**: Planning  
> **Last Updated**: 2026-01-04

---

## Overview

User authentication via Supabase Auth met OAuth providers en passkeys.

### Waarom Supabase?
- Self-hostable (Docker) voor productie
- Cloud gratis tier (50k MAU) voor development
- Native passkeys support
- OAuth: Google, GitHub, Discord, Apple, etc.
- JWT tokens → compatible met oelala-storage
- Python + JS SDKs

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Frontend (React)                        │
│  ┌─────────────────────────────────────────────────┐    │
│  │  AuthContext.jsx - Global auth state            │    │
│  │  LoginPage.jsx - OAuth buttons + passkey        │    │
│  │  useAuth() hook - Access user/session           │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────┬───────────────────────────────────┘
                      │ JWT in Authorization header
                      ▼
┌─────────────────────────────────────────────────────────┐
│                Backend (FastAPI)                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  auth.py - JWT validation middleware            │    │
│  │  get_current_user() dependency                  │    │
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
│  │  Passkey (WebAuthn) support                     │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## User Model

```python
# Supabase provides auth.users, we extend with profiles
class UserProfile:
    id: UUID           # Supabase user ID
    email: str
    display_name: str
    avatar_url: str | None
    is_adult: bool     # Age verification for NSFW
    tier: str          # 'free' | 'pro' | 'enterprise'
    created_at: datetime
    
    # Preferences
    nsfw_enabled: bool      # User's NSFW toggle preference
    default_model: str      # Preferred generation model
    theme: str              # UI theme
```

---

## OAuth Providers (Priority Order)

| Provider | Priority | Notes |
|----------|----------|-------|
| Google | 🔴 High | Most users have Google |
| GitHub | 🔴 High | Developer audience |
| Discord | 🟡 Medium | Gaming/creator community |
| Passkey | 🟡 Medium | Modern, passwordless |
| Apple | 🟢 Low | iOS users (we don't support iOS) |
| Email/Password | 🟢 Low | Fallback only |

---

## Implementation Plan

### Phase 1: Basic Auth (MVP)
- [ ] Supabase project setup (cloud for now)
- [ ] Frontend: AuthContext + LoginPage
- [ ] Backend: JWT middleware
- [ ] Google OAuth working
- [ ] Protected routes (My Media)
- [ ] NSFW toggle gated to logged-in users

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
- [ ] Custom domain (auth.oelala.ai)
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
