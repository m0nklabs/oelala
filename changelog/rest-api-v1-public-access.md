### Added
- **Public REST API v1**: Programmatic access to Oelala AI generation services
  - Database migration `004_api_keys.sql` for secure API key storage with SHA-256 hashing
  - Backend module `api_key_auth.py` with key generation, validation, and authentication
  - API v1 router `api_v1.py` with versioned endpoints at `/api/v1/*`
  - API key management module `api_keys_management.py` for CRUD operations
  - 13 unit tests for API key authentication (all passing)
  - Comprehensive API documentation in `docs/API_v1.md`
  
- **API v1 Endpoints**:
  - `POST /api/v1/generate` - Generate images or videos with AI
  - `GET /api/v1/jobs/{id}` - Poll job status with progress tracking
  - `GET /api/v1/jobs/{id}/download` - Download completed results
  - `GET /api/v1/credits` - Check credit balance
  - `GET /api/v1/health` - Health check (no auth required)
  
- **API Key Management** (JWT authenticated):
  - `POST /api/keys` - Create new API key with optional expiration
  - `GET /api/keys` - List all user API keys
  - `GET /api/keys/{id}` - Get specific API key details
  - `PATCH /api/keys/{id}` - Update key (rename, enable/disable)
  - `DELETE /api/keys/{id}` - Permanently delete/revoke key
  
- **Security Features**:
  - API keys use `oelala_` prefix for easy identification
  - SHA-256 hashing for secure storage (keys never stored in plaintext)
  - Key prefix display (first 8 chars) for user identification
  - Usage tracking (count, last used timestamp)
  - Optional expiration dates
  - Row Level Security (RLS) policies on `api_keys` table
  - Database function `validate_api_key()` for atomic validation
  
- **Integration**:
  - Reuses existing credits system for billing
  - Compatible with current User model from JWT auth
  - Works alongside frontend JWT authentication
  - Integrated into main FastAPI app
