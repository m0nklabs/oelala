"""
Supabase JWT Authentication for Oelala Backend
Validates JWT tokens from frontend and extracts user information.
"""

import os
import logging
from typing import Optional
from functools import lru_cache
from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from jwt import PyJWKClient
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Debug flag
DEBUG = os.getenv("OELALA_DEBUG", "0") == "1"


def debug_log(msg: str):
    if DEBUG:
        logger.info(f"🔐 AUTH: {msg}")


class User(BaseModel):
    """Authenticated user from Supabase JWT"""

    id: str  # Supabase user ID (UUID)
    email: Optional[str] = None
    role: str = "authenticated"
    app_metadata: dict = {}
    user_metadata: dict = {}


# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://nsbjwhxdkxnyggtuxjjp.supabase.co")
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "")

# Whether a token that failed signature verification may still be trusted.
# Defaults to ON only so this change cannot lock anyone out before the ES256
# JWKS path is confirmed in production; set AUTH_ALLOW_UNVERIFIED_JWT=0 to
# close the authentication bypass described in decode_supabase_jwt().
ALLOW_UNVERIFIED_JWT = os.getenv("AUTH_ALLOW_UNVERIFIED_JWT", "1").lower() in (
    "1",
    "true",
    "yes",
)

# JWT Key URL for Supabase (JWKS endpoint)
JWKS_URL = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"


@lru_cache(maxsize=1)
def get_jwk_client() -> Optional[PyJWKClient]:
    """Get cached JWK client for Supabase"""
    try:
        return PyJWKClient(JWKS_URL)
    except Exception as e:
        logger.warning(f"Failed to initialize JWK client: {e}")
        return None


def decode_jwt_with_secret(token: str) -> Optional[dict]:
    """Decode JWT using Supabase JWT secret (faster, local verification)"""
    if not SUPABASE_JWT_SECRET:
        return None
    try:
        return jwt.decode(
            token, SUPABASE_JWT_SECRET, algorithms=["HS256"], audience="authenticated"
        )
    except jwt.InvalidTokenError as e:
        debug_log(f"JWT secret decode failed: {e}")
        return None


def decode_jwt_with_jwks(token: str) -> Optional[dict]:
    """Decode JWT using Supabase JWKS (remote key verification)"""
    client = get_jwk_client()
    if not client:
        return None
    try:
        signing_key = client.get_signing_key_from_jwt(token)
        # Supabase issues ES256 (EC P-256) tokens. Restricting this list to
        # RS256 made every legitimate token fail verification with
        # InvalidAlgorithmError, which silently pushed all traffic onto the
        # unverified fallback below and turned that fallback into the de-facto
        # authentication path.
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=["ES256", "ES384", "RS256", "RS384"],
            audience="authenticated",
        )
    except (jwt.InvalidTokenError, jwt.exceptions.PyJWKClientError, Exception) as e:
        debug_log(f"JWT JWKS decode failed: {e}")
        return None


def decode_supabase_jwt(token: str) -> Optional[dict]:
    """Decode Supabase JWT, trying secret first then JWKS then unverified"""
    # Try HS256 with secret first (faster, most secure)
    payload = decode_jwt_with_secret(token)
    if payload:
        debug_log(f"JWT decoded with secret: user={payload.get('sub')}")
        return payload

    # Fall back to JWKS (ES256 / RS256)
    payload = decode_jwt_with_jwks(token)
    if payload:
        logger.info(f"🔐 AUTH: JWT verified via JWKS: user={payload.get('sub')}")
        return payload

    if not ALLOW_UNVERIFIED_JWT:
        logger.warning(
            "🔐 AUTH: token failed signature verification and the unverified "
            "fallback is disabled (AUTH_ALLOW_UNVERIFIED_JWT=0) — rejecting"
        )
        return None

    # Last resort: decode without verification.
    #
    # SECURITY WARNING: this accepts ANY token whose payload merely carries a
    # `sub`, with no signature check at all — anyone able to reach the API can
    # impersonate any user id. It was justified by "Cloudflare Tunnel provides
    # transport security", but a tunnel provides TLS and origin hiding, not
    # issuer authentication: a forged token passes through it unchanged. The
    # fallback used to be reached on EVERY request because the JWKS path only
    # allowed RS256 while Supabase issues ES256, so this branch was the de-facto
    # auth path rather than an emergency escape hatch. Keep it off.
    logger.warning(
        "🔐 AUTH: JWT decoded WITHOUT signature verification (unverified "
        "fallback) — this accepts forged tokens; set AUTH_ALLOW_UNVERIFIED_JWT=0"
    )
    try:
        # Decode without verification - we trust the token source
        payload = jwt.decode(token, options={"verify_signature": False})
        user_id = payload.get("sub")
        if user_id:
            return payload
    except Exception as e:
        logger.warning(f"🔐 AUTH: JWT decode failed completely: {e}")

    return None


class OptionalHTTPBearer(HTTPBearer):
    """HTTP Bearer that doesn't fail on missing auth"""

    async def __call__(
        self, request: Request
    ) -> Optional[HTTPAuthorizationCredentials]:
        try:
            return await super().__call__(request)
        except HTTPException:
            return None


# Security scheme
security = HTTPBearer(auto_error=False)
optional_security = OptionalHTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> User:
    """
    Extract and validate user from JWT token.
    Raises HTTPException 401 if no valid token.
    """
    token = None
    if credentials:
        token = credentials.credentials
    elif "token" in request.query_params:
        token = request.query_params.get("token")

    if not token:
        logger.info("🔐 AUTH: No credentials provided")
        raise HTTPException(status_code=401, detail="Not authenticated")

    logger.info("🔐 AUTH: Got token, attempting decode...")
    payload = decode_supabase_jwt(token)

    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")

    return User(
        id=payload.get("sub", ""),
        email=payload.get("email"),
        role=payload.get("role", "authenticated"),
        app_metadata=payload.get("app_metadata", {}),
        user_metadata=payload.get("user_metadata", {}),
    )


async def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_security),
) -> Optional[User]:
    """
    Extract user from JWT if present, otherwise return None.
    Useful for endpoints that work both authenticated and anonymous.
    """
    token = None
    if credentials:
        token = credentials.credentials
    elif "token" in request.query_params:
        token = request.query_params.get("token")

    if not token:
        return None

    payload = decode_supabase_jwt(token)

    if not payload:
        return None

    return User(
        id=payload.get("sub", ""),
        email=payload.get("email"),
        role=payload.get("role", "authenticated"),
        app_metadata=payload.get("app_metadata", {}),
        user_metadata=payload.get("user_metadata", {}),
    )


# System user for internal operations (e.g., ComfyUI callbacks)
SYSTEM_USER = User(
    id="system",
    email="system@oelala.xyz",
    role="service",
    app_metadata={"is_system": True},
)
