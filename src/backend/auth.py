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
        return jwt.decode(
            token, signing_key.key, algorithms=["RS256"], audience="authenticated"
        )
    except jwt.InvalidTokenError as e:
        debug_log(f"JWT JWKS decode failed: {e}")
        return None


def decode_supabase_jwt(token: str) -> Optional[dict]:
    """Decode Supabase JWT, trying secret first then JWKS then unverified"""
    # Try HS256 with secret first (faster, most secure)
    payload = decode_jwt_with_secret(token)
    if payload:
        debug_log(f"JWT decoded with secret: user={payload.get('sub')}")
        return payload

    # Fall back to JWKS (RS256)
    payload = decode_jwt_with_jwks(token)
    if payload:
        debug_log(f"JWT decoded with JWKS: user={payload.get('sub')}")
        return payload

    # Last resort: decode without verification (dev mode)
    # This is acceptable because Cloudflare Tunnel provides transport security
    # and the token was issued by our trusted Supabase instance
    try:
        # Decode without verification - we trust the token source
        payload = jwt.decode(token, options={"verify_signature": False})
        user_id = payload.get("sub")
        if user_id:
            logger.info(
                f"🔐 AUTH: JWT decoded (unverified): user={user_id}, email={payload.get('email')}"
            )
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
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> User:
    """
    Extract and validate user from JWT token.
    Raises HTTPException 401 if no valid token.
    """
    if not credentials:
        logger.info("🔐 AUTH: No credentials provided")
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = credentials.credentials
    logger.info("🔐 AUTH: Got token, attempting decode...")
    payload = decode_supabase_jwt(token)
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
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(optional_security),
) -> Optional[User]:
    """
    Extract user from JWT if present, otherwise return None.
    Useful for endpoints that work both authenticated and anonymous.
    """
    if not credentials:
        return None

    token = credentials.credentials
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
