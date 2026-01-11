"""
API Key Authentication for Oelala REST API v1
Validates API keys for programmatic access.
"""

import os
import hashlib
import secrets
import logging
from typing import Optional
from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

from auth import User  # Reuse User model from JWT auth

logger = logging.getLogger(__name__)

# Debug flag
DEBUG = os.getenv("OELALA_DEBUG", "0") == "1"


def debug_log(msg: str):
    if DEBUG:
        logger.info(f"🔑 API-KEY: {msg}")


class APIKey(BaseModel):
    """API key information"""

    id: str
    user_id: str
    name: str
    key_prefix: str
    is_active: bool
    usage_count: int
    last_used_at: Optional[str] = None
    created_at: str
    expires_at: Optional[str] = None


# API Key header scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def generate_api_key() -> tuple[str, str, str]:
    """
    Generate a new API key.

    Returns:
        tuple: (full_key, key_hash, key_prefix)
            - full_key: The actual key to give to the user (starts with "oelala_")
            - key_hash: SHA-256 hash to store in database
            - key_prefix: First 15 chars of the full key for display (e.g., "oelala_12345678")
    """
    # Generate 32 random bytes = 64 hex chars
    random_part = secrets.token_hex(32)
    full_key = f"oelala_{random_part}"

    # Hash for storage
    key_hash = hashlib.sha256(full_key.encode()).hexdigest()

    # Prefix for display (first 8 chars after "oelala_")
    key_prefix = full_key[:15]  # "oelala_" + first 8 random chars

    return full_key, key_hash, key_prefix


def hash_api_key(api_key: str) -> str:
    """Hash an API key for storage/validation."""
    return hashlib.sha256(api_key.encode()).hexdigest()


async def validate_api_key_db(key_hash: str) -> Optional[tuple[str, str]]:
    """
    Validate API key against database.

    Args:
        key_hash: SHA-256 hash of the API key

    Returns:
        Optional[tuple[str, str]]: (user_id, key_id) if valid, None otherwise
    """
    # Import here to avoid circular dependency
    from credits import get_credit_manager

    manager = get_credit_manager()

    if not manager.service_key:
        logger.warning("🔑 SUPABASE_SERVICE_KEY not configured - API key auth disabled")
        return None

    try:
        # Call Supabase function to validate key
        result = manager.supabase.rpc(
            "validate_api_key", {"p_key_hash": key_hash}
        ).execute()

        if not result.data or len(result.data) == 0:
            debug_log("No API key found for hash")
            return None

        record = result.data[0]

        if not record.get("valid"):
            error = record.get("error", "Unknown error")
            debug_log(f"API key validation failed: {error}")
            return None

        user_id = record.get("user_id")
        key_id = record.get("key_id")

        if not user_id or not key_id:
            debug_log("API key validation returned incomplete data")
            return None

        debug_log(f"API key validated: user={user_id}, key={key_id}")
        return (str(user_id), str(key_id))

    except Exception as e:
        logger.error(f"Error validating API key: {e}")
        return None


async def get_api_key_user(api_key: Optional[str] = Security(api_key_header)) -> User:
    """
    Extract and validate user from API key.
    Raises HTTPException 401 if invalid.

    This is the main dependency for API v1 endpoints.
    """
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Provide via X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Validate format
    if not api_key.startswith("oelala_"):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key format. Key must start with 'oelala_'",
        )

    # Hash and validate
    key_hash = hash_api_key(api_key)
    result = await validate_api_key_db(key_hash)

    if not result:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired API key",
        )

    user_id, key_id = result

    # Return User object (compatible with existing credit/auth system)
    return User(
        id=user_id,
        email=None,  # API keys don't have email in context
        role="authenticated",
        app_metadata={"api_key_id": key_id},
    )


async def get_optional_api_key_user(
    api_key: Optional[str] = Security(api_key_header),
) -> Optional[User]:
    """
    Extract user from API key if present, otherwise return None.
    Useful for endpoints that work both authenticated and anonymous.
    """
    if not api_key:
        return None

    try:
        return await get_api_key_user(api_key)
    except HTTPException:
        return None
