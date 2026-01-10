"""
API Key Management Endpoints
Allows users to create, list, and revoke their API keys.
"""

import os
import logging
from typing import List, Optional
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from auth import get_current_user, User
from api_key_auth import generate_api_key, hash_api_key
from credits import get_credit_manager

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/keys", tags=["API Keys"])

# Debug flag
DEBUG = os.getenv("OELALA_DEBUG", "0") == "1"


def debug_log(msg: str):
    if DEBUG:
        logger.info(f"🔑 API-KEYS: {msg}")


# =============================================================================
# Pydantic Models
# =============================================================================


class CreateAPIKeyRequest(BaseModel):
    """Request to create a new API key."""

    name: str = Field(..., min_length=1, max_length=100, description="Friendly name for the key")
    expires_days: Optional[int] = Field(
        None, ge=1, le=365, description="Days until expiration (optional)"
    )


class CreateAPIKeyResponse(BaseModel):
    """Response with new API key (only shown once!)."""

    id: str
    name: str
    api_key: str = Field(..., description="Full API key - save this! It won't be shown again.")
    key_prefix: str
    created_at: str
    expires_at: Optional[str]


class APIKeyInfo(BaseModel):
    """API key information (without the full key)."""

    id: str
    name: str
    key_prefix: str = Field(..., description="First few characters for identification")
    is_active: bool
    usage_count: int
    last_used_at: Optional[str]
    created_at: str
    expires_at: Optional[str]


class UpdateAPIKeyRequest(BaseModel):
    """Request to update API key."""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    is_active: Optional[bool] = None


# =============================================================================
# Endpoints
# =============================================================================


@router.post("", response_model=CreateAPIKeyResponse)
async def create_api_key(
    request: CreateAPIKeyRequest,
    user: User = Depends(get_current_user),
):
    """
    Create a new API key.

    **Authentication:** Requires JWT token (login required).

    **Important:** The full API key is only shown once during creation.
    Save it securely - you won't be able to retrieve it later!

    **Example:**
    ```bash
    curl -X POST https://oelala.xyz/api/keys \\
      -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
      -H "Content-Type: application/json" \\
      -d '{
        "name": "My Production App",
        "expires_days": 90
      }'
    ```
    """
    debug_log(f"Creating API key for user={user.id}, name={request.name}")

    manager = get_credit_manager()

    if not manager.service_key:
        raise HTTPException(
            status_code=503,
            detail="API key management not available (database not configured)",
        )

    # Generate key
    full_key, key_hash, key_prefix = generate_api_key()

    # Calculate expiration
    expires_at = None
    if request.expires_days:
        expires_at = (datetime.utcnow() + timedelta(days=request.expires_days)).isoformat() + "Z"

    # Insert into database
    try:
        result = manager.supabase.table("api_keys").insert({
            "user_id": user.id,
            "name": request.name,
            "key_hash": key_hash,
            "key_prefix": key_prefix,
            "expires_at": expires_at,
            "is_active": True,
            "usage_count": 0,
        }).execute()

        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to create API key")

        key_record = result.data[0]

        logger.info(f"✅ API key created: {key_record['id']} for user={user.id}")

        return CreateAPIKeyResponse(
            id=key_record["id"],
            name=key_record["name"],
            api_key=full_key,  # Only time this is returned!
            key_prefix=key_record["key_prefix"],
            created_at=key_record["created_at"],
            expires_at=key_record.get("expires_at"),
        )

    except Exception as e:
        logger.error(f"Error creating API key: {e}")
        raise HTTPException(status_code=500, detail="Failed to create API key")


@router.get("", response_model=List[APIKeyInfo])
async def list_api_keys(
    user: User = Depends(get_current_user),
):
    """
    List all API keys for the current user.

    **Authentication:** Requires JWT token (login required).

    **Note:** Full API keys are never returned - only the first few characters
    are shown for identification.
    """
    debug_log(f"Listing API keys for user={user.id}")

    manager = get_credit_manager()

    if not manager.service_key:
        raise HTTPException(
            status_code=503,
            detail="API key management not available (database not configured)",
        )

    try:
        result = manager.supabase.table("api_keys").select("*").eq(
            "user_id", user.id
        ).order("created_at", desc=True).execute()

        keys = []
        for record in result.data:
            keys.append(
                APIKeyInfo(
                    id=record["id"],
                    name=record["name"],
                    key_prefix=record["key_prefix"],
                    is_active=record["is_active"],
                    usage_count=record["usage_count"],
                    last_used_at=record.get("last_used_at"),
                    created_at=record["created_at"],
                    expires_at=record.get("expires_at"),
                )
            )

        return keys

    except Exception as e:
        logger.error(f"Error listing API keys: {e}")
        raise HTTPException(status_code=500, detail="Failed to list API keys")


@router.get("/{key_id}", response_model=APIKeyInfo)
async def get_api_key(
    key_id: str,
    user: User = Depends(get_current_user),
):
    """
    Get details of a specific API key.

    **Authentication:** Requires JWT token (login required).
    """
    debug_log(f"Getting API key {key_id} for user={user.id}")

    manager = get_credit_manager()

    if not manager.service_key:
        raise HTTPException(
            status_code=503,
            detail="API key management not available (database not configured)",
        )

    try:
        result = manager.supabase.table("api_keys").select("*").eq("id", key_id).eq(
            "user_id", user.id
        ).single().execute()

        if not result.data:
            raise HTTPException(status_code=404, detail="API key not found")

        record = result.data

        return APIKeyInfo(
            id=record["id"],
            name=record["name"],
            key_prefix=record["key_prefix"],
            is_active=record["is_active"],
            usage_count=record["usage_count"],
            last_used_at=record.get("last_used_at"),
            created_at=record["created_at"],
            expires_at=record.get("expires_at"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting API key: {e}")
        raise HTTPException(status_code=500, detail="Failed to get API key")


@router.patch("/{key_id}", response_model=APIKeyInfo)
async def update_api_key(
    key_id: str,
    request: UpdateAPIKeyRequest,
    user: User = Depends(get_current_user),
):
    """
    Update an API key (rename or enable/disable).

    **Authentication:** Requires JWT token (login required).

    **Example:** Disable a key
    ```bash
    curl -X PATCH https://oelala.xyz/api/keys/KEY_ID \\
      -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
      -H "Content-Type: application/json" \\
      -d '{"is_active": false}'
    ```
    """
    debug_log(f"Updating API key {key_id} for user={user.id}")

    manager = get_credit_manager()

    if not manager.service_key:
        raise HTTPException(
            status_code=503,
            detail="API key management not available (database not configured)",
        )

    # Build update payload
    updates = {}
    if request.name is not None:
        updates["name"] = request.name
    if request.is_active is not None:
        updates["is_active"] = request.is_active

    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    try:
        # Verify ownership first
        existing = manager.supabase.table("api_keys").select("user_id").eq(
            "id", key_id
        ).single().execute()

        if not existing.data or existing.data["user_id"] != user.id:
            raise HTTPException(status_code=404, detail="API key not found")

        # Update
        result = manager.supabase.table("api_keys").update(updates).eq(
            "id", key_id
        ).execute()

        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=500, detail="Failed to update API key")

        record = result.data[0]

        logger.info(f"✅ API key updated: {key_id} for user={user.id}")

        return APIKeyInfo(
            id=record["id"],
            name=record["name"],
            key_prefix=record["key_prefix"],
            is_active=record["is_active"],
            usage_count=record["usage_count"],
            last_used_at=record.get("last_used_at"),
            created_at=record["created_at"],
            expires_at=record.get("expires_at"),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating API key: {e}")
        raise HTTPException(status_code=500, detail="Failed to update API key")


@router.delete("/{key_id}")
async def delete_api_key(
    key_id: str,
    user: User = Depends(get_current_user),
):
    """
    Delete (revoke) an API key permanently.

    **Authentication:** Requires JWT token (login required).

    **Warning:** This action cannot be undone!

    **Example:**
    ```bash
    curl -X DELETE https://oelala.xyz/api/keys/KEY_ID \\
      -H "Authorization: Bearer YOUR_JWT_TOKEN"
    ```
    """
    debug_log(f"Deleting API key {key_id} for user={user.id}")

    manager = get_credit_manager()

    if not manager.service_key:
        raise HTTPException(
            status_code=503,
            detail="API key management not available (database not configured)",
        )

    try:
        # Verify ownership and delete
        result = manager.supabase.table("api_keys").delete().eq("id", key_id).eq(
            "user_id", user.id
        ).execute()

        if not result.data or len(result.data) == 0:
            raise HTTPException(status_code=404, detail="API key not found")

        logger.info(f"✅ API key deleted: {key_id} for user={user.id}")

        return {"message": "API key deleted successfully", "id": key_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting API key: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete API key")
