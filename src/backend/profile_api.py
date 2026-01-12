#!/usr/bin/env python3
"""
User Profiles API for Oelala
Handles user profile CRUD operations and lookups
"""

import os
import logging
import re
import random
import string
import httpx
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field, validator
from auth import get_current_user, get_optional_user, User

logger = logging.getLogger(__name__)
DEBUG = os.getenv("OELALA_DEBUG", "0") == "1"

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://nsbjwhxdkxnyggtuxjjp.supabase.co")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")


def debug_log(msg: str):
    if DEBUG:
        logger.info(f"👤 PROFILE: {msg}")


# =============================================================================
# Pydantic Models
# =============================================================================


class ProfileCreateRequest(BaseModel):
    """Request to create/update user profile"""

    username: Optional[str] = Field(
        None,
        min_length=3,
        max_length=30,
        description="Unique username (3-30 chars, alphanumeric + _ -)",
    )
    display_name: Optional[str] = Field(
        None, max_length=100, description="Display name shown in UI"
    )
    avatar_url: Optional[str] = Field(None, description="URL to user avatar image")
    bio: Optional[str] = Field(None, max_length=500, description="User biography")
    social_links: Optional[dict] = Field(
        default={}, description="Social media links (twitter, instagram, etc.)"
    )
    is_public: bool = Field(True, description="Whether profile is publicly visible")

    @validator("username")
    def validate_username(cls, v):
        if v is None:
            return v
        # Username must be alphanumeric plus underscore and hyphen
        if not re.match(r"^[a-zA-Z0-9_-]+$", v):
            raise ValueError(
                "Username can only contain letters, numbers, underscores, and hyphens"
            )
        # Cannot start or end with underscore/hyphen
        if v.startswith(("_", "-")) or v.endswith(("_", "-")):
            raise ValueError("Username cannot start or end with underscore or hyphen")
        return v.lower()  # Store usernames in lowercase

    @validator("social_links")
    def validate_social_links(cls, v):
        if v is None:
            return {}
        # Validate social link keys
        allowed_keys = {
            "twitter",
            "instagram",
            "youtube",
            "tiktok",
            "github",
            "discord",
            "website",
        }
        for key in v.keys():
            if key not in allowed_keys:
                raise ValueError(f"Invalid social link key: {key}")
        return v


class ProfileResponse(BaseModel):
    """User profile response"""

    id: str
    username: Optional[str]
    display_name: Optional[str]
    avatar_url: Optional[str]
    bio: Optional[str]
    social_links: dict
    is_public: bool
    created_at: str


class ProfileStats(BaseModel):
    """User profile statistics"""

    total_media: int
    published_media: int
    total_likes_received: int
    total_views_received: int


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/api/profile", tags=["profile"])


# =============================================================================
# Helper Functions
# =============================================================================


async def get_supabase_client() -> httpx.AsyncClient:
    """Get Supabase REST API client"""
    return httpx.AsyncClient(
        base_url=f"{SUPABASE_URL}/rest/v1",
        headers={
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        },
        timeout=30.0,
    )


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/me", response_model=ProfileResponse)
async def get_my_profile(user: User = Depends(get_current_user)):
    """
    Get authenticated user's profile.
    Creates profile if it doesn't exist.
    """
    async with await get_supabase_client() as client:
        # Try to get existing profile
        response = await client.get(
            "/profiles",
            params={"id": f"eq.{user.id}", "select": "*"},
        )

        if response.status_code == 200 and response.json():
            profile = response.json()[0]
            debug_log(f"Retrieved profile for user {user.id}")
            return ProfileResponse(**profile)

        # Profile doesn't exist - create default one
        debug_log(f"Creating new profile for user {user.id}")

        # Generate default username from email
        default_username = user.email.split("@")[0] if user.email else "user"
        default_username = re.sub(r"[^a-zA-Z0-9_-]", "", default_username).lower()

        # Ensure username meets minimum length requirements (>= 3 chars)
        if not default_username or len(default_username) < 3:
            # Fallback if email prefix was fully stripped by sanitization
            if not default_username:
                default_username = "user"
            # Pad to minimum length if needed
            if len(default_username) < 3:
                padding_length = 3 - len(default_username)
                padding_chars = string.ascii_lowercase + string.digits
                padding = "".join(random.choices(padding_chars, k=padding_length))
                default_username = f"{default_username}{padding}"
                debug_log(
                    f"Padded short default username to meet length constraint: {default_username}"
                )

        # Ensure uniqueness
        check_response = await client.get(
            "/profiles",
            params={"username": f"eq.{default_username}", "select": "username"},
        )
        if check_response.status_code == 200 and check_response.json():
            # Username taken, append random suffix
            suffix = "".join(random.choices(string.digits, k=4))
            default_username = f"{default_username}_{suffix}"

        # Create profile
        create_response = await client.post(
            "/profiles",
            json={
                "id": user.id,
                "username": default_username,
                "display_name": user.email.split("@")[0] if user.email else "User",
                "is_public": True,
            },
        )

        if create_response.status_code not in (200, 201):
            logger.error(f"Failed to create profile: {create_response.text}")
            raise HTTPException(status_code=500, detail="Failed to create user profile")

        profile = create_response.json()[0]
        return ProfileResponse(**profile)


@router.put("/me", response_model=ProfileResponse)
async def update_my_profile(
    profile_data: ProfileCreateRequest,
    user: User = Depends(get_current_user),
):
    """
    Update authenticated user's profile.
    """
    async with await get_supabase_client() as client:
        # Prepare update data (only include non-None fields)
        update_data = {k: v for k, v in profile_data.dict().items() if v is not None}

        if not update_data:
            raise HTTPException(status_code=400, detail="No fields to update")

        # Check username uniqueness if username is being updated
        if "username" in update_data:
            check_response = await client.get(
                "/profiles",
                params={
                    "username": f"eq.{update_data['username']}",
                    "id": f"neq.{user.id}",
                    "select": "username",
                },
            )
            if check_response.status_code == 200 and check_response.json():
                raise HTTPException(status_code=400, detail="Username already taken")

        # Update profile
        response = await client.patch(
            "/profiles",
            params={"id": f"eq.{user.id}"},
            json=update_data,
        )

        if response.status_code not in (200, 204):
            logger.error(f"Failed to update profile: {response.text}")
            raise HTTPException(status_code=500, detail="Failed to update profile")

        # Fetch updated profile
        get_response = await client.get(
            "/profiles",
            params={"id": f"eq.{user.id}", "select": "*"},
        )

        if get_response.status_code == 200 and get_response.json():
            profile = get_response.json()[0]
            debug_log(f"Updated profile for user {user.id}")
            return ProfileResponse(**profile)

        raise HTTPException(status_code=404, detail="Profile not found after update")


@router.get("/username/{username}", response_model=ProfileResponse)
async def get_profile_by_username(
    username: str,
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Get user profile by username.
    Only public profiles are visible to non-owners.
    """
    async with await get_supabase_client() as client:
        response = await client.get(
            "/profiles",
            params={"username": f"eq.{username.lower()}", "select": "*"},
        )

        if response.status_code != 200 or not response.json():
            raise HTTPException(status_code=404, detail="Profile not found")

        profile = response.json()[0]

        # Check if user can view this profile
        is_owner = current_user and current_user.id == profile["id"]
        is_public = profile.get("is_public", True)

        if not is_public and not is_owner:
            raise HTTPException(status_code=403, detail="This profile is private")

        debug_log(f"Retrieved profile for username {username}")
        return ProfileResponse(**profile)


@router.get("/id/{user_id}", response_model=ProfileResponse)
async def get_profile_by_id(
    user_id: str,
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Get user profile by user ID.
    Only public profiles are visible to non-owners.
    """
    async with await get_supabase_client() as client:
        response = await client.get(
            "/profiles",
            params={"id": f"eq.{user_id}", "select": "*"},
        )

        if response.status_code != 200 or not response.json():
            raise HTTPException(status_code=404, detail="Profile not found")

        profile = response.json()[0]

        # Check if user can view this profile
        is_owner = current_user and current_user.id == profile["id"]
        is_public = profile.get("is_public", True)

        if not is_public and not is_owner:
            raise HTTPException(status_code=403, detail="This profile is private")

        debug_log(f"Retrieved profile for user_id {user_id}")
        return ProfileResponse(**profile)


@router.get("/me/stats", response_model=ProfileStats)
async def get_my_stats(user: User = Depends(get_current_user)):
    """
    Get statistics for authenticated user's content.
    """
    async with await get_supabase_client() as client:
        # Get total media count
        media_response = await client.get(
            "/user_media",
            params={
                "user_id": f"eq.{user.id}",
                "select": "id,is_published",
            },
        )

        total_media = 0
        published_media = 0
        if media_response.status_code == 200:
            media_list = media_response.json()
            total_media = len(media_list)
            published_media = sum(1 for m in media_list if m.get("is_published"))

        # Get likes and views from published media
        published_response = await client.get(
            "/published_media",
            params={
                "user_id": f"eq.{user.id}",
                "select": "like_count,view_count",
            },
        )

        total_likes = 0
        total_views = 0
        if published_response.status_code == 200:
            published_list = published_response.json()
            total_likes = sum(p.get("like_count", 0) for p in published_list)
            total_views = sum(p.get("view_count", 0) for p in published_list)

        debug_log(
            f"Stats for user {user.id}: {total_media} media, {published_media} published"
        )

        return ProfileStats(
            total_media=total_media,
            published_media=published_media,
            total_likes_received=total_likes,
            total_views_received=total_views,
        )


@router.delete("/me")
async def delete_my_profile(user: User = Depends(get_current_user)):
    """
    Delete authenticated user's profile.
    WARNING: This does not delete the auth user, only the profile data.
    """
    async with await get_supabase_client() as client:
        response = await client.delete(
            "/profiles",
            params={"id": f"eq.{user.id}"},
        )

        if response.status_code not in (200, 204):
            logger.error(f"Failed to delete profile: {response.text}")
            raise HTTPException(status_code=500, detail="Failed to delete profile")

        debug_log(f"Deleted profile for user {user.id}")
        return {"success": True, "message": "Profile deleted successfully"}


# =============================================================================
# Admin Endpoints
# =============================================================================


@router.get("/admin/list")
async def list_all_profiles(
    user: User = Depends(get_current_user),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    """
    List all user profiles (admin only).
    """
    # Check if user is admin
    async with await get_supabase_client() as client:
        admin_check = await client.get(
            "/user_credits",
            params={"user_id": f"eq.{user.id}", "select": "is_admin"},
        )

        if admin_check.status_code != 200 or not admin_check.json():
            raise HTTPException(status_code=403, detail="Admin access required")

        is_admin = admin_check.json()[0].get("is_admin", False)
        if not is_admin:
            raise HTTPException(status_code=403, detail="Admin access required")

        # Get all profiles
        response = await client.get(
            "/profiles",
            params={
                "select": "*",
                "order": "created_at.desc",
                "limit": limit,
                "offset": offset,
            },
        )

        if response.status_code != 200:
            logger.error(f"Failed to fetch profiles: {response.text}")
            raise HTTPException(status_code=500, detail="Failed to fetch profiles")

        profiles = response.json()
        debug_log(f"Admin listed {len(profiles)} profiles")

        return {
            "profiles": profiles,
            "count": len(profiles),
            "limit": limit,
            "offset": offset,
        }
