#!/usr/bin/env python3
"""
User Profiles API for Oelala
Handles user profile CRUD operations and lookups
"""

import os
import io
import logging
import re
import random
import string
from pathlib import Path
import httpx
from typing import Optional
from fastapi import APIRouter, HTTPException, Depends, Query, File, UploadFile
from pydantic import BaseModel, Field, validator
from PIL import Image
from auth import get_current_user, get_optional_user, User

AVATARS_DIR = Path("/home/flip/oelala/media/avatars")
AVATARS_DIR.mkdir(parents=True, exist_ok=True)
AVATAR_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
AVATAR_SIZE = 256  # square px

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
        min_length=1,
        max_length=30,
        description="Unique username (1-30 chars, alphanumeric + _ -)",
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
    follower_count: int = 0
    following_count: int = 0


class FollowResponse(BaseModel):
    """Follow action response"""

    followed: bool
    follower_count: int
    following_count: int


class FollowListItem(BaseModel):
    """User in a followers/following list"""

    id: str
    username: Optional[str] = None
    display_name: Optional[str] = None
    avatar_url: Optional[str] = None
    bio: Optional[str] = None


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

        # Remove leading/trailing underscores and hyphens
        default_username = default_username.strip("_-")

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

        # Get follower/following counts from profile
        follower_count = 0
        following_count = 0
        try:
            profile_response = await client.get(
                "/profiles",
                params={
                    "id": f"eq.{user.id}",
                    "select": "follower_count,following_count",
                },
            )

            if profile_response.status_code == 200 and profile_response.json():
                p = profile_response.json()[0]
                follower_count = p.get("follower_count", 0)
                following_count = p.get("following_count", 0)
        except Exception:
            # Columns may not exist yet (migration 009 not applied)
            pass

        return ProfileStats(
            total_media=total_media,
            published_media=published_media,
            total_likes_received=total_likes,
            total_views_received=total_views,
            follower_count=follower_count,
            following_count=following_count,
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
# Avatar Upload
# =============================================================================


@router.post("/me/avatar")
async def upload_avatar(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    """
    Upload a profile avatar image.
    Accepts JPEG/PNG/WebP/GIF, max 5 MB.
    Image is cropped to a centered square and saved as 256×256 JPEG.
    """
    # Validate content type
    allowed = {"image/jpeg", "image/png", "image/webp", "image/gif"}
    if file.content_type not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{file.content_type}'. Use JPEG, PNG, WebP or GIF.",
        )

    # Read and size-check
    data = await file.read()
    if len(data) > AVATAR_MAX_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(data) // 1024} KB). Max 5 MB.",
        )

    # Open + crop to centered square + resize
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
        w, h = img.size
        short = min(w, h)
        left = (w - short) // 2
        top = (h - short) // 2
        img = img.crop((left, top, left + short, top + short))
        img = img.resize((AVATAR_SIZE, AVATAR_SIZE), Image.LANCZOS)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not process image: {exc}")

    # Save to disk
    dest = AVATARS_DIR / f"{user.id}.jpg"
    img.save(str(dest), "JPEG", quality=85, optimize=True)
    debug_log(f"👤 Saved avatar for user {user.id} → {dest}")

    # Build public URL (served via /avatars/ static mount)
    avatar_url = f"/avatars/{user.id}.jpg"

    # Persist in Supabase
    async with await get_supabase_client() as client:
        resp = await client.patch(
            "/profiles",
            params={"id": f"eq.{user.id}"},
            json={"avatar_url": avatar_url},
        )
        if resp.status_code not in (200, 204):
            logger.error(f"Failed to update avatar_url in Supabase: {resp.text}")
            raise HTTPException(
                status_code=500, detail="Saved image but failed to update profile"
            )

    return {"avatar_url": avatar_url}


# =============================================================================
# Follow / Unfollow
# =============================================================================


@router.post("/{user_id}/follow", response_model=FollowResponse)
async def follow_user(user_id: str, user: User = Depends(get_current_user)):
    """Follow another user."""
    if user.id == user_id:
        raise HTTPException(status_code=400, detail="Cannot follow yourself")

    async with await get_supabase_client() as client:
        # Check target user exists
        target = await client.get(
            "/profiles", params={"id": f"eq.{user_id}", "select": "id"}
        )
        if target.status_code != 200 or not target.json():
            raise HTTPException(status_code=404, detail="User not found")

        # Check if already following
        existing = await client.get(
            "/follows",
            params={
                "follower_id": f"eq.{user.id}",
                "following_id": f"eq.{user_id}",
                "select": "follower_id",
            },
        )
        if existing.status_code == 200 and existing.json():
            raise HTTPException(status_code=409, detail="Already following this user")

        # Insert follow (trigger auto-updates counts)
        resp = await client.post(
            "/follows",
            json={"follower_id": user.id, "following_id": user_id},
        )
        if resp.status_code not in (200, 201):
            logger.error(f"Failed to follow: {resp.text}")
            raise HTTPException(status_code=500, detail="Failed to follow user")

        # Fetch updated counts
        profile_resp = await client.get(
            "/profiles",
            params={"id": f"eq.{user_id}", "select": "follower_count,following_count"},
        )
        counts = (
            profile_resp.json()[0]
            if profile_resp.status_code == 200 and profile_resp.json()
            else {}
        )

        debug_log(f"User {user.id} followed {user_id}")
        return FollowResponse(
            followed=True,
            follower_count=counts.get("follower_count", 0),
            following_count=counts.get("following_count", 0),
        )


@router.delete("/{user_id}/follow", response_model=FollowResponse)
async def unfollow_user(user_id: str, user: User = Depends(get_current_user)):
    """Unfollow a user."""
    if user.id == user_id:
        raise HTTPException(status_code=400, detail="Cannot unfollow yourself")

    async with await get_supabase_client() as client:
        # Delete follow (trigger auto-updates counts)
        resp = await client.delete(
            "/follows",
            params={
                "follower_id": f"eq.{user.id}",
                "following_id": f"eq.{user_id}",
            },
        )
        if resp.status_code not in (200, 204):
            logger.error(f"Failed to unfollow: {resp.text}")
            raise HTTPException(status_code=500, detail="Failed to unfollow user")

        # Fetch updated counts
        profile_resp = await client.get(
            "/profiles",
            params={"id": f"eq.{user_id}", "select": "follower_count,following_count"},
        )
        counts = (
            profile_resp.json()[0]
            if profile_resp.status_code == 200 and profile_resp.json()
            else {}
        )

        debug_log(f"User {user.id} unfollowed {user_id}")
        return FollowResponse(
            followed=False,
            follower_count=counts.get("follower_count", 0),
            following_count=counts.get("following_count", 0),
        )


@router.get("/{user_id}/followers")
async def get_followers(
    user_id: str,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _current_user: Optional[User] = Depends(get_optional_user),
):
    """Get user's followers list."""
    async with await get_supabase_client() as client:
        # Get follower IDs
        resp = await client.get(
            "/follows",
            params={
                "following_id": f"eq.{user_id}",
                "select": "follower_id,created_at",
                "order": "created_at.desc",
                "limit": limit,
                "offset": offset,
            },
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to get followers")

        follows = resp.json()
        if not follows:
            return {"followers": [], "count": 0}

        # Get profiles for these follower IDs
        follower_ids = [f["follower_id"] for f in follows]
        ids_filter = ",".join(follower_ids)
        profiles_resp = await client.get(
            "/profiles",
            params={
                "id": f"in.({ids_filter})",
                "select": "id,username,display_name,avatar_url,bio",
            },
        )
        profiles = profiles_resp.json() if profiles_resp.status_code == 200 else []

        return {"followers": profiles, "count": len(profiles)}


@router.get("/{user_id}/following")
async def get_following(
    user_id: str,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    _current_user: Optional[User] = Depends(get_optional_user),
):
    """Get list of users this user follows."""
    async with await get_supabase_client() as client:
        resp = await client.get(
            "/follows",
            params={
                "follower_id": f"eq.{user_id}",
                "select": "following_id,created_at",
                "order": "created_at.desc",
                "limit": limit,
                "offset": offset,
            },
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to get following")

        follows = resp.json()
        if not follows:
            return {"following": [], "count": 0}

        following_ids = [f["following_id"] for f in follows]
        ids_filter = ",".join(following_ids)
        profiles_resp = await client.get(
            "/profiles",
            params={
                "id": f"in.({ids_filter})",
                "select": "id,username,display_name,avatar_url,bio",
            },
        )
        profiles = profiles_resp.json() if profiles_resp.status_code == 200 else []

        return {"following": profiles, "count": len(profiles)}


@router.get("/{user_id}/is-following")
async def check_is_following(
    user_id: str,
    user: User = Depends(get_current_user),
):
    """Check if the authenticated user is following a specific user."""
    async with await get_supabase_client() as client:
        resp = await client.get(
            "/follows",
            params={
                "follower_id": f"eq.{user.id}",
                "following_id": f"eq.{user_id}",
                "select": "follower_id",
            },
        )
        is_following = resp.status_code == 200 and bool(resp.json())
        return {"is_following": is_following}


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
