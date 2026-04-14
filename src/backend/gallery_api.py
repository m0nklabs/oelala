#!/usr/bin/env python3
"""
Gallery API for Oelala
Handles publishing/unpublishing media and fetching gallery content
"""

import os
import logging
import re
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field, validator
from auth import get_current_user, get_optional_user, User

logger = logging.getLogger(__name__)
DEBUG = os.getenv("OELALA_DEBUG", "0") == "1"
# Note: thumbnails served from MinIO, no local dir needed


def debug_log(msg: str):
    if DEBUG:
        logger.info(f"🖼️ GALLERY: {msg}")


# Pydantic models
class PublishRequest(BaseModel):
    storage_path: str = Field(..., description="Path to media file in user storage")
    title: str = Field(
        ..., min_length=1, max_length=100, description="Title for the media"
    )
    description: Optional[str] = Field(
        None, max_length=500, description="Optional description"
    )
    tags: List[str] = Field(default=[], description="List of tags")
    is_nsfw: bool = Field(False, description="Whether content is NSFW")
    media_type: str = Field(..., description="Type of media: video, image, or audio")
    thumbnail_url: Optional[str] = Field(None, description="URL to thumbnail")
    metadata: dict = Field(
        default={}, description="Additional metadata (prompt, settings, etc.)"
    )

    @validator("storage_path")
    def validate_storage_path(cls, v):
        # Validate storage path format to prevent path traversal
        # Expected format: "video/filename.mp4", "image/filename.png", etc.
        if not v or ".." in v or v.startswith("/") or "\\" in v:
            raise ValueError("Invalid storage path format")

        # Must match pattern: media_type/filename
        if not re.match(r"^(video|image|audio)/[^/]+\.[a-zA-Z0-9]+$", v):
            raise ValueError("Storage path must be in format: media_type/filename.ext")

        return v

    @validator("media_type")
    def validate_media_type(cls, v, values):
        # Ensure media_type is one of the supported types
        if v not in ["video", "image", "audio"]:
            raise ValueError("media_type must be one of: video, image, audio")

        # Ensure media_type matches the leading segment of storage_path
        storage_path = values.get("storage_path")
        if storage_path:
            path_media_type = storage_path.split("/", 1)[0]
            if path_media_type != v:
                raise ValueError(
                    f'media_type "{v}" must match the media_type segment in storage_path "{storage_path}"'
                )

        return v

    @validator("tags")
    def validate_tags(cls, v):
        # Limit to 10 tags
        if len(v) > 10:
            raise ValueError("Maximum 10 tags allowed")
        # Trim whitespace and filter empty tags
        return [tag.strip() for tag in v if tag.strip()]


class PublishedMediaResponse(BaseModel):
    id: str
    user_id: str
    storage_path: str
    title: str
    description: Optional[str]
    tags: List[str]
    is_nsfw: bool
    media_type: str
    thumbnail_url: Optional[str]
    metadata: dict
    view_count: int
    like_count: int
    created_at: str
    updated_at: str
    # Additional fields for frontend
    user_email: Optional[str] = None
    user_liked: Optional[bool] = None
    # Creator info (from profiles table)
    creator_username: Optional[str] = None
    creator_display_name: Optional[str] = None
    creator_avatar_url: Optional[str] = None


class GalleryListResponse(BaseModel):
    items: List[PublishedMediaResponse]
    total: int
    page: int
    per_page: int
    has_more: bool


# Create router
router = APIRouter(prefix="/api/gallery", tags=["gallery"])


# ============================================================================
# Helper: Get Supabase client (module-level singleton)
# ============================================================================
_supabase_client = None


def get_supabase_client():
    """Get Supabase client (service role for admin operations).
    Uses a singleton pattern to avoid creating new clients on every request."""
    global _supabase_client

    if _supabase_client is not None:
        return _supabase_client

    try:
        from supabase import create_client

        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")  # Service role key for admin operations
        if not url or not key:
            logger.warning("Supabase credentials not configured")
            return None
        _supabase_client = create_client(url, key)
        return _supabase_client
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {e}")
        return None


# ============================================================================
# Endpoint: Publish media item
# ============================================================================
@router.post("/publish", response_model=PublishedMediaResponse)
async def publish_media(
    request: PublishRequest, user: User = Depends(get_current_user)
):
    """
    Publish a media item to the community gallery.
    Requires authentication.
    """
    debug_log(f"Publishing media for user {user.id}: {request.title}")

    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=503, detail="Gallery service unavailable")

    try:
        # Check if already published (same storage_path)
        existing = (
            supabase.table("published_media")
            .select(
                "id,user_id,storage_path,title,description,tags,is_nsfw,media_type,thumbnail_url,metadata,view_count,like_count,created_at,updated_at"
            )
            .eq("user_id", user.id)
            .eq("storage_path", request.storage_path)
            .execute()
        )

        if existing.data:
            # Return existing published item instead of error
            existing_media = existing.data[0]
            debug_log(
                f"Media already published with id {existing_media['id']}, returning existing item"
            )
            return PublishedMediaResponse(
                id=existing_media["id"],
                user_id=existing_media["user_id"],
                storage_path=existing_media["storage_path"],
                title=existing_media["title"],
                description=existing_media.get("description"),
                tags=existing_media.get("tags", []),
                is_nsfw=existing_media["is_nsfw"],
                media_type=existing_media["media_type"],
                thumbnail_url=existing_media.get("thumbnail_url"),
                metadata=existing_media.get("metadata", {}),
                view_count=existing_media.get("view_count", 0),
                like_count=existing_media.get("like_count", 0),
                created_at=existing_media["created_at"],
                updated_at=existing_media["updated_at"],
                user_email=user.email,
            )

        # Insert new published media
        data = {
            "user_id": user.id,
            "storage_path": request.storage_path,
            "title": request.title,
            "description": request.description,
            "tags": request.tags,
            "is_nsfw": request.is_nsfw,
            "media_type": request.media_type,
            "thumbnail_url": request.thumbnail_url,
            "metadata": request.metadata,
        }

        result = supabase.table("published_media").insert(data).execute()

        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to publish media")

        published = result.data[0]
        debug_log(f"Published media {published['id']} successfully")

        return PublishedMediaResponse(
            id=published["id"],
            user_id=published["user_id"],
            storage_path=published["storage_path"],
            title=published["title"],
            description=published.get("description"),
            tags=published.get("tags", []),
            is_nsfw=published["is_nsfw"],
            media_type=published["media_type"],
            thumbnail_url=published.get("thumbnail_url"),
            metadata=published.get("metadata", {}),
            view_count=published.get("view_count", 0),
            like_count=published.get("like_count", 0),
            created_at=published["created_at"],
            updated_at=published["updated_at"],
            user_email=user.email,
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Error publishing media")
        raise HTTPException(status_code=500, detail="Failed to publish media")


# ============================================================================
# Endpoint: Unpublish media item
# ============================================================================
@router.delete("/{media_id}")
async def unpublish_media(media_id: str, user: User = Depends(get_current_user)):
    """
    Unpublish (delete) a media item from the gallery.
    Users can only unpublish their own content.
    """
    debug_log(f"Unpublishing media {media_id} for user {user.id}")

    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=503, detail="Gallery service unavailable")

    try:
        # Delete (RLS ensures user can only delete their own)
        result = (
            supabase.table("published_media")
            .delete()
            .eq("id", media_id)
            .eq("user_id", user.id)
            .execute()
        )

        if not result.data:
            raise HTTPException(
                status_code=404,
                detail="Media not found or you don't have permission to unpublish it",
            )

        debug_log(f"Unpublished media {media_id} successfully")
        return {"success": True, "message": "Media unpublished successfully"}

    except HTTPException:
        raise
    except Exception:
        logger.exception("Error unpublishing media")
        raise HTTPException(status_code=500, detail="Failed to unpublish media")


# ============================================================================
# Endpoint: List published media (public gallery)
# ============================================================================
@router.get("", response_model=GalleryListResponse)
async def list_published_media(
    media_type: Optional[str] = Query(None, description="Filter by media type"),
    is_nsfw: Optional[bool] = Query(None, description="Filter by NSFW status"),
    sort_by: str = Query(
        "created_at", description="Sort by: created_at, like_count, view_count"
    ),
    order: str = Query("desc", description="Order: asc or desc"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(30, ge=1, le=100, description="Items per page"),
    user: Optional[User] = Depends(get_optional_user),
):
    """
    List published media items in the gallery.
    Public endpoint (no auth required for SFW content).
    Authenticated users can see NSFW content if is_nsfw filter is set.
    """
    debug_log(
        f"Listing gallery items: type={media_type}, nsfw={is_nsfw}, sort={sort_by}, user={user.id if user else 'anonymous'}"
    )

    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=503, detail="Gallery service unavailable")

    try:
        # Start query - explicitly select only needed columns
        query = supabase.table("published_media").select(
            "id,user_id,storage_path,title,description,tags,is_nsfw,media_type,thumbnail_url,metadata,view_count,like_count,created_at,updated_at",
            count="exact",
        )

        # Filter by media type
        if media_type and media_type in ["video", "image", "audio"]:
            query = query.eq("media_type", media_type)

        # Filter by NSFW status
        # If not authenticated, force SFW only
        if not user:
            query = query.eq("is_nsfw", False)
        elif is_nsfw is not None:
            query = query.eq("is_nsfw", is_nsfw)

        # Sort
        if sort_by in ["created_at", "like_count", "view_count"]:
            ascending = order == "asc"
            query = query.order(sort_by, desc=not ascending)

        # Pagination
        start = (page - 1) * per_page
        end = start + per_page - 1
        query = query.range(start, end)

        result = query.execute()

        # Batch-fetch user likes for all items in this page (authenticated users only)
        liked_ids: set = set()
        if user and result.data:
            item_ids = [item["id"] for item in result.data]
            try:
                likes_result = (
                    supabase.table("published_media_likes")
                    .select("media_id")
                    .eq("user_id", user.id)
                    .in_("media_id", item_ids)
                    .execute()
                )
                liked_ids = {r["media_id"] for r in likes_result.data}
            except Exception as e:
                logger.warning(f"Failed to batch-fetch user likes: {e}")

        # Batch-fetch creator profiles for all user_ids in this page
        creator_profiles: dict = {}
        if result.data:
            user_ids = list({item["user_id"] for item in result.data})
            try:
                profiles_result = (
                    supabase.table("profiles")
                    .select("id,username,display_name,avatar_url")
                    .in_("id", user_ids)
                    .execute()
                )
                creator_profiles = {p["id"]: p for p in profiles_result.data}
            except Exception as e:
                logger.warning(f"Failed to batch-fetch creator profiles: {e}")

        items = []
        for item in result.data:
            creator = creator_profiles.get(item["user_id"], {})
            items.append(
                PublishedMediaResponse(
                    id=item["id"],
                    user_id=item["user_id"],
                    storage_path=item["storage_path"],
                    title=item["title"],
                    description=item.get("description"),
                    tags=item.get("tags", []),
                    is_nsfw=item["is_nsfw"],
                    media_type=item["media_type"],
                    thumbnail_url=item.get("thumbnail_url"),
                    metadata=item.get("metadata", {}),
                    view_count=item.get("view_count", 0),
                    like_count=item.get("like_count", 0),
                    user_liked=(item["id"] in liked_ids) if user else None,
                    created_at=item["created_at"],
                    updated_at=item["updated_at"],
                    creator_username=creator.get("username"),
                    creator_display_name=creator.get("display_name"),
                    creator_avatar_url=creator.get("avatar_url"),
                )
            )

        total = result.count or 0
        has_more = (start + len(items)) < total

        debug_log(f"Returning {len(items)} items, total={total}, has_more={has_more}")

        return GalleryListResponse(
            items=items,
            total=total,
            page=page,
            per_page=per_page,
            has_more=has_more,
        )

    except Exception:
        logger.exception("Error listing gallery")
        raise HTTPException(status_code=500, detail="Failed to list gallery items")


# ============================================================================
# Endpoint: Get single published media item
# ============================================================================
@router.get("/{media_id}", response_model=PublishedMediaResponse)
async def get_published_media(
    media_id: str, user: Optional[User] = Depends(get_optional_user)
):
    """
    Get details of a single published media item.
    Increments view count.
    """
    debug_log(f"Getting media {media_id}")

    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=503, detail="Gallery service unavailable")

    try:
        # Fetch media - explicitly select only needed columns
        result = (
            supabase.table("published_media")
            .select(
                "id,user_id,storage_path,title,description,tags,is_nsfw,media_type,thumbnail_url,metadata,view_count,like_count,created_at,updated_at"
            )
            .eq("id", media_id)
            .execute()
        )

        if not result.data:
            raise HTTPException(status_code=404, detail="Media not found")

        item = result.data[0]

        # Check NSFW access
        if item["is_nsfw"] and not user:
            raise HTTPException(
                status_code=403, detail="Login required to view NSFW content"
            )

        # Increment view count (async, don't wait)
        try:
            supabase.rpc("increment_view_count", {"p_media_id": media_id}).execute()
        except Exception as e:
            logger.warning(f"Failed to increment view count: {e}")

        # Check if user liked this media
        user_liked = None
        if user:
            try:
                like_result = (
                    supabase.table("published_media_likes")
                    .select("id")
                    .eq("media_id", media_id)
                    .eq("user_id", user.id)
                    .execute()
                )
                user_liked = len(like_result.data) > 0
            except Exception as e:
                logger.warning(f"Failed to check like status: {e}")

        # Fetch creator profile
        creator = {}
        try:
            profile_result = (
                supabase.table("profiles")
                .select("id,username,display_name,avatar_url")
                .eq("id", item["user_id"])
                .execute()
            )
            if profile_result.data:
                creator = profile_result.data[0]
        except Exception as e:
            logger.warning(f"Failed to fetch creator profile: {e}")

        return PublishedMediaResponse(
            id=item["id"],
            user_id=item["user_id"],
            storage_path=item["storage_path"],
            title=item["title"],
            description=item.get("description"),
            tags=item.get("tags", []),
            is_nsfw=item["is_nsfw"],
            media_type=item["media_type"],
            thumbnail_url=item.get("thumbnail_url"),
            metadata=item.get("metadata", {}),
            view_count=item.get("view_count", 0),
            like_count=item.get("like_count", 0),
            created_at=item["created_at"],
            updated_at=item["updated_at"],
            user_liked=user_liked,
            creator_username=creator.get("username"),
            creator_display_name=creator.get("display_name"),
            creator_avatar_url=creator.get("avatar_url"),
        )

    except HTTPException:
        raise
    except Exception:
        logger.exception("Error getting media")
        raise HTTPException(status_code=500, detail="Failed to fetch media item")


# ============================================================================
# Endpoint: Get workflow from published media
# ============================================================================
@router.get("/{media_id}/workflow")
async def get_published_media_workflow(media_id: str):
    """
    Extract and return the ComfyUI workflow JSON from a published media item.
    """
    from pathlib import Path as PathLib

    debug_log(f"Extracting workflow from media {media_id}")

    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=503, detail="Gallery service unavailable")

    try:
        # Fetch media to get storage_path and user_id
        result = (
            supabase.table("published_media")
            .select("user_id,storage_path,media_type")
            .eq("id", media_id)
            .execute()
        )

        if not result.data:
            raise HTTPException(status_code=404, detail="Media not found")

        item = result.data[0]
        user_id = item["user_id"]
        storage_path = item["storage_path"]
        media_type = item["media_type"]

        # Get the actual file from storage
        from storage_client import get_storage_client

        storage = get_storage_client()

        # storage_path is like "video/filename.mp4"
        parts = storage_path.split("/", 1)
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid storage path")

        media_type_dir, filename = parts
        data = storage.get_user_media(user_id, media_type_dir, filename)

        ext = PathLib(filename).suffix.lower()

        # Write to temp file for analysis
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        workflow_json = None

        try:
            if ext in [".mp4", ".webm", ".mov"]:
                # Extract from video metadata using ffprobe
                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "quiet",
                        "-print_format",
                        "json",
                        "-show_format",
                        tmp_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    import json

                    probe_data = json.loads(result.stdout)
                    comment = (
                        probe_data.get("format", {}).get("tags", {}).get("comment", "")
                    )
                    if comment and comment.startswith("{"):
                        workflow_data = json.loads(comment)
                        prompt = workflow_data.get("prompt", workflow_data)
                        # Handle double-encoded JSON
                        if isinstance(prompt, str):
                            workflow_json = json.loads(prompt)
                        else:
                            workflow_json = prompt

            elif ext in [".png"]:
                # Extract from PNG metadata
                from PIL import Image
                import json

                img = Image.open(tmp_path)
                if hasattr(img, "text"):
                    if "prompt" in img.text:
                        workflow_json = json.loads(img.text["prompt"])
                    elif "workflow" in img.text:
                        workflow_json = json.loads(img.text["workflow"])
        finally:
            import os

            os.unlink(tmp_path)

        if workflow_json:
            return {"workflow": workflow_json}
        else:
            raise HTTPException(
                status_code=404, detail="No workflow found in media file"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error extracting workflow")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Endpoint: Get user's published media
# ============================================================================
@router.get("/users/{user_id}")
async def get_user_published_media(
    user_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(30, ge=1, le=100),
    viewer: Optional[User] = Depends(get_optional_user),
):
    """
    Get all published media for a specific user.
    Anonymous users can only see SFW content.
    """
    debug_log(f"Getting published media for user {user_id}")

    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=503, detail="Gallery service unavailable")

    try:
        query = (
            supabase.table("published_media")
            .select(
                "id,user_id,storage_path,title,description,tags,is_nsfw,media_type,thumbnail_url,metadata,view_count,like_count,created_at,updated_at",
                count="exact",
            )
            .eq("user_id", user_id)
        )

        # Anonymous users can only see SFW
        if not viewer:
            query = query.eq("is_nsfw", False)

        # Sort by created_at desc
        query = query.order("created_at", desc=True)

        # Pagination
        start = (page - 1) * per_page
        end = start + per_page - 1
        query = query.range(start, end)

        result = query.execute()

        items = []
        for item in result.data:
            items.append(
                PublishedMediaResponse(
                    id=item["id"],
                    user_id=item["user_id"],
                    storage_path=item["storage_path"],
                    title=item["title"],
                    description=item.get("description"),
                    tags=item.get("tags", []),
                    is_nsfw=item["is_nsfw"],
                    media_type=item["media_type"],
                    thumbnail_url=item.get("thumbnail_url"),
                    metadata=item.get("metadata", {}),
                    view_count=item.get("view_count", 0),
                    like_count=item.get("like_count", 0),
                    created_at=item["created_at"],
                    updated_at=item["updated_at"],
                )
            )

        total = result.count or 0
        has_more = (start + len(items)) < total

        return GalleryListResponse(
            items=items,
            total=total,
            page=page,
            per_page=per_page,
            has_more=has_more,
        )

    except Exception:
        logger.exception("Error getting user media")
        raise HTTPException(status_code=500, detail="Failed to fetch user media")


# ============================================================================
# Endpoint: Toggle like on media
# ============================================================================
@router.post("/{media_id}/like")
async def toggle_like(media_id: str, user: User = Depends(get_current_user)):
    """
    Toggle like on a media item.
    If user hasn't liked it, adds a like.
    If user has already liked it, removes the like.
    Uses auth.uid() internally for security.
    """
    debug_log(f"Toggling like on media {media_id} for user {user.id}")

    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=503, detail="Gallery service unavailable")

    try:
        # First verify the media exists
        media_check = (
            supabase.table("published_media").select("id").eq("id", media_id).execute()
        )

        if not media_check.data:
            raise HTTPException(status_code=404, detail="Media not found")

        # Check if user already liked this item
        existing = (
            supabase.table("published_media_likes")
            .select("id")
            .eq("media_id", media_id)
            .eq("user_id", user.id)
            .execute()
        )

        if existing.data:
            # Already liked — remove like
            supabase.table("published_media_likes").delete().eq(
                "media_id", media_id
            ).eq("user_id", user.id).execute()
            liked = False
        else:
            # Not yet liked — add like
            supabase.table("published_media_likes").insert(
                {"media_id": media_id, "user_id": user.id}
            ).execute()
            liked = True

        # Count actual likes and sync to published_media
        count_result = (
            supabase.table("published_media_likes")
            .select("id", count="exact")
            .eq("media_id", media_id)
            .execute()
        )
        like_count = count_result.count or 0

        supabase.table("published_media").update({"like_count": like_count}).eq(
            "id", media_id
        ).execute()

        debug_log(f"Like toggled: liked={liked}, count={like_count}")

        return {
            "success": True,
            "liked": liked,
            "like_count": like_count,
        }

    except HTTPException:
        raise
    except Exception:
        logger.exception("Error toggling like")
        raise HTTPException(status_code=500, detail="Failed to toggle like")


# ============================================================================
# Endpoint: Serve published media file (PUBLIC - no auth required)
# ============================================================================
@router.get("/{media_id}/file")
async def get_published_media_file(media_id: str):
    """
    Serve the actual media file for a published gallery item.
    This is PUBLIC - no authentication required for published content.

    Media source priority:
    1. MinIO storage (user's cloud storage)
    2. Local directories (media/generated/, ComfyUI/output/) for dev/testing
    """
    from fastapi.responses import StreamingResponse
    from storage_client import get_storage_client

    debug_log(f"Serving media file for {media_id}")

    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=503, detail="Gallery service unavailable")

    try:
        # Fetch media to get storage_path and user_id
        result = (
            supabase.table("published_media")
            .select("user_id,storage_path,media_type")
            .eq("id", media_id)
            .execute()
        )

        if not result.data:
            raise HTTPException(status_code=404, detail="Media not found")

        item = result.data[0]
        user_id = item["user_id"]
        storage_path = item["storage_path"]

        # Parse storage_path (format: "video/filename.mp4")
        parts = storage_path.split("/", 1)
        if len(parts) != 2:
            raise HTTPException(status_code=400, detail="Invalid storage path")

        media_type_dir, filename = parts

        # Determine content type
        ext = Path(filename).suffix.lower()
        content_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".mov": "video/quicktime",
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
        }
        content_type = content_types.get(ext, "application/octet-stream")

        # Try MinIO storage first (check existence eagerly — iter is lazy)
        try:
            storage = get_storage_client()
            bucket = storage.user_bucket(user_id)
            key = storage.user_key(media_type_dir, filename)
            if not storage.exists(bucket, key):
                raise FileNotFoundError(f"Not in storage: {filename}")
            stream = storage.iter_user_media(user_id, media_type_dir, filename)
            debug_log(
                f"Streaming from MinIO: {media_type_dir}/{filename} for user {user_id}"
            )

            return StreamingResponse(
                stream,
                media_type=content_type,
                headers={
                    "Content-Disposition": f'inline; filename="{filename}"',
                    "Cache-Control": "public, max-age=86400",
                },
            )
        except Exception as storage_err:
            debug_log(
                f"MinIO user media failed: {storage_err}, trying storage buckets"
            )

        # Fallback: try generated and comfyui-local storage buckets
        try:
            storage = get_storage_client()
            for bucket in ["generated", "comfyui-local"]:
                try:
                    data = storage.get(bucket, filename)
                    if data:
                        debug_log(f"Serving from storage bucket: {bucket}/{filename}")
                        return Response(
                            content=data,
                            media_type=content_type,
                            headers={
                                "Content-Disposition": f'inline; filename="{filename}"',
                                "Cache-Control": "public, max-age=86400",
                            },
                        )
                except Exception:
                    continue
        except Exception as e:
            debug_log(f"Storage bucket fallback also failed: {e}")

        # Nothing found
        raise HTTPException(status_code=404, detail=f"Media file not found: {filename}")

    except HTTPException:
        raise
    except Exception:
        logger.exception("Error serving media file")
        raise HTTPException(status_code=500, detail="Failed to serve media file")
