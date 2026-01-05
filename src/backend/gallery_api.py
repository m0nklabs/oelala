#!/usr/bin/env python3
"""
Gallery API for Oelala
Handles publishing/unpublishing media and fetching gallery content
"""

import os
import logging
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field, validator
from auth import get_current_user, get_optional_user, User

logger = logging.getLogger(__name__)
DEBUG = os.getenv("OELALA_DEBUG", "0") == "1"

def debug_log(msg: str):
    if DEBUG:
        logger.info(f"🖼️ GALLERY: {msg}")

# Pydantic models
class PublishRequest(BaseModel):
    storage_path: str = Field(..., description="Path to media file in user storage")
    title: str = Field(..., min_length=1, max_length=100, description="Title for the media")
    description: Optional[str] = Field(None, max_length=500, description="Optional description")
    tags: List[str] = Field(default=[], description="List of tags")
    is_nsfw: bool = Field(False, description="Whether content is NSFW")
    media_type: str = Field(..., description="Type of media: video, image, or audio")
    thumbnail_url: Optional[str] = Field(None, description="URL to thumbnail")
    metadata: dict = Field(default={}, description="Additional metadata (prompt, settings, etc.)")
    
    @validator('media_type')
    def validate_media_type(cls, v):
        if v not in ['video', 'image', 'audio']:
            raise ValueError('media_type must be one of: video, image, audio')
        return v
    
    @validator('tags')
    def validate_tags(cls, v):
        # Limit to 10 tags
        if len(v) > 10:
            raise ValueError('Maximum 10 tags allowed')
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


class GalleryListResponse(BaseModel):
    items: List[PublishedMediaResponse]
    total: int
    page: int
    per_page: int
    has_more: bool


# Create router
router = APIRouter(prefix="/api/gallery", tags=["gallery"])


# ============================================================================
# Helper: Get Supabase client
# ============================================================================
def get_supabase_client():
    """Get Supabase client (service role for admin operations)"""
    try:
        from supabase import create_client, Client
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")  # Service role key for admin operations
        if not url or not key:
            logger.warning("Supabase credentials not configured")
            return None
        return create_client(url, key)
    except Exception as e:
        logger.error(f"Failed to create Supabase client: {e}")
        return None


# ============================================================================
# Endpoint: Publish media item
# ============================================================================
@router.post("/publish", response_model=PublishedMediaResponse)
async def publish_media(
    request: PublishRequest,
    user: User = Depends(get_current_user)
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
        existing = supabase.table("published_media")\
            .select("id")\
            .eq("user_id", user.id)\
            .eq("storage_path", request.storage_path)\
            .execute()
        
        if existing.data:
            raise HTTPException(
                status_code=400,
                detail="This media item is already published"
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
    except Exception as e:
        logger.error(f"Error publishing media: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Endpoint: Unpublish media item
# ============================================================================
@router.delete("/{media_id}")
async def unpublish_media(
    media_id: str,
    user: User = Depends(get_current_user)
):
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
        result = supabase.table("published_media")\
            .delete()\
            .eq("id", media_id)\
            .eq("user_id", user.id)\
            .execute()
        
        if not result.data:
            raise HTTPException(
                status_code=404,
                detail="Media not found or you don't have permission to unpublish it"
            )
        
        debug_log(f"Unpublished media {media_id} successfully")
        return {"success": True, "message": "Media unpublished successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unpublishing media: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Endpoint: List published media (public gallery)
# ============================================================================
@router.get("", response_model=GalleryListResponse)
async def list_published_media(
    media_type: Optional[str] = Query(None, description="Filter by media type"),
    is_nsfw: Optional[bool] = Query(None, description="Filter by NSFW status"),
    sort_by: str = Query("created_at", description="Sort by: created_at, like_count, view_count"),
    order: str = Query("desc", description="Order: asc or desc"),
    page: int = Query(1, ge=1, description="Page number"),
    per_page: int = Query(30, ge=1, le=100, description="Items per page"),
    user: Optional[User] = Depends(get_optional_user)
):
    """
    List published media items in the gallery.
    Public endpoint (no auth required for SFW content).
    Authenticated users can see NSFW content if is_nsfw filter is set.
    """
    debug_log(f"Listing gallery items: type={media_type}, nsfw={is_nsfw}, sort={sort_by}, user={user.id if user else 'anonymous'}")
    
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=503, detail="Gallery service unavailable")
    
    try:
        # Start query
        query = supabase.table("published_media").select("*", count="exact")
        
        # Filter by media type
        if media_type and media_type in ['video', 'image', 'audio']:
            query = query.eq("media_type", media_type)
        
        # Filter by NSFW status
        # If not authenticated, force SFW only
        if not user:
            query = query.eq("is_nsfw", False)
        elif is_nsfw is not None:
            query = query.eq("is_nsfw", is_nsfw)
        
        # Sort
        if sort_by in ['created_at', 'like_count', 'view_count']:
            ascending = order == "asc"
            query = query.order(sort_by, desc=not ascending)
        
        # Pagination
        start = (page - 1) * per_page
        end = start + per_page - 1
        query = query.range(start, end)
        
        result = query.execute()
        
        items = []
        for item in result.data:
            items.append(PublishedMediaResponse(
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
            ))
        
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
        
    except Exception as e:
        logger.error(f"Error listing gallery: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Endpoint: Get single published media item
# ============================================================================
@router.get("/{media_id}", response_model=PublishedMediaResponse)
async def get_published_media(
    media_id: str,
    user: Optional[User] = Depends(get_optional_user)
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
        # Fetch media
        result = supabase.table("published_media")\
            .select("*")\
            .eq("id", media_id)\
            .execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Media not found")
        
        item = result.data[0]
        
        # Check NSFW access
        if item["is_nsfw"] and not user:
            raise HTTPException(
                status_code=403,
                detail="Login required to view NSFW content"
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
                like_result = supabase.table("published_media_likes")\
                    .select("id")\
                    .eq("media_id", media_id)\
                    .eq("user_id", user.id)\
                    .execute()
                user_liked = len(like_result.data) > 0
            except Exception as e:
                logger.warning(f"Failed to check like status: {e}")
        
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
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting media: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Endpoint: Get user's published media
# ============================================================================
@router.get("/user/{user_id}")
async def get_user_published_media(
    user_id: str,
    page: int = Query(1, ge=1),
    per_page: int = Query(30, ge=1, le=100),
    viewer: Optional[User] = Depends(get_optional_user)
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
        query = supabase.table("published_media")\
            .select("*", count="exact")\
            .eq("user_id", user_id)
        
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
            items.append(PublishedMediaResponse(
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
            ))
        
        total = result.count or 0
        has_more = (start + len(items)) < total
        
        return GalleryListResponse(
            items=items,
            total=total,
            page=page,
            per_page=per_page,
            has_more=has_more,
        )
        
    except Exception as e:
        logger.error(f"Error getting user media: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Endpoint: Toggle like on media
# ============================================================================
@router.post("/{media_id}/like")
async def toggle_like(
    media_id: str,
    user: User = Depends(get_current_user)
):
    """
    Toggle like on a media item.
    If user hasn't liked it, adds a like.
    If user has already liked it, removes the like.
    """
    debug_log(f"Toggling like on media {media_id} for user {user.id}")
    
    supabase = get_supabase_client()
    if not supabase:
        raise HTTPException(status_code=503, detail="Gallery service unavailable")
    
    try:
        # Call the toggle_like function
        result = supabase.rpc("toggle_like", {
            "p_media_id": media_id,
            "p_user_id": user.id
        }).execute()
        
        if not result.data:
            raise HTTPException(status_code=500, detail="Failed to toggle like")
        
        data = result.data[0]
        liked = data["liked"]
        like_count = data["like_count"]
        
        debug_log(f"Like toggled: liked={liked}, count={like_count}")
        
        return {
            "success": True,
            "liked": liked,
            "like_count": like_count,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error toggling like: {e}")
        raise HTTPException(status_code=500, detail=str(e))
