"""
Media Service - Unified interface for MinIO storage + Supabase metadata.

This module provides a high-level API for:
- Uploading files to MinIO with automatic Supabase metadata sync
- Generating presigned URLs for temporary public access (via MinIO S3)
- Managing user media (list, delete, update metadata)
- Tracking storage usage per user

Works with the existing user_media table schema from 006_user_media.sql:
- storage_path: Full path in storage (bucket/key format)
- metadata: JSONB for prompt, model, dimensions, etc.

Usage:
    from media_service import MediaService

    service = MediaService()

    # Upload with metadata sync
    media = await service.upload(
        user_id="uuid-here",
        file_data=video_bytes,
        filename="my_video.mp4",
        generation_type="t2v",
        prompt="A cat dancing"
    )

    # Get presigned URL for sharing
    url = service.generate_signed_url(media.storage_path, expires_in=3600)

    # List user's media
    media_list = await service.list_user_media(user_id, media_type="video")
"""

import os
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from dataclasses import dataclass
import mimetypes

import httpx

logger = logging.getLogger(__name__)


# Retention policy: days until media expires per tier
# Storage GC will clean up files after X-Expires-At passes
TIER_RETENTION_DAYS: Dict[str, int] = {
    "free": 30,  # 1 month
    "pro": 90,  # 3 months
    "vip": 365,  # 1 year
}

DEFAULT_RETENTION_DAYS = 30  # Fallback for unknown tiers


@dataclass
class MediaRecord:
    """Represents a media file record (matches 006_user_media.sql schema)."""

    id: str
    user_id: str
    storage_path: str  # Full path: users/{user_id}/videos/filename.mp4
    media_type: str  # video, image, audio
    workflow_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None  # JSONB: prompt, model, dimensions, etc.
    is_nsfw: bool = False
    is_published: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    @property
    def bucket(self) -> str:
        """Extract bucket from storage_path (first two segments: users/{user_id})."""
        parts = self.storage_path.split("/")
        if len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        return parts[0] if parts else ""

    @property
    def key(self) -> str:
        """Extract key from storage_path (everything after bucket)."""
        parts = self.storage_path.split("/", 2)
        return parts[2] if len(parts) > 2 else ""

    @property
    def filename(self) -> str:
        """Extract filename from storage_path."""
        return Path(self.storage_path).name

    @property
    def storage_url(self) -> str:
        """URL path to access via storage backend."""
        return f"/{self.storage_path}"

    # Convenience accessors for metadata
    @property
    def prompt(self) -> Optional[str]:
        return self.metadata.get("prompt") if self.metadata else None

    @property
    def model_name(self) -> Optional[str]:
        return self.metadata.get("model_name") if self.metadata else None

    @property
    def generation_type(self) -> Optional[str]:
        return self.metadata.get("generation_type") if self.metadata else None

    @property
    def width(self) -> Optional[int]:
        return self.metadata.get("width") if self.metadata else None

    @property
    def height(self) -> Optional[int]:
        return self.metadata.get("height") if self.metadata else None

    @property
    def duration_seconds(self) -> Optional[float]:
        return self.metadata.get("duration_seconds") if self.metadata else None

    @property
    def size_bytes(self) -> Optional[int]:
        return self.metadata.get("size_bytes") if self.metadata else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "storage_path": self.storage_path,
            "media_type": self.media_type,
            "workflow_id": self.workflow_id,
            "metadata": self.metadata or {},
            "is_nsfw": self.is_nsfw,
            "is_published": self.is_published,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            # Flattened convenience fields
            "filename": self.filename,
            "prompt": self.prompt,
            "model_name": self.model_name,
            "generation_type": self.generation_type,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.duration_seconds,
            "size_bytes": self.size_bytes,
        }


class MediaService:
    """
    Unified media service that syncs MinIO storage with Supabase metadata.
    """

    def __init__(
        self,
        storage_url: Optional[str] = None,
        storage_token: Optional[str] = None,
        supabase_url: Optional[str] = None,
        supabase_key: Optional[str] = None,
        signing_secret: Optional[str] = None,
    ):
        """
        Initialize media service.

        Args:
            storage_url: MinIO endpoint URL (default: env MINIO_ENDPOINT or localhost:9000)
            storage_token: Ignored (kept for backwards compat)
            supabase_url: Supabase project URL (default: env SUPABASE_URL)
            supabase_key: Supabase service key (default: env SUPABASE_SERVICE_KEY)
            signing_secret: Ignored (MinIO presigned URLs replace custom HMAC)
        """
        self.storage_url = (
            storage_url
            or os.getenv("MINIO_ENDPOINT")
            or os.getenv("STORAGE_URL", "http://localhost:9000")
        ).rstrip("/")
        self.supabase_url = (supabase_url or os.getenv("SUPABASE_URL", "")).rstrip("/")
        self.supabase_key = supabase_key or os.getenv("SUPABASE_SERVICE_KEY")

        self._http_client: Optional[httpx.AsyncClient] = None

        # Lazy-init storage client for presigned URLs and uploads
        self._storage_client = None

    @property
    def storage_client(self):
        """Lazy-init MinIO storage client."""
        if self._storage_client is None:
            from storage_client import get_client
            self._storage_client = get_client()
        return self._storage_client

    @property
    def http_client(self) -> httpx.AsyncClient:
        """Lazy-init async HTTP client (for Supabase REST calls)."""
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=60.0)
        return self._http_client

    async def close(self):
        """Close HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        await self.close()

    def _supabase_headers(self) -> Dict[str, str]:
        """Headers for Supabase requests."""
        if not self.supabase_key:
            return {}
        return {
            "apikey": self.supabase_key,
            "Authorization": f"Bearer {self.supabase_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    async def get_user_tier(self, user_id: str) -> str:
        """
        Fetch user's tier from Supabase user_credits table.

        Args:
            user_id: User's UUID

        Returns:
            Tier string: 'free', 'pro', or 'vip'
        """
        if not self.supabase_url or not self.supabase_key:
            logger.warning("⚠️ Supabase not configured, using default tier")
            return "free"

        try:
            resp = await self.http_client.get(
                f"{self.supabase_url}/rest/v1/user_credits",
                params={"user_id": f"eq.{user_id}", "select": "tier"},
                headers=self._supabase_headers(),
            )

            if resp.status_code == 200:
                data = resp.json()
                if data and isinstance(data, list) and len(data) > 0:
                    tier = data[0].get("tier", "free")
                    logger.debug(f"👤 User {user_id[:8]}... tier: {tier}")
                    return tier

            logger.debug(
                f"👤 No tier found for user {user_id[:8]}..., defaulting to free"
            )
            return "free"

        except Exception as e:
            logger.warning(f"⚠️ Failed to fetch user tier: {e}")
            return "free"

    def calculate_expires_at(self, tier: str) -> datetime:
        """
        Calculate expiration datetime based on user tier.

        Args:
            tier: User tier ('free', 'pro', 'vip')

        Returns:
            Datetime when the file should expire
        """
        retention_days = TIER_RETENTION_DAYS.get(tier, DEFAULT_RETENTION_DAYS)
        return datetime.utcnow() + timedelta(days=retention_days)

    def _detect_media_type(self, mime_type: str) -> str:
        """Detect media type from MIME type."""
        if mime_type.startswith("video/"):
            return "video"
        elif mime_type.startswith("image/"):
            return "image"
        elif mime_type.startswith("audio/"):
            return "audio"
        return "video"  # Default to video for unknown

    def _generate_storage_path(
        self, user_id: str, filename: str, media_type: str
    ) -> str:
        """Generate storage path for a file."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        safe_filename = Path(filename).name.replace(" ", "_")

        # Structure: users/{user_id}/{media_type}s/{timestamp}_{filename}
        type_folder = f"{media_type}s"  # videos, images, audios
        return f"users/{user_id}/{type_folder}/{timestamp}_{safe_filename}"

    async def upload(
        self,
        user_id: str,
        file_data: Union[bytes, Path],
        filename: str,
        mime_type: Optional[str] = None,
        generation_type: Optional[str] = None,
        prompt: Optional[str] = None,
        model_name: Optional[str] = None,
        workflow_id: Optional[str] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        duration_seconds: Optional[float] = None,
        is_nsfw: bool = False,
        user_tier: Optional[str] = None,
    ) -> MediaRecord:
        """
        Upload a file to storage and create Supabase metadata record.

        Args:
            user_id: User's UUID
            file_data: File content (bytes or Path)
            filename: Original filename
            mime_type: MIME type (auto-detected if not provided)
            generation_type: Type of AI generation (t2v, i2v, etc)
            prompt: Generation prompt if applicable
            model_name: Model used for generation
            workflow_id: ComfyUI workflow ID
            width: Media width in pixels
            height: Media height in pixels
            duration_seconds: Duration for video/audio
            is_nsfw: Whether content is NSFW
            user_tier: User tier for retention policy (auto-fetched if not provided)

        Returns:
            MediaRecord with full metadata
        """
        # Read file data
        if isinstance(file_data, Path):
            data = file_data.read_bytes()
        else:
            data = file_data

        # Auto-detect MIME type
        if not mime_type:
            guessed_type, _ = mimetypes.guess_type(filename)
            mime_type = guessed_type or "application/octet-stream"

        media_type = self._detect_media_type(mime_type)
        storage_path = self._generate_storage_path(user_id, filename, media_type)

        # Get user tier for retention policy if not provided
        if user_tier is None:
            user_tier = await self.get_user_tier(user_id)

        # Calculate expiration based on tier
        expires_at = self.calculate_expires_at(user_tier)
        retention_days = TIER_RETENTION_DAYS.get(user_tier, DEFAULT_RETENTION_DAYS)

        # 1. Upload to MinIO via storage client
        # storage_path = "users/{user_id}/{type}s/{timestamp}_{filename}"
        # Split into bucket (first two segments) and key (rest)
        logger.info(
            f"📤 Uploading {filename} to {storage_path} (tier={user_tier}, expires={expires_at.date()})"
        )

        path_parts = storage_path.split("/", 2)
        if len(path_parts) >= 3:
            bucket_path = f"{path_parts[0]}/{path_parts[1]}"
            key_path = path_parts[2]
        else:
            bucket_path = path_parts[0]
            key_path = path_parts[1] if len(path_parts) > 1 else ""

        storage_result = self.storage_client.put(
            bucket_path, key_path, data, content_type=mime_type,
        )
        storage_hash = storage_result.get("hash")

        logger.info(f"✅ Uploaded to storage: {storage_hash}")

        # Build metadata JSONB
        metadata: Dict[str, Any] = {
            "mime_type": mime_type,
            "size_bytes": len(data),
            "storage_hash": storage_hash,
        }
        if generation_type:
            metadata["generation_type"] = generation_type
        if prompt:
            metadata["prompt"] = prompt
        if model_name:
            metadata["model_name"] = model_name
        if width:
            metadata["width"] = width
        if height:
            metadata["height"] = height
        if duration_seconds:
            metadata["duration_seconds"] = duration_seconds

        # Add retention info to metadata
        metadata["tier"] = user_tier
        metadata["retention_days"] = retention_days
        metadata["expires_at"] = expires_at.isoformat() + "Z"

        # 2. Create Supabase metadata record
        record_id: Optional[str] = None
        created_at: Optional[str] = None

        if self.supabase_url and self.supabase_key:
            record_data = {
                "user_id": user_id,
                "storage_path": storage_path,
                "media_type": media_type,
                "workflow_id": workflow_id,
                "metadata": metadata,
                "is_nsfw": is_nsfw,
                "is_published": False,
            }

            supabase_resp = await self.http_client.post(
                f"{self.supabase_url}/rest/v1/user_media",
                json=record_data,
                headers=self._supabase_headers(),
            )

            if supabase_resp.status_code == 201:
                result = supabase_resp.json()
                if isinstance(result, list) and result:
                    result = result[0]
                if isinstance(result, dict):
                    record_id = result.get("id")
                    created_at = result.get("created_at")
                logger.info(f"✅ Supabase record created: {record_id}")
            else:
                logger.error(f"❌ Supabase insert failed: {supabase_resp.text}")
        else:
            # No Supabase, generate local ID
            record_id = (
                storage_hash[:16]
                if storage_hash
                else hashlib.sha256(data).hexdigest()[:16]
            )
            created_at = datetime.utcnow().isoformat()
            logger.warning("⚠️ Supabase not configured, metadata not synced")

        parsed_created_at = None
        if created_at:
            try:
                parsed_created_at = datetime.fromisoformat(
                    created_at.replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        return MediaRecord(
            id=record_id or "",
            user_id=user_id,
            storage_path=storage_path,
            media_type=media_type,
            workflow_id=workflow_id,
            metadata=metadata,
            is_nsfw=is_nsfw,
            is_published=False,
            created_at=parsed_created_at,
        )

    async def get(self, media_id: str) -> Optional[MediaRecord]:
        """Get media record by ID."""
        if not self.supabase_url:
            return None

        resp = await self.http_client.get(
            f"{self.supabase_url}/rest/v1/user_media",
            params={"id": f"eq.{media_id}", "select": "*"},
            headers=self._supabase_headers(),
        )

        if resp.status_code == 200:
            data = resp.json()
            if data and isinstance(data, list) and len(data) > 0:
                return self._record_from_dict(data[0])
        return None

    async def list_user_media(
        self,
        user_id: str,
        media_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "created_at",
        ascending: bool = False,
    ) -> List[MediaRecord]:
        """
        List user's media files.

        Args:
            user_id: User's UUID
            media_type: Filter by type (video, image, audio)
            limit: Max results
            offset: Pagination offset
            order_by: Sort field
            ascending: Sort direction
        """
        if not self.supabase_url:
            return []

        params: Dict[str, Any] = {
            "user_id": f"eq.{user_id}",
            "select": "*",
            "limit": limit,
            "offset": offset,
            "order": f"{order_by}.{'asc' if ascending else 'desc'}",
        }

        if media_type:
            params["media_type"] = f"eq.{media_type}"

        resp = await self.http_client.get(
            f"{self.supabase_url}/rest/v1/user_media",
            params=params,
            headers=self._supabase_headers(),
        )

        if resp.status_code == 200:
            return [self._record_from_dict(item) for item in resp.json()]
        return []

    async def delete(self, media_id: str, hard_delete: bool = False) -> bool:
        """
        Delete media file.

        Args:
            media_id: Media record ID
            hard_delete: If True, delete from storage too. Otherwise just remove record.
        """
        record = await self.get(media_id)
        if not record:
            return False

        if hard_delete:
            # Delete from MinIO storage
            try:
                path_parts = record.storage_path.split("/", 2)
                if len(path_parts) >= 3:
                    bucket_path = f"{path_parts[0]}/{path_parts[1]}"
                    key_path = path_parts[2]
                else:
                    bucket_path = path_parts[0]
                    key_path = path_parts[1] if len(path_parts) > 1 else ""
                self.storage_client.delete(bucket_path, key_path)
            except Exception as e:
                logger.warning(f"Failed to delete from storage: {e}")

        # Delete from Supabase
        if self.supabase_url:
            await self.http_client.delete(
                f"{self.supabase_url}/rest/v1/user_media",
                params={"id": f"eq.{media_id}"},
                headers=self._supabase_headers(),
            )

        return True

    def generate_signed_url(
        self,
        storage_path: str,
        expires_in: int = 3600,
    ) -> str:
        """
        Generate a presigned URL for temporary public access via MinIO.

        Uses MinIO's native S3 presigned URL mechanism (SigV4) which is
        much stronger than the previous custom HMAC-SHA256 scheme.

        Args:
            storage_path: Full storage path (e.g., users/{user_id}/videos/file.mp4)
            expires_in: Expiration time in seconds (default 1 hour)

        Returns:
            Presigned URL that can be used without authentication
        """
        # Split storage_path into bucket + key for the storage client
        path_parts = storage_path.split("/", 2)
        if len(path_parts) >= 3:
            bucket_path = f"{path_parts[0]}/{path_parts[1]}"
            key_path = path_parts[2]
        else:
            bucket_path = path_parts[0]
            key_path = path_parts[1] if len(path_parts) > 1 else ""

        return self.storage_client.presigned_get(
            bucket_path, key_path, expires=expires_in
        )

    async def get_signed_url(
        self, media_id: str, expires_in: int = 3600
    ) -> Optional[str]:
        """
        Get signed URL for a media record.

        Args:
            media_id: Media record ID
            expires_in: Expiration time in seconds

        Returns:
            Signed URL or None if record not found
        """
        record = await self.get(media_id)
        if not record:
            return None
        return self.generate_signed_url(record.storage_path, expires_in)

    async def get_user_quota(self, user_id: str) -> Dict[str, Any]:
        """
        Get storage quota information for a user.

        Calculates usage from Supabase user_media table instead of relying
        on a custom storage endpoint (MinIO has no per-prefix quota API).

        Args:
            user_id: User's UUID

        Returns:
            Dict with quota info
        """
        # Tier-based quota limits
        tier_quotas = {
            "free": 5 * 1024 * 1024 * 1024,     # 5 GB
            "pro": 50 * 1024 * 1024 * 1024,      # 50 GB
            "vip": 200 * 1024 * 1024 * 1024,     # 200 GB
        }

        try:
            # Get user tier
            tier = await self.get_user_tier(user_id)
            quota_bytes = tier_quotas.get(tier, tier_quotas["free"])

            # Calculate usage from Supabase user_media table
            used_bytes = 0
            file_count = 0

            if self.supabase_url and self.supabase_key:
                resp = await self.http_client.get(
                    f"{self.supabase_url}/rest/v1/user_media",
                    params={
                        "user_id": f"eq.{user_id}",
                        "select": "metadata",
                    },
                    headers=self._supabase_headers(),
                )

                if resp.status_code == 200:
                    records = resp.json()
                    file_count = len(records)
                    for record in records:
                        meta = record.get("metadata") or {}
                        used_bytes += meta.get("size_bytes", 0)

            percent = round((used_bytes / quota_bytes) * 100, 1) if quota_bytes > 0 else 0

            return {
                "used_bytes": used_bytes,
                "quota_bytes": quota_bytes,
                "file_count": file_count,
                "tier": tier,
                "used_percent": percent,
                "warning": percent > 80,
                "upgrade_needed": percent > 95,
                "human_used": self._human_size(used_bytes),
                "human_limit": self._human_size(quota_bytes),
            }

        except Exception as e:
            logger.error(f"❌ Failed to get quota for {user_id}: {e}")
            raise

    @staticmethod
    def _human_size(size_bytes: int) -> str:
        """Convert bytes to human-readable format."""
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if size_bytes < 1024:
                return (
                    f"{size_bytes:.1f} {unit}"
                    if size_bytes != int(size_bytes)
                    else f"{int(size_bytes)} {unit}"
                )
            size_bytes /= 1024
        return f"{size_bytes:.1f} PB"

    def _record_from_dict(self, data: Dict[str, Any]) -> MediaRecord:
        """Create MediaRecord from database dictionary."""
        created_at = data.get("created_at")
        updated_at = data.get("updated_at")

        if created_at and isinstance(created_at, str):
            try:
                created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            except ValueError:
                created_at = None

        if updated_at and isinstance(updated_at, str):
            try:
                updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            except ValueError:
                updated_at = None

        return MediaRecord(
            id=data.get("id", ""),
            user_id=data.get("user_id", ""),
            storage_path=data.get("storage_path", ""),
            media_type=data.get("media_type", "video"),
            workflow_id=data.get("workflow_id"),
            metadata=data.get("metadata") or {},
            is_nsfw=data.get("is_nsfw", False),
            is_published=data.get("is_published", False),
            created_at=created_at,
            updated_at=updated_at,
        )


# Singleton instance
_media_service: Optional[MediaService] = None


def get_media_service() -> MediaService:
    """Get or create the global media service instance."""
    global _media_service
    if _media_service is None:
        _media_service = MediaService()
    return _media_service
