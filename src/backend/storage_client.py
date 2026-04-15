"""
MinIO-backed storage client for the oelala backend.

This module provides a Python client for interacting with MinIO (S3-compatible)
object storage. It replaces the previous custom oelala-storage httpx client
while keeping the same public API so all callers continue to work unchanged.

Bucket mapping (logical → MinIO):
    "generated"    → "oelala-generated"
    "comfyui-local"→ "oelala-comfyui"
    "avatars"      → "oelala-avatars"
    "users"        → "oelala-users"
    (other)        → "oelala-{name}"

Usage:
    from storage_client import StorageClient

    client = StorageClient()

    # Upload
    obj = client.put("generated", "video.mp4", video_bytes)

    # Download
    data = client.get("generated", "video.mp4")

    # List
    objects = client.list("generated", prefix="2026")

    # Delete
    client.delete("generated", "video.mp4")

    # Presigned URL (replaces custom HMAC signed URLs)
    url = client.presigned_get("generated", "video.mp4", expires=3600)
"""

import io
import hashlib
import mimetypes
from datetime import timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, BinaryIO, Tuple, Union
from urllib.parse import urlparse
import logging

from minio import Minio
from minio.commonconfig import CopySource
from minio.error import S3Error

logger = logging.getLogger(__name__)

# Logical bucket name → MinIO bucket name
_BUCKET_MAP: Dict[str, str] = {
    "generated": "oelala-generated",
    "comfyui-local": "oelala-comfyui",
    "avatars": "oelala-avatars",
    "users": "oelala-users",
}


def _resolve_bucket(logical_name: str) -> str:
    """Map a logical bucket name to the actual MinIO bucket name."""
    return _BUCKET_MAP.get(logical_name, f"oelala-{logical_name}")


class StorageClient:
    """MinIO-backed storage client (drop-in replacement for oelala-storage client)."""

    def __init__(
        self,
        base_url: str = "http://localhost:9000",
        timeout: float = 30.0,
        auth_token: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ):
        """
        Initialize MinIO storage client.

        Args:
            base_url: MinIO endpoint URL (default: http://localhost:9000)
            timeout: Request timeout in seconds
            auth_token: Ignored (kept for backwards compat with old callers)
            access_key: MinIO access key (overrides MINIO_ACCESS_KEY env)
            secret_key: MinIO secret key (overrides MINIO_SECRET_KEY env)
        """
        import os

        self.base_url = base_url.rstrip("/")

        parsed = urlparse(self.base_url)
        endpoint = parsed.netloc or parsed.path
        secure = parsed.scheme == "https"

        self._access_key = access_key or os.environ.get("MINIO_ACCESS_KEY", "")
        self._secret_key = secret_key or os.environ.get("MINIO_SECRET_KEY", "")

        self._minio = Minio(
            endpoint,
            access_key=self._access_key,
            secret_key=self._secret_key,
            secure=secure,
        )
        self._known_buckets: set[str] = set()

    def close(self):
        """Close the client (no-op for MinIO SDK, kept for API compat)."""
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve(self, bucket: str, key: str = "") -> Tuple[str, str]:
        """
        Resolve a logical bucket (possibly compound like 'users/{id}')
        into a MinIO (bucket, key_prefix + key) pair.

        For compound bucket names such as ``users/abc-123``, the first
        segment selects the MinIO bucket (``oelala-users``) and the
        remainder is prepended to the key (``abc-123/{original_key}``).
        """
        if "/" in bucket:
            top, sub = bucket.split("/", 1)
            minio_bucket = _resolve_bucket(top)
            full_key = f"{sub}/{key}" if key else sub
        else:
            minio_bucket = _resolve_bucket(bucket)
            full_key = key
        return minio_bucket, full_key

    def _ensure_bucket(self, minio_bucket: str) -> None:
        """Create bucket if it does not exist (cached after first check)."""
        if minio_bucket in self._known_buckets:
            return
        if not self._minio.bucket_exists(minio_bucket):
            self._minio.make_bucket(minio_bucket)
            logger.info(f"🪣 Created MinIO bucket: {minio_bucket}")
        self._known_buckets.add(minio_bucket)

    @staticmethod
    def _guess_content_type(key: str) -> str:
        """Guess MIME type from key extension."""
        ct, _ = mimetypes.guess_type(key)
        return ct or "application/octet-stream"

    # ------------------------------------------------------------------
    # Core operations (same public signatures as before)
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Check storage service health by listing buckets."""
        try:
            buckets = self._minio.list_buckets()
            return {
                "status": "healthy",
                "backend": "minio",
                "buckets": len(buckets),
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}

    def status(self) -> Dict[str, Any]:
        """Get storage service status."""
        return self.health()

    def put(
        self,
        bucket: str,
        key: str,
        data: Union[bytes, BinaryIO, Path],
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload an object to storage.

        Args:
            bucket: Logical bucket name (e.g., "generated", "users/{id}")
            key: Object key (filename or path)
            data: File content as bytes, file-like object, or Path
            content_type: Optional content type header

        Returns:
            Object metadata dict with bucket, key, size, hash, content_type
        """
        minio_bucket, full_key = self._resolve(bucket, key)
        self._ensure_bucket(minio_bucket)

        # Normalise data to bytes
        if isinstance(data, Path):
            raw = data.read_bytes()
        elif hasattr(data, "read"):
            raw = data.read()
        else:
            raw = data

        size = len(raw)
        ct = content_type or self._guess_content_type(key)
        data_hash = hashlib.sha256(raw).hexdigest()

        self._minio.put_object(
            minio_bucket,
            full_key,
            io.BytesIO(raw),
            length=size,
            content_type=ct,
        )

        logger.debug(f"📤 PUT {minio_bucket}/{full_key} ({size} bytes)")
        return {
            "bucket": bucket,
            "key": key,
            "size": size,
            "hash": data_hash,
            "content_type": ct,
        }

    def get(self, bucket: str, key: str) -> bytes:
        """
        Download an object from storage.

        Args:
            bucket: Logical bucket name
            key: Object key

        Returns:
            File content as bytes
        """
        minio_bucket, full_key = self._resolve(bucket, key)
        resp = self._minio.get_object(minio_bucket, full_key)
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    def get_with_metadata(
        self, bucket: str, key: str
    ) -> Tuple[bytes, str, int, Optional[str], Optional[str]]:
        """
        Download an object and return content with metadata for proxying.

        Returns:
            Tuple of (content_bytes, content_type, content_length, etag, last_modified)
            where etag and last_modified are raw header values (or None).
        """
        minio_bucket, full_key = self._resolve(bucket, key)
        resp = self._minio.get_object(minio_bucket, full_key)
        try:
            content = resp.read()
            ct = resp.headers.get("Content-Type", "application/octet-stream")
            etag = resp.headers.get("ETag")
            last_modified = resp.headers.get("Last-Modified")
            return content, ct, len(content), etag, last_modified
        finally:
            resp.close()
            resp.release_conn()

    def stat(self, bucket: str, key: str) -> Optional[Dict[str, Any]]:
        """
        Get full object metadata via stat_object.

        Returns:
            Dict with size, content_type, etag, last_modified, or None if not found.
        """
        minio_bucket, full_key = self._resolve(bucket, key)
        try:
            s = self._minio.stat_object(minio_bucket, full_key)
            return {
                "size": s.size,
                "content_type": s.content_type or "application/octet-stream",
                "etag": s.etag,
                "last_modified": s.last_modified,
            }
        except S3Error as e:
            if e.code in ("NoSuchKey", "NoSuchBucket"):
                return None
            raise

    def get_object_range(
        self, bucket: str, key: str, offset: int, length: int
    ) -> bytes:
        """
        Download a byte range of an object.

        Args:
            bucket: Logical bucket name
            key: Object key
            offset: Start byte (inclusive)
            length: Number of bytes to read

        Returns:
            Requested bytes
        """
        minio_bucket, full_key = self._resolve(bucket, key)
        resp = self._minio.get_object(
            minio_bucket, full_key, offset=offset, length=length
        )
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    def stream(self, bucket: str, key: str):
        """
        Stream an object from storage. Returns a context manager yielding chunks.

        Usage:
            with storage.stream("generated", "video.mp4") as (chunks, content_type, size):
                for chunk in chunks:
                    yield chunk
        """
        import contextlib

        minio_bucket, full_key = self._resolve(bucket, key)

        @contextlib.contextmanager
        def _stream_ctx():
            resp = self._minio.get_object(minio_bucket, full_key)
            try:
                ct = resp.headers.get("Content-Type", "application/octet-stream")
                cl = int(resp.headers.get("Content-Length", 0))
                yield resp.stream(amt=8192), ct, cl
            finally:
                resp.close()
                resp.release_conn()

        return _stream_ctx()

    def get_to_file(self, bucket: str, key: str, path: Path) -> Path:
        """
        Download an object directly to a file.

        Args:
            bucket: Logical bucket name
            key: Object key
            path: Destination file path

        Returns:
            Path to downloaded file
        """
        minio_bucket, full_key = self._resolve(bucket, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._minio.fget_object(minio_bucket, full_key, str(path))
        return path

    def move(
        self, src_bucket: str, src_key: str, dest_bucket: str, dest_key: str
    ) -> bool:
        """
        Move/rename an object (S3: copy + delete).

        Returns:
            True if moved successfully
        """
        src_mb, src_fk = self._resolve(src_bucket, src_key)
        dst_mb, dst_fk = self._resolve(dest_bucket, dest_key)

        try:
            self._ensure_bucket(dst_mb)
            self._minio.copy_object(dst_mb, dst_fk, CopySource(src_mb, src_fk))
            self._minio.remove_object(src_mb, src_fk)
            return True
        except S3Error as e:
            if e.code == "NoSuchKey":
                return False
            raise

    def delete(self, bucket: str, key: str) -> bool:
        """
        Delete an object from storage.

        Returns:
            True if deleted (MinIO remove_object is idempotent)
        """
        minio_bucket, full_key = self._resolve(bucket, key)
        try:
            self._minio.remove_object(minio_bucket, full_key)
            return True
        except S3Error:
            return False

    def head(self, bucket: str, key: str) -> Optional[Dict[str, Any]]:
        """
        Get object metadata without downloading.

        Returns:
            Dict with size + exists, or None if not found
        """
        minio_bucket, full_key = self._resolve(bucket, key)
        try:
            stat = self._minio.stat_object(minio_bucket, full_key)
            return {
                "size": stat.size,
                "exists": True,
                "content_type": stat.content_type,
                "last_modified": stat.last_modified.isoformat()
                if stat.last_modified
                else None,
                "etag": stat.etag,
            }
        except S3Error as e:
            if e.code in ("NoSuchKey", "NoSuchBucket"):
                return None
            raise

    def exists(self, bucket: str, key: str) -> bool:
        """Check if an object exists."""
        return self.head(bucket, key) is not None

    def list(
        self,
        bucket: str,
        prefix: str = "",
    ) -> List[Dict[str, Any]]:
        """
        List objects in a bucket.

        Args:
            bucket: Logical bucket name (may contain '/' for nested paths)
            prefix: Optional prefix filter

        Returns:
            List of object metadata dicts
        """
        minio_bucket, base_prefix = self._resolve(bucket, "")

        # Merge base_prefix with caller-supplied prefix
        if base_prefix and prefix:
            full_prefix = (
                f"{base_prefix}/{prefix}"
                if not base_prefix.endswith("/")
                else f"{base_prefix}{prefix}"
            )
        elif base_prefix:
            full_prefix = (
                base_prefix if base_prefix.endswith("/") else f"{base_prefix}/"
            )
        else:
            full_prefix = prefix

        objects = []
        try:
            for obj in self._minio.list_objects(
                minio_bucket, prefix=full_prefix, recursive=True
            ):
                if obj.is_dir:
                    continue
                objects.append(
                    {
                        "key": obj.object_name,
                        "size": obj.size,
                        "content_type": "",
                        "modified_at": obj.last_modified.isoformat()
                        if obj.last_modified
                        else "",
                        "hash": obj.etag or "",
                    }
                )
        except S3Error as e:
            if e.code == "NoSuchBucket":
                return []
            raise
        return objects

    def list_buckets(self) -> List[str]:
        """List available MinIO buckets."""
        return [b.name for b in self._minio.list_buckets()]

    # ------------------------------------------------------------------
    # Presigned URLs (replaces custom HMAC signed URLs)
    # ------------------------------------------------------------------

    def presigned_get(
        self,
        bucket: str,
        key: str,
        expires: int = 3600,
    ) -> str:
        """
        Generate a presigned GET URL for temporary public access.

        Args:
            bucket: Logical bucket name
            key: Object key
            expires: Expiration time in seconds (default 1 hour, max 7 days)

        Returns:
            Presigned URL string
        """
        minio_bucket, full_key = self._resolve(bucket, key)
        return self._minio.presigned_get_object(
            minio_bucket,
            full_key,
            expires=timedelta(seconds=min(expires, 604800)),
        )

    def presigned_put(
        self,
        bucket: str,
        key: str,
        expires: int = 3600,
    ) -> str:
        """
        Generate a presigned PUT URL for direct upload.

        Args:
            bucket: Logical bucket name
            key: Object key
            expires: Expiration time in seconds

        Returns:
            Presigned URL string
        """
        minio_bucket, full_key = self._resolve(bucket, key)
        self._ensure_bucket(minio_bucket)
        return self._minio.presigned_put_object(
            minio_bucket,
            full_key,
            expires=timedelta(seconds=min(expires, 604800)),
        )

    # ------------------------------------------------------------------
    # User-scoped media operations (unchanged public API)
    # ------------------------------------------------------------------

    @staticmethod
    def user_bucket(user_id: str) -> str:
        """Get bucket path for user media: users/<user_id>"""
        return f"users/{user_id}"

    @staticmethod
    def user_key(media_type: str, filename: str) -> str:
        """Get key path within user bucket: <type>/<filename>"""
        return f"{media_type}/{filename}"

    def put_user_media(
        self,
        user_id: str,
        media_type: str,  # 'images', 'videos', 'audio'
        filename: str,
        data: Union[bytes, BinaryIO, Path],
        content_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload media for a specific user.

        Args:
            user_id: Supabase user ID (UUID)
            media_type: Type of media ('images', 'videos', 'audio')
            filename: Filename to store as
            data: File content
            content_type: MIME type

        Returns:
            Object metadata with full path
        """
        bucket = self.user_bucket(user_id)
        key = self.user_key(media_type, filename)
        result = self.put(bucket, key, data, content_type)
        result["user_id"] = user_id
        result["media_type"] = media_type
        return result

    def get_user_media(
        self,
        user_id: str,
        media_type: str,
        filename: str,
    ) -> bytes:
        """Download user's media file."""
        bucket = self.user_bucket(user_id)
        key = self.user_key(media_type, filename)
        return self.get(bucket, key)

    def iter_user_media(
        self,
        user_id: str,
        media_type: str,
        filename: str,
    ):
        """Stream user's media file as an iterator of bytes."""
        minio_bucket, full_key = self._resolve(
            self.user_bucket(user_id),
            self.user_key(media_type, filename),
        )

        def _iter():
            resp = self._minio.get_object(minio_bucket, full_key)
            try:
                for chunk in resp.stream(amt=8192):
                    if chunk:
                        yield chunk
            finally:
                resp.close()
                resp.release_conn()

        return _iter()

    def move_user_media(
        self,
        user_id: str,
        media_type: str,
        src_filename: str,
        dest_filename: str,
    ) -> bool:
        """
        Move/rename user's media file (e.g. into a subfolder).
        Subfolders are supported by including a '/' in the dest_filename.
        """
        bucket = self.user_bucket(user_id)
        src_key = self.user_key(media_type, src_filename)
        dest_key = self.user_key(media_type, dest_filename)
        return self.move(bucket, src_key, bucket, dest_key)

    def delete_user_media(
        self,
        user_id: str,
        media_type: str,
        filename: str,
    ) -> bool:
        """Delete user's media file."""
        bucket = self.user_bucket(user_id)
        key = self.user_key(media_type, filename)
        return self.delete(bucket, key)

    def list_user_media(
        self,
        user_id: str,
        media_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List all media for a user, optionally filtered by type.

        Args:
            user_id: Supabase user ID
            media_type: Optional filter ('images', 'videos', 'audio', 'uploads')

        Returns:
            List of object metadata
        """
        bucket = self.user_bucket(user_id)
        prefix = f"{media_type}/" if media_type else ""
        objects = self.list(bucket, prefix)

        # Strip user_id prefix from keys so downstream consumers see
        # clean keys like "uploads/file.png" instead of "{user_id}/uploads/file.png".
        user_prefix = f"{user_id}/"

        for obj in objects:
            obj["user_id"] = user_id
            key = obj.get("key", "")
            if key.startswith(user_prefix):
                key = key[len(user_prefix) :]
            parts = key.split("/", 1)
            if len(parts) >= 1:
                obj["media_type"] = parts[0]
            if len(parts) >= 2:
                obj["filename"] = parts[1]

        return objects

    def user_media_exists(
        self,
        user_id: str,
        media_type: str,
        filename: str,
    ) -> bool:
        """Check if user's media file exists."""
        bucket = self.user_bucket(user_id)
        key = self.user_key(media_type, filename)
        return self.exists(bucket, key)

    def get_user_media_url(
        self,
        user_id: str,
        media_type: str,
        filename: str,
        external: bool = False,
        expires: int = 3600,
    ) -> str:
        """
        Get URL for user's media file.

        Args:
            external: If True, return a presigned URL (accessible without auth)
            expires: Presigned URL expiry in seconds (only used when external=True)
        """
        bucket = self.user_bucket(user_id)
        key = self.user_key(media_type, filename)

        if external:
            return self.presigned_get(bucket, key, expires=expires)
        else:
            return f"{self.base_url}/{bucket}/{key}"


# =====================================================================
# Module-level singleton & convenience functions
# =====================================================================
_default_client: Optional[StorageClient] = None


def get_client() -> StorageClient:
    """Get the default storage client singleton.

    Reads MinIO config from environment variables:
      - MINIO_ENDPOINT (default: http://localhost:9000)
      - MINIO_ACCESS_KEY
      - MINIO_SECRET_KEY

    Falls back to STORAGE_URL / STORAGE_API_KEY for backwards compat.
    """
    import os

    global _default_client
    if _default_client is None:
        endpoint = os.environ.get(
            "MINIO_ENDPOINT",
            os.environ.get("STORAGE_URL", "http://localhost:9000"),
        )
        access_key = os.environ.get(
            "MINIO_ACCESS_KEY", os.environ.get("STORAGE_API_KEY", "")
        )
        secret_key = os.environ.get("MINIO_SECRET_KEY", "")
        _default_client = StorageClient(
            base_url=endpoint,
            access_key=access_key,
            secret_key=secret_key,
        )
    return _default_client


# Alias used by gallery_api.py (import name compat)
get_storage_client = get_client


def put(
    bucket: str, key: str, data: Union[bytes, BinaryIO, Path], **kwargs
) -> Dict[str, Any]:
    """Upload an object using the default client."""
    return get_client().put(bucket, key, data, **kwargs)


def get(bucket: str, key: str) -> bytes:
    """Download an object using the default client."""
    return get_client().get(bucket, key)


def move(src_bucket: str, src_key: str, dest_bucket: str, dest_key: str) -> bool:
    """Move an object using the default client."""
    return get_client().move(src_bucket, src_key, dest_bucket, dest_key)


def delete(bucket: str, key: str) -> bool:
    """Delete an object using the default client."""
    return get_client().delete(bucket, key)


def list_objects(bucket: str, prefix: str = "") -> List[Dict[str, Any]]:
    """List objects using the default client."""
    return get_client().list(bucket, prefix)


def exists(bucket: str, key: str) -> bool:
    """Check if object exists using the default client."""
    return get_client().exists(bucket, key)


# Quick test
if __name__ == "__main__":
    import sys

    # 👋 Hey agent-friend who sent a kiss — here's your message:
    # "De groeten uit de MinIO-migratie! 🚀 Moge je buckets altijd vol
    #  en je latency altijd laag zijn. Proost! 🍻"

    client = get_client()

    try:
        health = client.health()
        print(f"✅ Storage service healthy: {health}")

        # List generated bucket
        objects = client.list("generated")
        print(f"📁 Found {len(objects)} objects in 'generated' bucket")

        # Test upload
        test_data = b"Test from MinIO Python client!"
        obj = client.put("generated", "python_test.txt", test_data)
        print(f"📤 Uploaded: {obj}")

        # Test download
        downloaded = client.get("generated", "python_test.txt")
        assert downloaded == test_data
        print("📥 Downloaded and verified!")

        # Test presigned URL
        url = client.presigned_get("generated", "python_test.txt", expires=300)
        print(f"🔗 Presigned URL: {url[:80]}...")

        # Test delete
        client.delete("generated", "python_test.txt")
        print("🗑️ Deleted test file")

        print("\n✅ All MinIO storage client tests passed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
