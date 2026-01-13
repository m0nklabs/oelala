"""
oelala-storage client for the backend.

This module provides a Python client for interacting with the oelala-storage service.
The storage service runs on port 7990 and provides S3-compatible operations.

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
"""

import httpx
from pathlib import Path
from typing import Optional, List, Dict, Any, BinaryIO, Union
import logging

logger = logging.getLogger(__name__)


class StorageClient:
    """Client for oelala-storage service."""

    def __init__(
        self,
        base_url: str = "http://localhost:7990",
        timeout: float = 30.0,
        auth_token: Optional[str] = None,
    ):
        """
        Initialize storage client.

        Args:
            base_url: Storage service URL (default: http://localhost:7990)
            timeout: Request timeout in seconds
            auth_token: Optional Bearer token for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.auth_token = auth_token
        self._client: Optional[httpx.Client] = None

    @property
    def client(self) -> httpx.Client:
        """Lazy-init httpx client."""
        if self._client is None:
            headers = {}
            if self.auth_token:
                headers["Authorization"] = f"Bearer {self.auth_token}"
            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout,
                headers=headers,
            )
        return self._client

    def close(self):
        """Close the client."""
        if self._client:
            self._client.close()
            self._client = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def health(self) -> Dict[str, Any]:
        """Check storage service health."""
        resp = self.client.get("/health")
        resp.raise_for_status()
        return resp.json()

    def status(self) -> Dict[str, Any]:
        """Get storage service status."""
        resp = self.client.get("/status")
        resp.raise_for_status()
        return resp.json()

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
            bucket: Bucket name (e.g., "generated", "uploads")
            key: Object key (filename or path)
            data: File content as bytes, file-like object, or Path
            content_type: Optional content type header

        Returns:
            Object metadata dict with bucket, key, size, hash, content_type
        """
        url = f"/{bucket}/{key}"
        headers = {}
        if content_type:
            headers["Content-Type"] = content_type

        # Handle different data types
        if isinstance(data, Path):
            data = data.read_bytes()
        elif hasattr(data, "read"):
            data = data.read()

        resp = self.client.put(url, content=data, headers=headers)
        resp.raise_for_status()
        return resp.json()

    def get(self, bucket: str, key: str) -> bytes:
        """
        Download an object from storage.

        Args:
            bucket: Bucket name
            key: Object key

        Returns:
            File content as bytes
        """
        url = f"/{bucket}/{key}"
        resp = self.client.get(url)
        resp.raise_for_status()
        return resp.content

    def get_to_file(self, bucket: str, key: str, path: Path) -> Path:
        """
        Download an object directly to a file.

        Args:
            bucket: Bucket name
            key: Object key
            path: Destination file path

        Returns:
            Path to downloaded file
        """
        data = self.get(bucket, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def delete(self, bucket: str, key: str) -> bool:
        """
        Delete an object from storage.

        Args:
            bucket: Bucket name
            key: Object key

        Returns:
            True if deleted successfully
        """
        url = f"/{bucket}/{key}"
        resp = self.client.delete(url)
        return resp.status_code == 204

    def head(self, bucket: str, key: str) -> Optional[Dict[str, Any]]:
        """
        Get object metadata without downloading.

        Args:
            bucket: Bucket name
            key: Object key

        Returns:
            Dict with Content-Length header, or None if not found
        """
        url = f"/{bucket}/{key}"
        resp = self.client.head(url)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return {
            "size": int(resp.headers.get("Content-Length", 0)),
            "exists": True,
        }

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
            bucket: Bucket name
            prefix: Optional prefix filter

        Returns:
            List of object metadata dicts
        """
        url = f"/{bucket}"
        params = {}
        if prefix:
            params["prefix"] = prefix

        resp = self.client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data.get("objects", [])

    def list_buckets(self) -> List[str]:
        """
        List available buckets (directories in storage root).

        Note: This assumes bucket = top-level directory structure.
        """
        # The storage service doesn't have a native list-buckets endpoint yet
        # For now, we know the buckets are: generated, uploads, archive, temp
        return ["generated", "uploads", "archive", "temp"]

    # =========================================================================
    # User-scoped media operations
    # =========================================================================

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
            media_type: Optional filter ('images', 'videos', 'audio')

        Returns:
            List of object metadata
        """
        bucket = self.user_bucket(user_id)
        prefix = f"{media_type}/" if media_type else ""
        objects = self.list(bucket, prefix)

        # Enrich with user info and parsed media type
        for obj in objects:
            obj["user_id"] = user_id
            key = obj.get("key", "")
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
    ) -> str:
        """
        Get URL for user's media file.

        Args:
            external: If True, return production URL (storage.oelala.xyz)
        """
        bucket = self.user_bucket(user_id)
        key = self.user_key(media_type, filename)

        if external:
            return f"https://storage.oelala.xyz/{bucket}/{key}"
        else:
            return f"{self.base_url}/{bucket}/{key}"


# Module-level singleton for convenience
_default_client: Optional[StorageClient] = None


def get_client() -> StorageClient:
    """Get the default storage client singleton.

    Auth token is loaded from STORAGE_API_KEY environment variable.
    """
    import os

    global _default_client
    if _default_client is None:
        auth_token = os.environ.get("STORAGE_API_KEY")
        base_url = os.environ.get("STORAGE_URL", "http://localhost:7990")
        _default_client = StorageClient(base_url=base_url, auth_token=auth_token)
    return _default_client


def put(
    bucket: str, key: str, data: Union[bytes, BinaryIO, Path], **kwargs
) -> Dict[str, Any]:
    """Upload an object using the default client."""
    return get_client().put(bucket, key, data, **kwargs)


def get(bucket: str, key: str) -> bytes:
    """Download an object using the default client."""
    return get_client().get(bucket, key)


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

    client = StorageClient()

    try:
        health = client.health()
        print(f"✅ Storage service healthy: {health}")

        # List generated bucket
        objects = client.list("generated")
        print(f"📁 Found {len(objects)} objects in 'generated' bucket")

        # Test upload
        test_data = b"Test from Python client!"
        obj = client.put("test", "python_test.txt", test_data)
        print(f"📤 Uploaded: {obj}")

        # Test download
        downloaded = client.get("test", "python_test.txt")
        assert downloaded == test_data
        print("📥 Downloaded and verified!")

        # Test delete
        client.delete("test", "python_test.txt")
        print("🗑️ Deleted test file")

        print("\n✅ All storage client tests passed!")

    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        client.close()
