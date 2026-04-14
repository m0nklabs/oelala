#!/usr/bin/env python3
"""
Unit tests for MinIO-backed storage_client.

Tests the StorageClient with mocked MinIO SDK calls so they run
without a live MinIO instance.
"""

import io
import hashlib
import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

from storage_client import (
    StorageClient,
    _resolve_bucket,
    _BUCKET_MAP,
    get_client,
    get_storage_client,
)


class TestBucketMapping:
    """Test logical → MinIO bucket name mapping."""

    def test_known_buckets(self):
        assert _resolve_bucket("generated") == "oelala-generated"
        assert _resolve_bucket("comfyui-local") == "oelala-comfyui"
        assert _resolve_bucket("avatars") == "oelala-avatars"
        assert _resolve_bucket("users") == "oelala-users"

    def test_unknown_bucket_gets_prefix(self):
        assert _resolve_bucket("uploads") == "oelala-uploads"
        assert _resolve_bucket("temp") == "oelala-temp"
        assert _resolve_bucket("archive") == "oelala-archive"


class TestResolve:
    """Test _resolve helper for bucket + key decomposition."""

    def setup_method(self):
        self.client = StorageClient(
            base_url="http://localhost:9000",
            access_key="test",
            secret_key="test",
        )

    def test_simple_bucket(self):
        bucket, key = self.client._resolve("generated", "video.mp4")
        assert bucket == "oelala-generated"
        assert key == "video.mp4"

    def test_compound_bucket(self):
        bucket, key = self.client._resolve("users/abc-123", "videos/test.mp4")
        assert bucket == "oelala-users"
        assert key == "abc-123/videos/test.mp4"

    def test_compound_bucket_empty_key(self):
        bucket, key = self.client._resolve("users/abc-123", "")
        assert bucket == "oelala-users"
        assert key == "abc-123"

    def test_nested_key(self):
        bucket, key = self.client._resolve("generated", "cloud-wan22/output.mp4")
        assert bucket == "oelala-generated"
        assert key == "cloud-wan22/output.mp4"


class TestUserHelpers:
    """Test user-scoped helper methods."""

    def test_user_bucket(self):
        assert StorageClient.user_bucket("user-123") == "users/user-123"

    def test_user_key(self):
        assert StorageClient.user_key("videos", "test.mp4") == "videos/test.mp4"
        assert StorageClient.user_key("images", "photo.jpg") == "images/photo.jpg"


class TestPut:
    """Test upload operation."""

    def setup_method(self):
        self.client = StorageClient(
            base_url="http://localhost:9000",
            access_key="test",
            secret_key="test",
        )

    @patch.object(StorageClient, "_ensure_bucket")
    def test_put_bytes(self, mock_ensure):
        """Test uploading bytes data."""
        mock_minio = MagicMock()
        self.client._minio = mock_minio

        data = b"Hello MinIO!"
        result = self.client.put("generated", "test.txt", data, content_type="text/plain")

        assert result["bucket"] == "generated"
        assert result["key"] == "test.txt"
        assert result["size"] == len(data)
        assert result["hash"] == hashlib.sha256(data).hexdigest()
        assert result["content_type"] == "text/plain"

        mock_minio.put_object.assert_called_once()
        call_args = mock_minio.put_object.call_args
        assert call_args[0][0] == "oelala-generated"
        assert call_args[0][1] == "test.txt"

    @patch.object(StorageClient, "_ensure_bucket")
    def test_put_path(self, mock_ensure, tmp_path):
        """Test uploading from a Path."""
        mock_minio = MagicMock()
        self.client._minio = mock_minio

        test_file = tmp_path / "test.txt"
        test_file.write_bytes(b"file content")

        result = self.client.put("generated", "test.txt", test_file)

        assert result["size"] == 12
        mock_minio.put_object.assert_called_once()

    @patch.object(StorageClient, "_ensure_bucket")
    def test_put_user_media(self, mock_ensure):
        """Test user media upload resolves to oelala-users bucket."""
        mock_minio = MagicMock()
        self.client._minio = mock_minio

        data = b"video data"
        result = self.client.put_user_media(
            "user-abc", "videos", "my_video.mp4", data, "video/mp4"
        )

        assert result["user_id"] == "user-abc"
        assert result["media_type"] == "videos"

        call_args = mock_minio.put_object.call_args
        assert call_args[0][0] == "oelala-users"
        assert call_args[0][1] == "user-abc/videos/my_video.mp4"


class TestGet:
    """Test download operations."""

    def setup_method(self):
        self.client = StorageClient(
            base_url="http://localhost:9000",
            access_key="test",
            secret_key="test",
        )

    def test_get(self):
        mock_minio = MagicMock()
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"file content"
        mock_minio.get_object.return_value = mock_resp
        self.client._minio = mock_minio

        result = self.client.get("generated", "test.txt")

        assert result == b"file content"
        mock_minio.get_object.assert_called_once_with("oelala-generated", "test.txt")
        mock_resp.close.assert_called_once()
        mock_resp.release_conn.assert_called_once()

    def test_get_with_metadata(self):
        mock_minio = MagicMock()
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"file content"
        mock_resp.headers = {"Content-Type": "text/plain"}
        mock_minio.get_object.return_value = mock_resp
        self.client._minio = mock_minio

        content, ct, cl = self.client.get_with_metadata("generated", "test.txt")

        assert content == b"file content"
        assert ct == "text/plain"
        assert cl == 12

    def test_get_user_media(self):
        mock_minio = MagicMock()
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"video bytes"
        mock_minio.get_object.return_value = mock_resp
        self.client._minio = mock_minio

        result = self.client.get_user_media("user-abc", "videos", "clip.mp4")

        assert result == b"video bytes"
        mock_minio.get_object.assert_called_once_with(
            "oelala-users", "user-abc/videos/clip.mp4"
        )


class TestDelete:
    """Test delete operations."""

    def setup_method(self):
        self.client = StorageClient(
            base_url="http://localhost:9000",
            access_key="test",
            secret_key="test",
        )

    def test_delete(self):
        mock_minio = MagicMock()
        self.client._minio = mock_minio

        result = self.client.delete("generated", "test.txt")

        assert result is True
        mock_minio.remove_object.assert_called_once_with("oelala-generated", "test.txt")

    def test_delete_user_media(self):
        mock_minio = MagicMock()
        self.client._minio = mock_minio

        result = self.client.delete_user_media("user-abc", "videos", "clip.mp4")

        assert result is True
        mock_minio.remove_object.assert_called_once_with(
            "oelala-users", "user-abc/videos/clip.mp4"
        )


class TestMove:
    """Test move (copy + delete) operations."""

    def setup_method(self):
        self.client = StorageClient(
            base_url="http://localhost:9000",
            access_key="test",
            secret_key="test",
        )

    @patch.object(StorageClient, "_ensure_bucket")
    def test_move_same_bucket(self, mock_ensure):
        mock_minio = MagicMock()
        self.client._minio = mock_minio

        result = self.client.move("generated", "old.mp4", "generated", "new.mp4")

        assert result is True
        mock_minio.copy_object.assert_called_once()
        mock_minio.remove_object.assert_called_once_with("oelala-generated", "old.mp4")


class TestHeadAndExists:
    """Test metadata and existence checks."""

    def setup_method(self):
        self.client = StorageClient(
            base_url="http://localhost:9000",
            access_key="test",
            secret_key="test",
        )

    def test_head_existing(self):
        from minio.error import S3Error

        mock_minio = MagicMock()
        mock_stat = MagicMock()
        mock_stat.size = 1024
        mock_stat.content_type = "video/mp4"
        mock_stat.last_modified = None
        mock_stat.etag = "abc123"
        mock_minio.stat_object.return_value = mock_stat
        self.client._minio = mock_minio

        result = self.client.head("generated", "test.mp4")

        assert result is not None
        assert result["size"] == 1024
        assert result["exists"] is True
        assert result["content_type"] == "video/mp4"

    def test_head_not_found(self):
        from minio.error import S3Error

        mock_minio = MagicMock()
        mock_minio.stat_object.side_effect = S3Error(
            "NoSuchKey", "NoSuchKey", "", "", "", ""
        )
        self.client._minio = mock_minio

        result = self.client.head("generated", "missing.mp4")
        assert result is None

    def test_exists_true(self):
        mock_minio = MagicMock()
        mock_stat = MagicMock()
        mock_stat.size = 100
        mock_stat.content_type = "text/plain"
        mock_stat.last_modified = None
        mock_stat.etag = "x"
        mock_minio.stat_object.return_value = mock_stat
        self.client._minio = mock_minio

        assert self.client.exists("generated", "test.txt") is True

    def test_exists_false(self):
        from minio.error import S3Error

        mock_minio = MagicMock()
        mock_minio.stat_object.side_effect = S3Error(
            "NoSuchKey", "NoSuchKey", "", "", "", ""
        )
        self.client._minio = mock_minio

        assert self.client.exists("generated", "missing.txt") is False


class TestList:
    """Test list operations."""

    def setup_method(self):
        self.client = StorageClient(
            base_url="http://localhost:9000",
            access_key="test",
            secret_key="test",
        )

    def test_list_simple(self):
        mock_minio = MagicMock()
        obj1 = MagicMock()
        obj1.is_dir = False
        obj1.object_name = "video1.mp4"
        obj1.size = 1000
        obj1.last_modified = None
        obj1.etag = "abc"

        obj2 = MagicMock()
        obj2.is_dir = False
        obj2.object_name = "video2.mp4"
        obj2.size = 2000
        obj2.last_modified = None
        obj2.etag = "def"

        mock_minio.list_objects.return_value = [obj1, obj2]
        self.client._minio = mock_minio

        result = self.client.list("generated")

        assert len(result) == 2
        assert result[0]["key"] == "video1.mp4"
        assert result[1]["size"] == 2000

    def test_list_skips_dirs(self):
        mock_minio = MagicMock()
        dir_obj = MagicMock()
        dir_obj.is_dir = True

        file_obj = MagicMock()
        file_obj.is_dir = False
        file_obj.object_name = "file.txt"
        file_obj.size = 100
        file_obj.last_modified = None
        file_obj.etag = "x"

        mock_minio.list_objects.return_value = [dir_obj, file_obj]
        self.client._minio = mock_minio

        result = self.client.list("generated")
        assert len(result) == 1

    def test_list_compound_bucket(self):
        """Test listing with compound bucket (users/{id})."""
        mock_minio = MagicMock()
        mock_minio.list_objects.return_value = []
        self.client._minio = mock_minio

        self.client.list("users/abc-123", prefix="videos/")

        call_args = mock_minio.list_objects.call_args
        assert call_args[0][0] == "oelala-users"
        assert "abc-123" in call_args[1]["prefix"]


class TestPresignedUrls:
    """Test presigned URL generation."""

    def setup_method(self):
        self.client = StorageClient(
            base_url="http://localhost:9000",
            access_key="test",
            secret_key="test",
        )

    def test_presigned_get(self):
        mock_minio = MagicMock()
        mock_minio.presigned_get_object.return_value = "http://localhost:9000/oelala-generated/test.mp4?X-Amz-Signature=abc"
        self.client._minio = mock_minio

        url = self.client.presigned_get("generated", "test.mp4", expires=3600)

        assert "oelala-generated" in url
        mock_minio.presigned_get_object.assert_called_once()

    def test_presigned_put(self):
        mock_minio = MagicMock()
        mock_minio.presigned_put_object.return_value = "http://localhost:9000/oelala-generated/upload.mp4?X-Amz-Signature=xyz"
        mock_minio.bucket_exists.return_value = True
        self.client._minio = mock_minio

        url = self.client.presigned_put("generated", "upload.mp4", expires=3600)

        assert "oelala-generated" in url


class TestHealth:
    """Test health check."""

    def setup_method(self):
        self.client = StorageClient(
            base_url="http://localhost:9000",
            access_key="test",
            secret_key="test",
        )

    def test_health_success(self):
        mock_minio = MagicMock()
        mock_minio.list_buckets.return_value = [MagicMock(), MagicMock()]
        self.client._minio = mock_minio

        health = self.client.health()

        assert health["status"] == "healthy"
        assert health["backend"] == "minio"
        assert health["buckets"] == 2

    def test_health_failure(self):
        mock_minio = MagicMock()
        mock_minio.list_buckets.side_effect = Exception("Connection refused")
        self.client._minio = mock_minio

        health = self.client.health()

        assert health["status"] == "unhealthy"
        assert "Connection refused" in health["error"]


class TestGetClientSingleton:
    """Test module-level singleton factory."""

    @patch.dict("os.environ", {
        "MINIO_ENDPOINT": "http://test-minio:9000",
        "MINIO_ACCESS_KEY": "testkey",
        "MINIO_SECRET_KEY": "testsecret",
    })
    def test_get_client_reads_env(self):
        import storage_client as sc
        sc._default_client = None  # Reset singleton

        client = sc.get_client()
        assert client is not None
        assert client.base_url == "http://test-minio:9000"

        sc._default_client = None  # Clean up

    def test_get_storage_client_is_alias(self):
        """Verify get_storage_client is same as get_client (gallery_api compat)."""
        assert get_storage_client is get_client


class TestBackwardsCompat:
    """Test backwards compatibility with old callers."""

    def test_close_is_noop(self):
        """Close should not raise (kept for API compat)."""
        client = StorageClient(
            base_url="http://localhost:9000",
            access_key="test",
            secret_key="test",
        )
        client.close()  # Should not raise

    def test_context_manager(self):
        """Context manager protocol should work."""
        with StorageClient(
            base_url="http://localhost:9000",
            access_key="test",
            secret_key="test",
        ) as client:
            assert client is not None

    def test_auth_token_ignored(self):
        """Old auth_token parameter should be accepted but ignored."""
        client = StorageClient(
            base_url="http://localhost:9000",
            auth_token="old-bearer-token",
            access_key="real-key",
            secret_key="real-secret",
        )
        assert client._access_key == "real-key"


class TestStat:
    """Test stat() for full object metadata."""

    def setup_method(self):
        self.client = StorageClient(
            base_url="http://localhost:9000",
            access_key="test",
            secret_key="test",
        )

    def test_stat_existing(self):
        from datetime import datetime, timezone

        mock_minio = MagicMock()
        mock_stat = MagicMock()
        mock_stat.size = 1024
        mock_stat.content_type = "video/mp4"
        mock_stat.etag = "abc123"
        mock_stat.last_modified = datetime(2026, 4, 14, 12, 0, 0, tzinfo=timezone.utc)
        mock_minio.stat_object.return_value = mock_stat
        self.client._minio = mock_minio

        result = self.client.stat("generated", "video.mp4")

        assert result is not None
        assert result["size"] == 1024
        assert result["content_type"] == "video/mp4"
        assert result["etag"] == "abc123"
        assert result["last_modified"] == mock_stat.last_modified
        mock_minio.stat_object.assert_called_once_with("oelala-generated", "video.mp4")

    def test_stat_not_found(self):
        from minio.error import S3Error

        mock_minio = MagicMock()
        mock_minio.stat_object.side_effect = S3Error(
            "NoSuchKey", "NoSuchKey", "", "", "", ""
        )
        self.client._minio = mock_minio

        result = self.client.stat("generated", "missing.mp4")
        assert result is None

    def test_stat_no_content_type(self):
        mock_minio = MagicMock()
        mock_stat = MagicMock()
        mock_stat.size = 512
        mock_stat.content_type = None
        mock_stat.etag = "def456"
        mock_stat.last_modified = None
        mock_minio.stat_object.return_value = mock_stat
        self.client._minio = mock_minio

        result = self.client.stat("generated", "file.bin")
        assert result["content_type"] == "application/octet-stream"
        assert result["last_modified"] is None


class TestGetObjectRange:
    """Test get_object_range() for partial content retrieval."""

    def setup_method(self):
        self.client = StorageClient(
            base_url="http://localhost:9000",
            access_key="test",
            secret_key="test",
        )

    def test_range_read(self):
        mock_minio = MagicMock()
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"partial content"
        mock_minio.get_object.return_value = mock_resp
        self.client._minio = mock_minio

        result = self.client.get_object_range("generated", "video.mp4", offset=100, length=15)

        assert result == b"partial content"
        mock_minio.get_object.assert_called_once_with(
            "oelala-generated", "video.mp4", offset=100, length=15
        )
        mock_resp.close.assert_called_once()
        mock_resp.release_conn.assert_called_once()

    def test_range_read_compound_bucket(self):
        """Range reads should work with compound bucket resolution."""
        mock_minio = MagicMock()
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"user data"
        mock_minio.get_object.return_value = mock_resp
        self.client._minio = mock_minio

        result = self.client.get_object_range(
            "users/user123", "videos/clip.mp4", offset=0, length=9
        )

        assert result == b"user data"
        mock_minio.get_object.assert_called_once_with(
            "oelala-users", "user123/videos/clip.mp4", offset=0, length=9
        )
