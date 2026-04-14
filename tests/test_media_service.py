#!/usr/bin/env python3
"""
Unit tests for media_service.py MinIO integration.

Tests the MediaService with mocked dependencies (MinIO storage client + Supabase).
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

from media_service import MediaService, MediaRecord, TIER_RETENTION_DAYS


class TestMediaRecord:
    """Test MediaRecord dataclass."""

    def test_bucket_from_storage_path(self):
        record = MediaRecord(
            id="1",
            user_id="user-abc",
            storage_path="users/user-abc/videos/test.mp4",
            media_type="video",
        )
        assert record.bucket == "users/user-abc"
        assert record.key == "videos/test.mp4"
        assert record.filename == "test.mp4"

    def test_storage_url(self):
        record = MediaRecord(
            id="1",
            user_id="user-abc",
            storage_path="users/user-abc/videos/test.mp4",
            media_type="video",
        )
        assert record.storage_url == "/users/user-abc/videos/test.mp4"


class TestMediaServiceInit:
    """Test MediaService initialisation."""

    @patch.dict("os.environ", {
        "MINIO_ENDPOINT": "http://test-minio:9000",
        "SUPABASE_URL": "https://test.supabase.co",
    })
    def test_reads_minio_endpoint(self):
        service = MediaService()
        assert service.storage_url == "http://test-minio:9000"

    @patch.dict("os.environ", {
        "STORAGE_URL": "http://fallback:7990",
    }, clear=False)
    def test_falls_back_to_storage_url(self):
        """When MINIO_ENDPOINT not set, falls back to STORAGE_URL."""
        import os
        os.environ.pop("MINIO_ENDPOINT", None)
        service = MediaService()
        assert service.storage_url == "http://fallback:7990"


class TestMediaServiceSignedUrl:
    """Test presigned URL generation (replaces HMAC)."""

    def test_generate_signed_url_uses_storage_client(self):
        service = MediaService()
        mock_client = MagicMock()
        mock_client.presigned_get.return_value = "http://minio:9000/oelala-users/user-abc/videos/test.mp4?X-Amz-Signature=xyz"
        service._storage_client = mock_client

        url = service.generate_signed_url(
            "users/user-abc/videos/test.mp4", expires_in=3600
        )

        assert "X-Amz-Signature" in url
        mock_client.presigned_get.assert_called_once_with(
            "users/user-abc", "videos/test.mp4", expires=3600
        )

    def test_generate_signed_url_simple_path(self):
        service = MediaService()
        mock_client = MagicMock()
        mock_client.presigned_get.return_value = "http://minio:9000/oelala-generated/video.mp4?sig=abc"
        service._storage_client = mock_client

        url = service.generate_signed_url("generated/video.mp4", expires_in=7200)

        mock_client.presigned_get.assert_called_once_with(
            "generated", "video.mp4", expires=7200
        )


class TestMediaServiceQuota:
    """Test quota calculation via Supabase."""

    @pytest.mark.asyncio
    async def test_quota_from_supabase(self):
        service = MediaService(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )

        # Mock tier fetch
        mock_http = AsyncMock()
        tier_resp = MagicMock()
        tier_resp.status_code = 200
        tier_resp.json.return_value = [{"tier": "free"}]

        # Mock media list fetch
        media_resp = MagicMock()
        media_resp.status_code = 200
        media_resp.json.return_value = [
            {"metadata": {"size_bytes": 1000}},
            {"metadata": {"size_bytes": 2000}},
            {"metadata": {"size_bytes": 500}},
        ]

        mock_http.get = AsyncMock(side_effect=[tier_resp, media_resp])
        service._http_client = mock_http

        quota = await service.get_user_quota("user-abc")

        assert quota["used_bytes"] == 3500
        assert quota["file_count"] == 3
        assert quota["tier"] == "free"
        assert quota["quota_bytes"] == 5 * 1024 * 1024 * 1024  # 5 GB for free tier


class TestMediaServiceUpload:
    """Test upload with MinIO storage client."""

    @pytest.mark.asyncio
    async def test_upload_uses_storage_client(self):
        service = MediaService(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )

        # Mock storage client
        mock_storage = MagicMock()
        mock_storage.put.return_value = {
            "bucket": "users",
            "key": "user-abc/videos/test.mp4",
            "size": 1000,
            "hash": "abc123",
            "content_type": "video/mp4",
        }
        service._storage_client = mock_storage

        # Mock HTTP client for tier + supabase insert
        mock_http = AsyncMock()
        tier_resp = MagicMock()
        tier_resp.status_code = 200
        tier_resp.json.return_value = [{"tier": "free"}]

        insert_resp = MagicMock()
        insert_resp.status_code = 201
        insert_resp.json.return_value = [{"id": "record-1", "created_at": "2026-01-01T00:00:00Z"}]

        mock_http.get = AsyncMock(return_value=tier_resp)
        mock_http.post = AsyncMock(return_value=insert_resp)
        service._http_client = mock_http

        record = await service.upload(
            user_id="user-abc",
            file_data=b"video bytes",
            filename="test.mp4",
            mime_type="video/mp4",
        )

        assert record.user_id == "user-abc"
        assert record.media_type == "video"
        mock_storage.put.assert_called_once()


class TestMediaServiceDelete:
    """Test delete with MinIO storage client."""

    @pytest.mark.asyncio
    async def test_hard_delete_uses_storage_client(self):
        service = MediaService(
            supabase_url="https://test.supabase.co",
            supabase_key="test-key",
        )

        # Mock storage client
        mock_storage = MagicMock()
        mock_storage.delete.return_value = True
        service._storage_client = mock_storage

        # Mock get() to return a record
        mock_http = AsyncMock()
        get_resp = MagicMock()
        get_resp.status_code = 200
        get_resp.json.return_value = [{
            "id": "record-1",
            "user_id": "user-abc",
            "storage_path": "users/user-abc/videos/test.mp4",
            "media_type": "video",
        }]

        delete_resp = MagicMock()
        delete_resp.status_code = 204

        mock_http.get = AsyncMock(return_value=get_resp)
        mock_http.delete = AsyncMock(return_value=delete_resp)
        service._http_client = mock_http

        result = await service.delete("record-1", hard_delete=True)

        assert result is True
        mock_storage.delete.assert_called_once_with("users/user-abc", "videos/test.mp4")


class TestTierRetention:
    """Test tier-based retention configuration."""

    def test_known_tiers(self):
        assert TIER_RETENTION_DAYS["free"] == 30
        assert TIER_RETENTION_DAYS["pro"] == 90
        assert TIER_RETENTION_DAYS["vip"] == 365

    def test_calculate_expires_at(self):
        service = MediaService()
        from datetime import datetime

        expires = service.calculate_expires_at("pro")
        now = datetime.utcnow()

        # Should be ~90 days in the future
        delta = (expires - now).days
        assert 89 <= delta <= 91
