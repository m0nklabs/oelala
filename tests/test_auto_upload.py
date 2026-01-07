#!/usr/bin/env python3
"""
Unit tests for auto-upload functionality.
Tests job tracking and upload completion hooks.
"""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

from comfyui_client import ComfyUIClient


class TestJobTracking:
    """Test job metadata tracking."""

    def test_register_job(self):
        """Test that jobs can be registered with user_id."""
        client = ComfyUIClient()

        prompt_id = "test_prompt_123"
        user_id = "user_abc"
        prompt = "a beautiful sunset"
        settings = {"width": 512, "height": 512}

        # Register job
        client.register_job(prompt_id, user_id, prompt, settings)

        # Verify job metadata is stored
        metadata = client.get_job_metadata(prompt_id)
        assert metadata is not None
        assert metadata["user_id"] == user_id
        assert metadata["prompt"] == prompt
        assert metadata["settings"] == settings
        assert "started_at" in metadata

    def test_get_job_metadata_missing(self):
        """Test getting metadata for non-existent job."""
        client = ComfyUIClient()

        metadata = client.get_job_metadata("non_existent_prompt")
        assert metadata is None

    def test_clear_job_metadata(self):
        """Test clearing job metadata after completion."""
        client = ComfyUIClient()

        prompt_id = "test_prompt_456"
        client.register_job(prompt_id, "user_xyz", "test prompt", {})

        # Verify it exists
        assert client.get_job_metadata(prompt_id) is not None

        # Clear it
        client.clear_job_metadata(prompt_id)

        # Verify it's gone
        assert client.get_job_metadata(prompt_id) is None


class TestAutoUpload:
    """Test auto-upload on job completion."""

    @patch('storage_client.get_client')
    def test_on_job_complete_video(self, mock_get_storage_client):
        """Test auto-upload of video file."""
        client = ComfyUIClient()

        # Setup mock storage client
        mock_storage = MagicMock()
        mock_storage.put_user_media.return_value = {"status": "success"}
        mock_get_storage_client.return_value = mock_storage

        # Register a job
        prompt_id = "video_job_123"
        user_id = "user_video"
        client.register_job(prompt_id, user_id, "test video", {})

        # Create a temporary test file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.mp4', delete=False) as f:
            f.write(b"fake video content")
            test_file = f.name

        try:
            # Call on_job_complete
            storage_path = client.on_job_complete(prompt_id, test_file, "video")

            # Verify storage client was called
            assert mock_storage.put_user_media.called
            call_args = mock_storage.put_user_media.call_args
            assert call_args.kwargs["user_id"] == user_id
            assert call_args.kwargs["media_type"] == "videos"
            assert call_args.kwargs["content_type"] == "video/mp4"
            assert call_args.kwargs["data"] == b"fake video content"

            # Verify storage path returned
            assert storage_path is not None
            assert storage_path.startswith("videos/")

            # Verify job metadata was cleared
            assert client.get_job_metadata(prompt_id) is None
        finally:
            # Cleanup
            Path(test_file).unlink(missing_ok=True)

    @patch('storage_client.get_client')
    def test_on_job_complete_image(self, mock_get_storage_client):
        """Test auto-upload of image file."""
        client = ComfyUIClient()

        # Setup mock storage client
        mock_storage = MagicMock()
        mock_storage.put_user_media.return_value = {"status": "success"}
        mock_get_storage_client.return_value = mock_storage

        # Register a job
        prompt_id = "image_job_456"
        user_id = "user_image"
        client.register_job(prompt_id, user_id, "test image", {})

        # Create a temporary test file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.png', delete=False) as f:
            f.write(b"fake image content")
            test_file = f.name

        try:
            # Call on_job_complete
            storage_path = client.on_job_complete(prompt_id, test_file, "image")

            # Verify storage client was called
            assert mock_storage.put_user_media.called
            call_args = mock_storage.put_user_media.call_args
            assert call_args.kwargs["user_id"] == user_id
            assert call_args.kwargs["media_type"] == "images"
            assert call_args.kwargs["content_type"] == "image/png"

            # Verify storage path returned
            assert storage_path is not None
            assert storage_path.startswith("images/")

            # Verify job metadata was cleared
            assert client.get_job_metadata(prompt_id) is None
        finally:
            # Cleanup
            Path(test_file).unlink(missing_ok=True)

    @patch('storage_client.get_client')
    def test_on_job_complete_no_metadata(self, mock_get_storage_client):
        """Test that upload is skipped when no job metadata exists."""
        client = ComfyUIClient()

        # Don't register a job
        prompt_id = "missing_job"

        # Call on_job_complete without registering first
        storage_path = client.on_job_complete(prompt_id, "/fake/path.mp4", "video")

        # Verify upload was skipped
        assert storage_path is None
        mock_get_storage_client.assert_not_called()

    @patch('storage_client.get_client')
    def test_on_job_complete_upload_failure(self, mock_get_storage_client):
        """Test that failures don't raise exceptions."""
        client = ComfyUIClient()

        # Setup mock storage client that raises an exception
        mock_storage = MagicMock()
        mock_storage.put_user_media.side_effect = Exception("Upload failed")
        mock_get_storage_client.return_value = mock_storage

        # Register a job
        prompt_id = "fail_job"
        user_id = "user_fail"
        client.register_job(prompt_id, user_id, "test", {})

        # Create a temporary test file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.mp4', delete=False) as f:
            f.write(b"content")
            test_file = f.name

        try:
            # Call on_job_complete - should not raise
            storage_path = client.on_job_complete(prompt_id, test_file, "video")

            # Verify it returns None on failure
            assert storage_path is None

            # Verify job metadata is NOT cleared on failure
            assert client.get_job_metadata(prompt_id) is not None
        finally:
            # Cleanup
            Path(test_file).unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
