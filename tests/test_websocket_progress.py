#!/usr/bin/env python3
"""
Tests for WebSocket Progress Events and Queue Tracking
"""

import pytest
import json
from unittest.mock import AsyncMock, patch

# Import modules to test - use relative imports when possible
import sys
from pathlib import Path

# Add backend directory to path for testing
backend_path = Path(__file__).parent.parent / "src" / "backend"
sys.path.insert(0, str(backend_path))

from websocket_handler import WebSocketManager
from job_queue import JobQueueManager


class TestWebSocketManager:
    """Test WebSocket event broadcasting"""

    @pytest.mark.asyncio
    async def test_connect_disconnect(self):
        """Test WebSocket connection lifecycle"""
        manager = WebSocketManager()
        mock_ws = AsyncMock()

        # Connect
        await manager.connect(mock_ws, user_id="test_user")
        assert "test_user" in manager.connections
        assert mock_ws in manager.connections["test_user"]
        mock_ws.accept.assert_called_once()

        # Disconnect
        manager.disconnect(mock_ws, user_id="test_user")
        assert mock_ws not in manager.connections.get("test_user", set())

    @pytest.mark.asyncio
    async def test_job_registration(self):
        """Test job ownership tracking"""
        manager = WebSocketManager()

        manager.register_job("job123", user_id="user1")
        assert manager.job_ownership["job123"] == "user1"

        manager.unregister_job("job123")
        assert "job123" not in manager.job_ownership

    @pytest.mark.asyncio
    async def test_broadcast_queue_update(self):
        """Test queue update broadcasting"""
        manager = WebSocketManager()
        mock_ws = AsyncMock()

        # Setup connection
        await manager.connect(mock_ws, user_id="user1")
        manager.register_job("job123", user_id="user1")

        # Broadcast queue update
        await manager.broadcast_queue_update(
            job_id="job123", queue_position=3, total_pending=5, eta_seconds=120
        )

        # Verify message was sent
        mock_ws.send_text.assert_called_once()
        call_args = mock_ws.send_text.call_args[0][0]
        message = json.loads(call_args)

        assert message["type"] == "queue_update"
        assert message["data"]["job_id"] == "job123"
        assert message["data"]["queue_position"] == 3
        assert message["data"]["total_pending"] == 5
        assert message["data"]["eta_seconds"] == 120
        assert message["data"]["status"] == "queued"

    @pytest.mark.asyncio
    async def test_broadcast_progress(self):
        """Test progress event broadcasting"""
        manager = WebSocketManager()
        mock_ws = AsyncMock()

        await manager.connect(mock_ws, user_id="user1")
        manager.register_job("job456", user_id="user1")

        # Broadcast progress
        await manager.broadcast_progress(
            job_id="job456", progress=45, message="Processing...", node_name="VAE Encode"
        )

        mock_ws.send_text.assert_called()
        call_args = mock_ws.send_text.call_args[0][0]
        message = json.loads(call_args)

        assert message["type"] == "progress"
        assert message["data"]["job_id"] == "job456"
        assert message["data"]["progress"] == 45
        assert message["data"]["message"] == "Processing..."
        assert message["data"]["node_name"] == "VAE Encode"

    @pytest.mark.asyncio
    async def test_broadcast_job_complete(self):
        """Test job completion broadcasting"""
        manager = WebSocketManager()
        mock_ws = AsyncMock()

        await manager.connect(mock_ws, user_id="user1")
        manager.register_job("job789", user_id="user1")

        # Broadcast completion
        await manager.broadcast_job_complete(
            job_id="job789",
            output_url="/comfyui-output/video.mp4",
            metadata={"duration": 5.2},
        )

        mock_ws.send_text.assert_called()
        call_args = mock_ws.send_text.call_args[0][0]
        message = json.loads(call_args)

        assert message["type"] == "job_complete"
        assert message["data"]["job_id"] == "job789"
        assert message["data"]["status"] == "completed"
        assert message["data"]["output_url"] == "/comfyui-output/video.mp4"
        assert "job789" not in manager.job_ownership  # Job unregistered after completion

    @pytest.mark.asyncio
    async def test_broadcast_job_failed(self):
        """Test job failure broadcasting"""
        manager = WebSocketManager()
        mock_ws = AsyncMock()

        await manager.connect(mock_ws, user_id="user1")
        manager.register_job("job999", user_id="user1")

        # Broadcast failure
        await manager.broadcast_job_failed(
            job_id="job999", error="Out of memory", metadata={"node": "7"}
        )

        mock_ws.send_text.assert_called()
        call_args = mock_ws.send_text.call_args[0][0]
        message = json.loads(call_args)

        assert message["type"] == "job_failed"
        assert message["data"]["job_id"] == "job999"
        assert message["data"]["status"] == "failed"
        assert message["data"]["error"] == "Out of memory"

    @pytest.mark.asyncio
    async def test_multiple_clients_per_user(self):
        """Test broadcasting to multiple connections for same user"""
        manager = WebSocketManager()
        mock_ws1 = AsyncMock()
        mock_ws2 = AsyncMock()

        # Connect two clients for same user
        await manager.connect(mock_ws1, user_id="user1")
        await manager.connect(mock_ws2, user_id="user1")

        assert len(manager.connections["user1"]) == 2

        manager.register_job("job_multi", user_id="user1")

        # Broadcast should reach both clients
        await manager.broadcast_progress(job_id="job_multi", progress=50)

        mock_ws1.send_text.assert_called()
        mock_ws2.send_text.assert_called()

    @pytest.mark.asyncio
    async def test_eta_formatting(self):
        """Test ETA human-readable formatting"""
        manager = WebSocketManager()

        assert manager._format_eta(30) == "30s"
        assert manager._format_eta(90) == "1m 30s"
        assert manager._format_eta(3661) == "1h 1m"


class TestJobQueueManager:
    """Test job queue tracking and ETA estimation"""

    def test_job_registration(self):
        """Test job registration and metadata storage"""
        manager = JobQueueManager()

        manager.register_job(
            prompt_id="prompt123",
            user_id="user1",
            job_type="i2v",
            metadata={"width": 480, "height": 848},
        )

        job = manager.get_job("prompt123")
        assert job is not None
        assert job["prompt_id"] == "prompt123"
        assert job["user_id"] == "user1"
        assert job["job_type"] == "i2v"
        assert job["status"] == "queued"
        assert job["metadata"]["width"] == 480

    def test_job_status_updates(self):
        """Test job status transitions"""
        manager = JobQueueManager()
        manager.register_job("prompt456", user_id="user2")

        # Update to running
        manager.update_job_status("prompt456", "running", started_at=1234567890.0)
        job = manager.get_job("prompt456")
        assert job["status"] == "running"
        assert job["started_at"] == 1234567890.0

        # Complete job
        manager.complete_job("prompt456")
        assert job["status"] == "completed"
        assert job["completed_at"] is not None

    def test_eta_estimation(self):
        """Test ETA calculation based on historical data"""
        manager = JobQueueManager()

        # No history - should use default
        eta = manager.estimate_eta(queue_position=2)
        assert eta == 240  # 2 * 120 (default)

        # Add completion history
        manager.completion_times.extend([100.0, 120.0, 110.0])

        # Should use average
        eta = manager.estimate_eta(queue_position=2)
        avg = sum(manager.completion_times) / len(manager.completion_times)
        assert eta == int(avg * 2)

    def test_average_execution_time(self):
        """Test average execution time calculation"""
        manager = JobQueueManager()

        # No data
        assert manager.get_average_execution_time() == 120.0

        # With data
        manager.completion_times.extend([100.0, 150.0, 125.0])
        assert manager.get_average_execution_time() == 125.0

    @pytest.mark.asyncio
    async def test_poll_queue_updates(self):
        """Test queue polling and position detection"""
        manager = JobQueueManager()
        mock_ws_manager = AsyncMock()

        # Register a job
        manager.register_job("prompt789", user_id="user1")

        # Mock ComfyUI queue response
        mock_queue_data = {
            "queue_running": [],
            "queue_pending": [
                [1, "prompt789", {"prompt": {}}],
                [2, "other_job", {"prompt": {}}],
            ],
        }

        with patch.object(manager, "get_comfyui_queue", return_value=mock_queue_data):
            await manager.poll_queue_updates(mock_ws_manager)

        # Should broadcast queue position
        mock_ws_manager.broadcast_queue_update.assert_called()
        call_args = mock_ws_manager.broadcast_queue_update.call_args[1]
        assert call_args["job_id"] == "prompt789"
        assert call_args["queue_position"] == 1

    @pytest.mark.asyncio
    async def test_job_transition_to_running(self):
        """Test detection of job starting execution"""
        manager = JobQueueManager()
        mock_ws_manager = AsyncMock()

        manager.register_job("prompt_run", user_id="user1")

        # Mock queue with job running
        mock_queue_data = {
            "queue_running": [[1, "prompt_run", {"prompt": {}}]],
            "queue_pending": [],
        }

        with patch.object(manager, "get_comfyui_queue", return_value=mock_queue_data):
            await manager.poll_queue_updates(mock_ws_manager)

        # Job should be marked as running
        job = manager.get_job("prompt_run")
        assert job["status"] == "running"
        assert job["started_at"] is not None

        # Should broadcast running status
        mock_ws_manager.broadcast_queue_update.assert_called()
        call_args = mock_ws_manager.broadcast_queue_update.call_args[1]
        assert call_args["queue_position"] == 0

    def test_output_url_extraction(self):
        """Test extraction of output URL from ComfyUI history"""
        manager = JobQueueManager()

        # Video output
        history = {
            "outputs": {
                "12": {
                    "gifs": [{"filename": "video.mp4", "type": "output", "subfolder": ""}]
                }
            }
        }
        url = manager._extract_output_url(history)
        assert url == "/comfyui-output/video.mp4"

        # Image output
        history = {
            "outputs": {
                "8": {
                    "images": [
                        {"filename": "image.png", "type": "output", "subfolder": ""}
                    ]
                }
            }
        }
        url = manager._extract_output_url(history)
        assert url == "/comfyui-output/image.png"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
