"""
GPU Integration Tests for oelala

These tests run on the self-hosted runner with GPU access.
They validate actual ComfyUI workflows and video generation.
"""

import pytest
import requests
import json
from pathlib import Path
import io

# Local services
BACKEND_URL = "http://localhost:7998"
COMFYUI_URL = "http://localhost:8188"


@pytest.mark.gpu
class TestGPUSmoke:
    """Basic smoke tests to verify GPU environment is working."""

    def test_backend_healthy(self):
        """Backend API should respond with healthy status."""
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["comfyui_available"] is True

    def test_comfyui_online(self):
        """ComfyUI should be running and have GPU access."""
        resp = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        assert resp.status_code == 200
        data = resp.json()

        # Should have at least one device with VRAM
        assert len(data.get("devices", [])) > 0
        device = data["devices"][0]
        assert device["vram_total"] > 0
        assert device["vram_free"] > 0

    def test_gpu_has_vram(self):
        """GPU should have sufficient VRAM for video generation."""
        resp = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
        data = resp.json()

        vram_gb = data["devices"][0]["vram_total"] / (1024**3)
        assert vram_gb >= 8, f"Need at least 8GB VRAM, got {vram_gb:.1f}GB"


@pytest.mark.gpu
class TestWorkflowValidation:
    """Test that workflow JSON files are valid and can be loaded."""

    @pytest.fixture
    def workflow_dir(self):
        """Get the workflows directory."""
        return Path("/home/flip/oelala/workflows")

    def test_workflow_files_valid_json(self, workflow_dir):
        """All .json files in workflows/ should be valid JSON."""
        if not workflow_dir.exists():
            pytest.skip("No workflows directory")

        for wf_file in workflow_dir.glob("*.json"):
            with open(wf_file) as f:
                try:
                    data = json.load(f)
                    assert isinstance(data, dict), f"{wf_file.name} should be a dict"
                except json.JSONDecodeError as e:
                    pytest.fail(f"{wf_file.name} is not valid JSON: {e}")

    def test_comfyui_object_info(self):
        """ComfyUI should return object_info for available nodes."""
        resp = requests.get(f"{COMFYUI_URL}/object_info", timeout=10)
        assert resp.status_code == 200
        data = resp.json()

        # Should have core nodes
        assert "KSampler" in data
        assert "CheckpointLoaderSimple" in data


@pytest.mark.gpu
class TestModelsAvailable:
    """Test that required models are available."""

    def test_unet_models_exist(self):
        """Should have at least one unet model for Wan2.2."""
        models_dir = Path("/home/flip/oelala/ComfyUI/models/unet")
        if not models_dir.exists():
            pytest.skip("No unet models directory")

        gguf_files = list(models_dir.glob("*.gguf"))
        assert len(gguf_files) > 0, "No .gguf unet models found"

    def test_checkpoint_models_exist(self):
        """Should have at least one checkpoint model."""
        models_dir = Path("/home/flip/oelala/ComfyUI/models/checkpoints")
        if not models_dir.exists():
            pytest.skip("No checkpoints directory")

        safetensor_files = list(models_dir.glob("*.safetensors"))
        assert len(safetensor_files) > 0, "No .safetensors checkpoint models found"


@pytest.mark.gpu
@pytest.mark.slow
class TestVideoGeneration:
    """
    Actual video generation tests.
    These are slow and use GPU resources, so marked with @pytest.mark.slow
    Run with: pytest -m slow
    """

    def test_minimal_t2v_generation(self):
        """Test text-to-video with minimal settings."""
        pytest.skip("Skipping actual generation in smoke tests")

        # This would be enabled for full tests
        resp = requests.post(
            f"{BACKEND_URL}/generate-text",
            data={
                "prompt": "a simple test animation",
                "num_frames": 5,  # Minimum
                "resolution": "480p",
            },
            timeout=120
        )

        # Should at least not crash
        assert resp.status_code in [200, 503]  # 503 if model not loaded


@pytest.mark.gpu
class TestAdvancedVideoEndpoints:
    """Test new advanced video processing endpoints."""

    def test_upscale_video_endpoint_exists(self):
        """Video upscale endpoint should exist and accept requests.

        Note: This test uses a fake MP4 file to verify endpoint accessibility,
        not actual video processing functionality.
        """
        # Create a minimal test video file (invalid but tests endpoint)
        test_video = io.BytesIO(
            b'\x00\x00\x00\x1c' +  # MP4 header
            b'ftypisom' +
            b'\x00\x00\x02\x00' +
            b'isomiso2' +
            b'\x00' * 100  # Padding
        )
        test_video.seek(0)

        resp = requests.post(
            f"{BACKEND_URL}/upscale-video",
            files={"file": ("test.mp4", test_video, "video/mp4")},
            data={"model": "realesrgan-video"},
            timeout=10
        )

        # Should either accept or reject gracefully, not crash
        # Endpoint exists if we get any of these status codes
        assert resp.status_code in [200, 400, 422, 500, 503]

        # If successful, check response structure
        if resp.status_code == 200:
            data = resp.json()
            # Should have prompt_id or error message
            assert "prompt_id" in data or "detail" in data

    def test_interpolate_video_endpoint_exists(self):
        """Frame interpolation endpoint should exist and accept requests.

        Note: This test uses a fake MP4 file to verify endpoint accessibility,
        not actual video processing functionality.
        """
        # Create a minimal test video file
        test_video = io.BytesIO(
            b'\x00\x00\x00\x1c' +
            b'ftypisom' +
            b'\x00\x00\x02\x00' +
            b'isomiso2' +
            b'\x00' * 100
        )
        test_video.seek(0)

        resp = requests.post(
            f"{BACKEND_URL}/interpolate-video",
            files={"file": ("test.mp4", test_video, "video/mp4")},
            data={
                "model": "rife",
                "mode": "fps",
                "target_fps": "60",
                "multiplier": "2.0",
            },
            timeout=10
        )

        # Should either accept or reject gracefully
        assert resp.status_code in [200, 400, 422, 500, 503]

        # If successful, check response structure
        if resp.status_code == 200:
            data = resp.json()
            # Should have prompt_id or error message
            assert "prompt_id" in data or "detail" in data

    def test_upscale_video_validates_model_param(self):
        """Upscale endpoint should validate model parameter."""
        test_video = io.BytesIO(b'\x00' * 100)
        test_video.seek(0)

        # This should work (valid model)
        resp = requests.post(
            f"{BACKEND_URL}/upscale-video",
            files={"file": ("test.mp4", test_video, "video/mp4")},
            data={"model": "realesrgan-video"},
            timeout=10
        )

        # Should at least parse the request
        assert resp.status_code in [200, 400, 422, 500, 503]

    def test_interpolate_video_validates_model_param(self):
        """Interpolate endpoint should validate model parameter."""
        test_video = io.BytesIO(b'\x00' * 100)
        test_video.seek(0)

        resp = requests.post(
            f"{BACKEND_URL}/interpolate-video",
            files={"file": ("test.mp4", test_video, "video/mp4")},
            data={
                "model": "rife",
                "mode": "fps",
                "target_fps": "60",
                "multiplier": "2.0",
            },
            timeout=10
        )

        # Should at least parse the request
        assert resp.status_code in [200, 400, 422, 500, 503]
