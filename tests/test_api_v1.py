"""
Tests for API v1 REST endpoints.

These tests verify the public REST API for programmatic access.
"""

import pytest
import hashlib
from unittest.mock import Mock, AsyncMock, patch
from fastapi import FastAPI
from fastapi.testclient import TestClient

# Mock Supabase and dependencies before importing the app
import sys
from pathlib import Path

# Add src/backend to path
backend_dir = Path(__file__).parent.parent / "src" / "backend"
sys.path.insert(0, str(backend_dir))


@pytest.fixture
def mock_supabase():
    """Mock Supabase client."""
    with patch("credits.get_supabase_client") as mock:
        # Create a mock client
        client = Mock()

        # Mock RPC call for API key validation
        def mock_rpc(func_name, params):
            result = Mock()
            if func_name == "validate_api_key":
                # Simulate valid key
                if params.get("p_key_hash") == hashlib.sha256(
                    b"oelala_test_valid_key"
                ).hexdigest():
                    result.data = [
                        {
                            "valid": True,
                            "user_id": "test-user-123",
                            "key_id": "test-key-456",
                            "error": None,
                        }
                    ]
                else:
                    result.data = [
                        {
                            "valid": False,
                            "user_id": None,
                            "key_id": None,
                            "error": "Invalid API key",
                        }
                    ]
            result.execute = lambda: result
            return result

        client.rpc = mock_rpc

        # Mock from_ for balance queries
        def mock_from(table):
            query = Mock()
            query.select = lambda *args: query
            query.eq = lambda *args, **kwargs: query
            query.single = lambda: query

            if table == "user_credits":
                query.execute = lambda: Mock(
                    data={
                        "balance": 100,
                        "lifetime_purchased": 200,
                        "lifetime_used": 100,
                    }
                )

            return query

        client.from_ = mock_from

        mock.return_value = client
        yield client


@pytest.fixture
def mock_comfyui():
    """Mock ComfyUI client."""
    with patch("api_v1.get_comfyui_client") as mock:
        client = Mock()
        client.is_available = Mock(return_value=True)
        client.queue_prompt = Mock(return_value="prompt-123")
        mock.return_value = client
        yield client


@pytest.fixture
def mock_credits():
    """Mock credit operations."""
    with patch("api_v1.check_credits", new_callable=AsyncMock) as check_mock, patch(
        "api_v1.deduct_credits", new_callable=AsyncMock
    ) as deduct_mock:
        check_mock.return_value = None  # No exception = sufficient credits
        deduct_mock.return_value = None
        yield {"check": check_mock, "deduct": deduct_mock}


@pytest.fixture
def client(mock_supabase, mock_comfyui, mock_credits):
    """Test client with mocked dependencies."""
    # Set environment variable for service key
    import os

    os.environ["SUPABASE_SERVICE_KEY"] = "test-service-key"
    os.environ["SUPABASE_URL"] = "https://test.supabase.co"

    # Import app after mocks are set up
    from app import app

    return TestClient(app)


def test_health_check(client):
    """Test API v1 health check endpoint (no auth required)."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "timestamp" in data


def test_generate_without_api_key(client):
    """Test generation endpoint requires API key."""
    response = client.post(
        "/api/v1/generate",
        json={
            "type": "text-to-image",
            "prompt": "a beautiful landscape",
        },
    )
    assert response.status_code == 401
    assert "API key required" in response.json()["detail"]


def test_generate_with_invalid_api_key(client):
    """Test generation with invalid API key."""
    response = client.post(
        "/api/v1/generate",
        headers={"X-API-Key": "oelala_invalid_key"},
        json={
            "type": "text-to-image",
            "prompt": "a beautiful landscape",
        },
    )
    assert response.status_code == 401
    assert "Invalid or expired API key" in response.json()["detail"]


def test_generate_text_to_image(client, mock_credits):
    """Test text-to-image generation with valid API key."""
    response = client.post(
        "/api/v1/generate",
        headers={"X-API-Key": "oelala_test_valid_key"},
        json={
            "type": "text-to-image",
            "prompt": "a beautiful landscape",
            "width": 1024,
            "height": 1024,
            "steps": 20,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert data["status"] == "queued"
    assert data["credits_used"] > 0
    assert "estimated_time_seconds" in data

    # Verify credits were checked and deducted
    mock_credits["check"].assert_called_once()
    mock_credits["deduct"].assert_called_once()


def test_generate_text_to_video(client, mock_credits):
    """Test text-to-video generation."""
    response = client.post(
        "/api/v1/generate",
        headers={"X-API-Key": "oelala_test_valid_key"},
        json={
            "type": "text-to-video",
            "prompt": "a flowing river",
            "duration_seconds": 3,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "queued"
    assert data["credits_used"] > 0


def test_generate_image_to_video(client, mock_credits):
    """Test image-to-video generation."""
    response = client.post(
        "/api/v1/generate",
        headers={"X-API-Key": "oelala_test_valid_key"},
        json={
            "type": "image-to-video",
            "prompt": "camera pans slowly",
            "image_url": "https://example.com/image.jpg",
            "duration_seconds": 5,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "queued"
    assert data["credits_used"] > 0


def test_generate_invalid_type(client):
    """Test generation with unsupported type."""
    response = client.post(
        "/api/v1/generate",
        headers={"X-API-Key": "oelala_test_valid_key"},
        json={
            "type": "invalid-type",
            "prompt": "test",
        },
    )

    # Should fail validation
    assert response.status_code == 422


def test_get_job_status(client):
    """Test job status endpoint."""
    response = client.get(
        "/api/v1/jobs/test-job-123",
        headers={"X-API-Key": "oelala_test_valid_key"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "test-job-123"
    assert data["status"] in ["queued", "running", "completed", "failed"]
    assert "created_at" in data


def test_get_job_status_without_api_key(client):
    """Test job status requires authentication."""
    response = client.get("/api/v1/jobs/test-job-123")
    assert response.status_code == 401


def test_download_job_result(client):
    """Test job result download endpoint."""
    response = client.get(
        "/api/v1/jobs/test-job-123/download",
        headers={"X-API-Key": "oelala_test_valid_key"},
    )

    # Currently returns 404 (not implemented yet)
    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


def test_get_credits(client):
    """Test credits balance endpoint."""
    response = client.get(
        "/api/v1/credits",
        headers={"X-API-Key": "oelala_test_valid_key"},
    )

    assert response.status_code == 200
    data = response.json()
    assert "balance" in data
    assert "lifetime_purchased" in data
    assert "lifetime_used" in data
    assert isinstance(data["balance"], int)


def test_get_credits_without_api_key(client):
    """Test credits endpoint requires authentication."""
    response = client.get("/api/v1/credits")
    assert response.status_code == 401


def test_api_key_format_validation(client):
    """Test API key format validation."""
    # Test with key that doesn't start with "oelala_"
    response = client.post(
        "/api/v1/generate",
        headers={"X-API-Key": "invalid_prefix_key"},
        json={
            "type": "text-to-image",
            "prompt": "test",
        },
    )

    assert response.status_code == 401
    assert "Invalid API key format" in response.json()["detail"]


@pytest.mark.parametrize(
    "prompt,expected_status",
    [
        ("valid prompt", 200),
        ("", 422),  # Empty prompt should fail validation
    ],
)
def test_generate_prompt_validation(client, prompt, expected_status):
    """Test prompt validation."""
    response = client.post(
        "/api/v1/generate",
        headers={"X-API-Key": "oelala_test_valid_key"},
        json={
            "type": "text-to-image",
            "prompt": prompt,
        },
    )

    assert response.status_code == expected_status


def test_generate_dimension_validation(client):
    """Test dimension constraints."""
    # Test with invalid dimensions
    response = client.post(
        "/api/v1/generate",
        headers={"X-API-Key": "oelala_test_valid_key"},
        json={
            "type": "text-to-image",
            "prompt": "test",
            "width": 100,  # Too small (min 256)
            "height": 5000,  # Too large (max 2048)
        },
    )

    assert response.status_code == 422  # Validation error


def test_comfyui_unavailable(client, mock_comfyui):
    """Test graceful handling when ComfyUI is down."""
    mock_comfyui.is_available.return_value = False

    response = client.post(
        "/api/v1/generate",
        headers={"X-API-Key": "oelala_test_valid_key"},
        json={
            "type": "text-to-image",
            "prompt": "test",
        },
    )

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"].lower()
