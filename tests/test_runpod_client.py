"""Unit tests for RunPod endpoint selection and failover."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

backend_dir = Path(__file__).parent.parent / "src" / "backend"
sys.path.insert(0, str(backend_dir))

from runpod_client import RunPodClient


def test_init_uses_endpoint_list_when_default_missing(monkeypatch):
    """Client should derive its default endpoint from the configured endpoint list."""
    monkeypatch.delenv("RUNPOD_ENDPOINT_ID", raising=False)
    monkeypatch.setenv("RUNPOD_ENDPOINT_IDS", "ep-primary, ep-eu, ep-primary")

    client = RunPodClient(api_key="test-key")

    assert client.default_endpoint_id == "ep-primary"
    assert client.endpoint_ids == ["ep-primary", "ep-eu"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("health_map", "expected_endpoint"),
    [
        (
            {
                "ep-primary": {"workers": {"ready": 0, "throttled": 1, "unhealthy": 0}},
                "ep-eu": {"workers": {"ready": 1, "throttled": 0, "unhealthy": 0}},
            },
            "ep-eu",
        ),
        (
            {
                "ep-primary": {"workers": {"ready": 0, "throttled": 1, "unhealthy": 0}},
                "ep-eu": {"workers": {"ready": 0, "throttled": 0, "unhealthy": 1}},
            },
            "ep-primary",
        ),
    ],
)
async def test_select_submit_endpoint_handles_health_failover(
    monkeypatch,
    health_map,
    expected_endpoint,
):
    """Client should prefer healthy endpoints and otherwise fall back to the first candidate."""
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "ep-primary")
    monkeypatch.setenv("RUNPOD_ENDPOINT_IDS", "ep-primary,ep-eu")

    client = RunPodClient(api_key="test-key")
    client.get_endpoint_health = AsyncMock(side_effect=lambda endpoint_id: health_map[endpoint_id])

    selected = await client.select_submit_endpoint()

    assert selected == expected_endpoint
    if expected_endpoint == "ep-eu":
        assert client.default_endpoint_id == "ep-eu"
    else:
        assert client.default_endpoint_id == "ep-primary"


@pytest.mark.asyncio
async def test_select_submit_endpoint_keeps_explicit_endpoint(monkeypatch):
    """Explicit endpoint selections should bypass automatic failover."""
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "ep-primary")
    monkeypatch.setenv("RUNPOD_ENDPOINT_IDS", "ep-primary,ep-eu")

    client = RunPodClient(api_key="test-key")
    client.get_endpoint_health = AsyncMock()

    selected = await client.select_submit_endpoint("ep-explicit")

    assert selected == "ep-explicit"
    client.get_endpoint_health.assert_not_called()
