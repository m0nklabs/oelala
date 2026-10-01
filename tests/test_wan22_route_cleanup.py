"""
Tests for the v1 route cleanup that followed the Wan 2.2 retirement (2026-10-01).

Wan 2.2 was removed from the product, but several legacy v1 routes in
``src/backend/app.py`` stayed hard-pinned to Wan adapters, so every call to them
ended in an HTTP 400 from ``generation/router.py::RETIRED_MODEL_FAMILIES``.

This module locks in the cleanup:

1. Retired Wan-specific routes still answer **400** with the canonical message
   naming MiniMax-H3 and LTX-2.3 (no bare 404, no confusing 500).
2. Repointed routes (``/generate``, ``/generate-pose``, ``/generate-text``)
   resolve a **real, live** adapter and complete normally.
3. ``/v2/generate`` — the surface the frontend uses — still works, and still
   rejects a request that names the retired family.

The registry holds the real adapter classes; only ``execute()`` is stubbed, so
no ComfyUI or RunPod work happens.
"""

import os
import sys
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# NOTE: the app imports the generation package as ``src.backend.generation.*``.
# Importing it as ``generation.*`` would create a second copy of the package
# with its own module globals (and its own pydantic classes), so the v1 compat
# layer this test initialises would not be the one app.py dispatches through.
from src.backend.generation.adapter import GenerationAdapter  # noqa: E402
from src.backend.generation.registry import AdapterRegistry  # noqa: E402
from src.backend.generation.router import (  # noqa: E402
    RETIRED_MODEL_FAMILIES,
    GenerationRouter,
)
from src.backend.generation.types import (  # noqa: E402
    ComputeTarget,
    GenerationResult,
)
from src.backend.generation import v1_compat, v2_api  # noqa: E402

from src.backend.generation.adapters.local.minimax_h3_i2v import (  # noqa: E402
    MiniMaxH3LocalI2VAdapter,
)
from src.backend.generation.adapters.local.minimax_h3_t2v import (  # noqa: E402
    MiniMaxH3LocalT2VAdapter,
)
from src.backend.generation.adapters.cloud.minimax_h3_t2v import (  # noqa: E402
    MiniMaxH3CloudT2VAdapter,
)
from src.backend.generation.adapters.cloud.ltx23_t2v import (  # noqa: E402
    LTX23CloudT2VAdapter,
)


class FakeUser:
    """Minimal stand-in for auth.User (only .id is read on this path)."""

    id = "test-user-wan22-cleanup"


async def _fake_current_user(request=None, credentials=None):
    return FakeUser()


# Wan-specific endpoints. They are retired: they exist only to answer with the
# canonical retirement error instead of a bare 404.
RETIRED_WAN22_ROUTES = [
    "/generate-wan22-comfyui",
    "/generate-wan22-async",
    "/generate-blockswap-q8-async",
    "/generate-distorch2-q8-async",
    "/generate-ultra-q8-async",
    "/generate-cloud-wan22-async",
]

# Retired family spellings a caller may still send as a model selection.
RETIRED_MODEL_TYPE_SPELLINGS = ["wan22", "wan2.2", "cloud_wan22", "WAN22"]


def _stub_execute(adapter: GenerationAdapter) -> AsyncMock:
    """Replace execute() so the real adapter never touches ComfyUI/RunPod."""
    mock = AsyncMock(
        return_value=GenerationResult(
            prompt_id=f"stub-{adapter.name}",
            status="queued_local"
            if adapter.compute == ComputeTarget.LOCAL
            else "queued_cloud",
            compute_target=adapter.compute,
            credits_used=0,
            adapter_name=adapter.name,
        )
    )
    adapter.execute = mock
    return mock


@pytest.fixture
def live_adapters():
    """Real adapter instances (execution stubbed) for the live video families."""
    adapters = [
        MiniMaxH3LocalI2VAdapter(comfyui_client_fn=lambda: None),
        MiniMaxH3LocalT2VAdapter(comfyui_client_fn=lambda: None),
        MiniMaxH3CloudT2VAdapter(
            submit_to_runpod_fn=AsyncMock(), comfyui_client_fn=lambda: None
        ),
        LTX23CloudT2VAdapter(
            submit_to_runpod_fn=AsyncMock(), comfyui_client_fn=lambda: None
        ),
    ]
    return {adapter.name: adapter for adapter in adapters}


@pytest.fixture
def client(live_adapters):
    """TestClient wired to the real app with the live adapters registered."""
    import app as app_module

    registry = AdapterRegistry()
    executes = {}
    for name, adapter in live_adapters.items():
        executes[name] = _stub_execute(adapter)
        registry.register(adapter)

    gen_router = GenerationRouter(registry)

    v1_compat.init_v1_compat(
        router=gen_router,
        check_credits=AsyncMock(),
        deduct_credits=AsyncMock(),
        get_comfyui_client=None,
    )
    v2_api.init_v2_api(
        registry=registry,
        gen_router=gen_router,
        get_current_user=_fake_current_user,
        check_credits=AsyncMock(),
        deduct_credits=AsyncMock(),
    )

    app_module.app.dependency_overrides[app_module.get_current_user] = (
        lambda: FakeUser()
    )
    try:
        yield TestClient(app_module.app), executes
    finally:
        app_module.app.dependency_overrides.clear()


@pytest.fixture
def no_startup_client(client):
    """Convenience: just the TestClient."""
    return client[0]


def _assert_retired_message(detail: str) -> None:
    """Every retirement error must name the live families."""
    assert "retired" in detail.lower(), detail
    assert "Wan 2.2" in detail, detail
    assert "MiniMax-H3" in detail, detail
    assert "LTX-2.3" in detail, detail


class TestRetiredWan22Routes:
    """Wan-only endpoints must fail clearly, not 404 or 500."""

    @pytest.mark.parametrize("path", RETIRED_WAN22_ROUTES)
    def test_retired_route_returns_400(self, no_startup_client, path):
        resp = no_startup_client.post(path, data={"prompt": "a cat"})
        assert resp.status_code == 400, (path, resp.status_code, resp.text)
        _assert_retired_message(resp.json()["detail"])

    @pytest.mark.parametrize("path", RETIRED_WAN22_ROUTES)
    def test_retired_route_message_names_live_families(self, no_startup_client, path):
        detail = no_startup_client.post(path, data={"prompt": "a cat"}).json()["detail"]
        assert path in detail, detail

    def test_retired_routes_are_registered_not_removed(self, no_startup_client):
        """A bare 404 would be a worse experience than an explicit 400."""
        registered = {route.path for route in no_startup_client.app.routes}
        for path in RETIRED_WAN22_ROUTES:
            assert path in registered, f"{path} disappeared from the v1 surface"


class TestRemovedWanInventoryRoute:
    """GET /unet-models was a Wan-era inventory route with nothing left to serve.

    It listed GGUF unets in ComfyUI/models/unet, grouped into the high/low-noise
    pairs a Wan 2.2 dual-pass workflow needed. The directory now holds no GGUF
    file at all, no code called the endpoint, and every live family loads
    safetensors checkpoints — so it was removed instead of kept empty.
    """

    def test_unet_models_route_is_gone(self, no_startup_client):
        registered = {route.path for route in no_startup_client.app.routes}
        assert "/unet-models" not in registered
        assert no_startup_client.get("/unet-models").status_code == 404

    def test_no_source_file_references_the_removed_route(self):
        from pathlib import Path

        repo = Path(__file__).resolve().parent.parent
        hits = []
        for root in (repo / "src" / "backend", repo / "scripts"):
            for path in root.rglob("*.py"):
                if "__pycache__" in path.parts:
                    continue
                if "unet-models" in path.read_text(errors="ignore"):
                    hits.append(str(path.relative_to(repo)))
        assert hits == [], hits


class TestRepointedGenerateRoutes:
    """/generate and /generate-pose now run the leading local family."""

    def test_generate_reaches_a_live_adapter(self, client):
        http, executes = client
        resp = http.post(
            "/generate",
            files={"file": ("frame.png", b"\x89PNG\r\n\x1a\n", "image/png")},
            data={"prompt": "a slow dolly shot", "num_frames": "41"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "queued"
        executes["minimax-h3-local-i2v"].assert_awaited_once()

    def test_generate_no_longer_names_a_retired_adapter(self, client):
        http, executes = client
        http.post(
            "/generate",
            files={"file": ("frame.png", b"\x89PNG\r\n\x1a\n", "image/png")},
            data={"prompt": "x"},
        )
        for name in executes:
            assert not any(
                family.replace("_", "").replace("-", "").replace(".", "")
                in name.replace("-", "")
                for family in RETIRED_MODEL_FAMILIES
            ), name

    def test_generate_pose_reaches_the_same_live_adapter(self, client):
        http, executes = client
        resp = http.post(
            "/generate-pose",
            files={"file": ("frame.png", b"\x89PNG\r\n\x1a\n", "image/png")},
            data={"num_frames": "41"},
        )
        assert resp.status_code == 200, resp.text
        executes["minimax-h3-local-i2v"].assert_awaited_once()

    def test_generate_rejects_a_retired_adapter_hint_via_v2(self, client):
        """The retirement guard is still the single source of the 400."""
        http, _ = client
        resp = http.post(
            "/v2/generate",
            json={
                "operation": "generate",
                "target_type": "video",
                "prompt": "x",
                "adapter_hint": "wan22-local-i2v-q6",
            },
        )
        assert resp.status_code == 400, resp.text
        _assert_retired_message(resp.json()["detail"])


class TestGenerateTextModelTypes:
    """/generate-text accepts live model types and rejects retired ones."""

    @pytest.mark.parametrize("model_type", RETIRED_MODEL_TYPE_SPELLINGS)
    def test_retired_model_type_returns_400(self, no_startup_client, model_type):
        resp = no_startup_client.post(
            "/generate-text", data={"prompt": "a cat", "model_type": model_type}
        )
        assert resp.status_code == 400, (model_type, resp.status_code, resp.text)
        _assert_retired_message(resp.json()["detail"])

    def test_default_model_type_is_live(self, client):
        """No model_type at all must not fall back to the retired family."""
        http, executes = client
        resp = http.post("/generate-text", data={"prompt": "a cat"})
        assert resp.status_code == 200, resp.text
        # Default compute_target is "local", so the local MiniMax-H3 T2V
        # adapter is the one that runs — never a retired Wan adapter.
        assert resp.json()["status"] == "queued"
        executes["minimax-h3-local-t2v"].assert_awaited_once()

    @pytest.mark.parametrize(
        "model_type,compute_target,expected_adapter",
        [
            ("minimax_h3", "cloud", "minimax-h3-cloud-t2v"),
            ("minimax_h3", "local", "minimax-h3-local-t2v"),
            ("minimax_h3_local", "local", "minimax-h3-local-t2v"),
            ("ltx2", "cloud", "ltx23-cloud-t2v"),
            ("ltx23", "cloud", "ltx23-cloud-t2v"),
        ],
    )
    def test_live_model_types_resolve_the_right_adapter(
        self, client, model_type, compute_target, expected_adapter
    ):
        http, executes = client
        resp = http.post(
            "/generate-text",
            data={
                "prompt": "a cat",
                "model_type": model_type,
                "compute_target": compute_target,
            },
        )
        assert resp.status_code == 200, (model_type, resp.status_code, resp.text)
        executes[expected_adapter].assert_awaited_once()

    def test_unknown_model_type_is_not_silently_swapped(self, no_startup_client):
        """An unknown model type must error, never fall back to another family."""
        resp = no_startup_client.post(
            "/generate-text", data={"prompt": "a cat", "model_type": "not-a-model"}
        )
        assert resp.status_code == 400, resp.text
        assert "not-a-model" in resp.json()["detail"]


class TestV2SurfaceUnchanged:
    """/v2/generate is what the frontend calls — it must keep working."""

    def test_live_v2_request_succeeds(self, client):
        http, executes = client
        resp = http.post(
            "/v2/generate",
            json={
                "operation": "generate",
                "target_type": "video",
                "prompt": "a cat",
                "adapter_hint": "ltx23-cloud-t2v",
            },
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["adapter_name"] == "ltx23-cloud-t2v"
        executes["ltx23-cloud-t2v"].assert_awaited_once()

    def test_adapters_endpoint_lists_live_families(self, no_startup_client):
        resp = no_startup_client.get("/v2/adapters")
        assert resp.status_code == 200, resp.text
        names = {a["name"] for a in resp.json()["adapters"]}
        assert {"minimax-h3-local-i2v", "ltx23-cloud-t2v"} <= names
        assert not any("wan22" in name for name in names)


class TestCreditPricingAfterRepoint:
    """Repointing the live surface must not move anyone's price."""

    def test_legacy_and_live_tiers_price_identically(self):
        from credits import calculate_credits

        for width, height in [(720, 720), (1920, 1080), (1280, 720), (480, 480)]:
            for duration in [None, 2, 3, 5, 6]:
                kwargs = dict(width=width, height=height, duration_seconds=duration)
                assert calculate_credits(
                    "wan22_i2v", **kwargs
                ) == calculate_credits("video_i2v", **kwargs)
                assert calculate_credits(
                    "wan22_t2v", **kwargs
                ) == calculate_credits("video_t2v", **kwargs)

    def test_historical_wan22_generation_types_still_resolve(self):
        from credits import calculate_credits

        # These strings are persisted in credit_transactions.metadata.
        assert calculate_credits(
            "wan22_i2v", width=720, height=720, duration_seconds=3
        ) == 5
        assert calculate_credits(
            "wan22_t2v", width=720, height=720, duration_seconds=3
        ) == 8
        assert calculate_credits(
            "wan22_i2v", width=1920, height=1080, duration_seconds=5
        ) == 15

    def test_live_generation_types_resolve(self):
        from credits import calculate_credits

        assert calculate_credits(
            "video_i2v", width=720, height=720, duration_seconds=3
        ) == 5
        assert calculate_credits(
            "video_t2v", width=720, height=720, duration_seconds=3
        ) == 8
        # The public v1 REST API bills video through the live tier.
        assert calculate_credits(
            "video_i2v", width=512, height=512, duration_seconds=3
        ) == 5
