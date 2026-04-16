"""
Tests for the V2 Generation API endpoints.

Tests cover:
- POST /v2/generate — dispatch to mock adapter
- GET /v2/adapters — list registered adapters
- POST /v2/estimate — credit estimation
- Error handling (no adapter, uninitialized)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "backend"))

from fastapi import FastAPI
from fastapi.testclient import TestClient

from generation.types import (
    AdapterConstraints,
    ComputeTarget,
    GenerationRequest,
    GenerationResult,
    LoraFormat,
    MediaType,
    Operation,
)
from generation.adapter import GenerationAdapter
from generation.registry import AdapterRegistry
from generation.router import GenerationRouter
from generation.v2_api import router, init_v2_api


class FakeLocalAdapter(GenerationAdapter):
    """Fake adapter for endpoint testing."""

    name = "fake-local-t2i"
    model_family = "test"
    supported_ops = {Operation.GENERATE}
    input_types = {MediaType.TEXT}
    output_type = MediaType.IMAGE
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.SINGLE_STAGE

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=1024,
            max_height=1024,
            default_steps=20,
            default_cfg=7.0,
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        return {"test": True}

    def cost(self, req: GenerationRequest) -> int:
        return 5

    async def execute(self, req, progress_callback=None):
        return GenerationResult(
            prompt_id="test-prompt-123",
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=0,
            adapter_name=self.name,
            meta={"prompt": req.prompt},
        )


class FakeEditAdapter(GenerationAdapter):
    """Fake edit adapter for endpoint testing."""

    name = "fake-cloud-edit"
    model_family = "qwen_image_edit"
    supported_ops = {Operation.EDIT}
    input_types = {MediaType.IMAGE}
    output_type = MediaType.IMAGE
    compute = ComputeTarget.CLOUD
    lora_format = LoraFormat.SINGLE_STAGE

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            min_width=512,
            max_width=2048,
            default_steps=40,
            default_cfg=4.0,
            supports_lightning=True,
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        return {"edit": True}

    def cost(self, req: GenerationRequest) -> int:
        base = 15
        if not req.lightning:
            base += 5
        return base

    async def execute(self, req, progress_callback=None):
        return GenerationResult(
            prompt_id="edit-prompt-456",
            status="queued_cloud",
            compute_target=ComputeTarget.CLOUD,
            credits_used=0,
            runpod_job_id="rp-789",
            adapter_name=self.name,
        )


@pytest.fixture
def test_app():
    """Create a FastAPI test app with v2 endpoints."""
    app = FastAPI()

    registry = AdapterRegistry()
    registry.register(FakeLocalAdapter())
    registry.register(FakeEditAdapter())

    gen_router = GenerationRouter(registry)

    mock_check_credits = AsyncMock()
    mock_deduct_credits = AsyncMock(return_value=True)

    init_v2_api(
        registry=registry,
        gen_router=gen_router,
        get_current_user=None,
        check_credits=mock_check_credits,
        deduct_credits=mock_deduct_credits,
    )

    app.include_router(router)
    return TestClient(app)


class TestV2Adapters:
    def test_list_adapters(self, test_app):
        resp = test_app.get("/v2/adapters")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        names = [a["name"] for a in data["adapters"]]
        assert "fake-local-t2i" in names
        assert "fake-cloud-edit" in names

    def test_adapter_constraints_included(self, test_app):
        resp = test_app.get("/v2/adapters")
        data = resp.json()
        local_adapter = next(
            a for a in data["adapters"] if a["name"] == "fake-local-t2i"
        )
        assert "constraints" in local_adapter
        assert local_adapter["constraints"]["max_width"] == 1024
        assert local_adapter["constraints"]["default_steps"] == 20


class TestV2Generate:
    def test_generate_text_to_image(self, test_app):
        resp = test_app.post(
            "/v2/generate",
            json={
                "operation": "generate",
                "target_type": "image",
                "prompt": "a beautiful landscape",
                "adapter_hint": "fake-local-t2i",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued_local"
        assert data["adapter_name"] == "fake-local-t2i"
        assert data["credits_used"] == 5

    def test_generate_edit(self, test_app):
        resp = test_app.post(
            "/v2/generate",
            json={
                "operation": "edit",
                "target_type": "image",
                "instruction": "remove background",
                "input_images": ["base64data"],
                "adapter_hint": "fake-cloud-edit",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "queued_cloud"
        assert data["adapter_name"] == "fake-cloud-edit"
        assert data["runpod_job_id"] == "rp-789"

    def test_generate_no_adapter_found(self, test_app):
        resp = test_app.post(
            "/v2/generate",
            json={
                "operation": "upscale",
                "target_type": "video",
            },
        )
        assert resp.status_code == 400
        assert "No adapter found" in resp.json()["detail"]

    def test_generate_invalid_adapter_hint(self, test_app):
        resp = test_app.post(
            "/v2/generate",
            json={
                "operation": "generate",
                "target_type": "image",
                "adapter_hint": "nonexistent-adapter",
            },
        )
        assert resp.status_code == 400
        assert "not found" in resp.json()["detail"]

    def test_generate_invalid_operation(self, test_app):
        resp = test_app.post(
            "/v2/generate",
            json={
                "operation": "invalid_op",
                "target_type": "image",
            },
        )
        assert resp.status_code == 422  # Pydantic validation


class TestV2Estimate:
    def test_estimate_cost(self, test_app):
        resp = test_app.post(
            "/v2/estimate",
            json={
                "operation": "generate",
                "target_type": "image",
                "prompt": "test",
                "adapter_hint": "fake-local-t2i",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["adapter"] == "fake-local-t2i"
        assert data["credits_required"] == 5
        assert "constraints" in data

    def test_estimate_edit_with_lightning(self, test_app):
        resp = test_app.post(
            "/v2/estimate",
            json={
                "operation": "edit",
                "target_type": "image",
                "input_images": ["base64data"],
                "lightning": True,
                "adapter_hint": "fake-cloud-edit",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["credits_required"] == 15  # lightning = no +5

    def test_estimate_no_adapter(self, test_app):
        resp = test_app.post(
            "/v2/estimate",
            json={
                "operation": "upscale",
                "target_type": "video",
            },
        )
        assert resp.status_code == 400
