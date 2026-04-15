"""
Tests for Phase 3 cloud adapters: Wan22 I2V/T2V + LTX-2.3 I2V/T2V.

Tests cover:
- Adapter metadata (name, model_family, ops, types, compute, LoRA format)
- Constraints
- Cost calculations
- Execute with mocked RunPod submission
- Build workflow delegation to ComfyUI client
"""

import os
import sys

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "backend"))

from generation.types import (
    AdapterConstraints,
    ComputeTarget,
    GenerationRequest,
    GenerationResult,
    LoraFormat,
    LoraStackItem,
    MediaType,
    Operation,
)
from generation.adapters.cloud.wan22_i2v import Wan22CloudI2VAdapter
from generation.adapters.cloud.wan22_t2v import Wan22CloudT2VAdapter
from generation.adapters.cloud.ltx23_i2v import LTX23CloudI2VAdapter
from generation.adapters.cloud.ltx23_t2v import LTX23CloudT2VAdapter


# ── Wan22 Cloud I2V ─────────────────────────────────────────────────


class TestWan22CloudI2V:
    def test_metadata(self):
        adapter = Wan22CloudI2VAdapter()
        assert adapter.name == "wan22-cloud-i2v"
        assert adapter.model_family == "wan2.2"
        assert Operation.GENERATE in adapter.supported_ops
        assert MediaType.IMAGE in adapter.input_types
        assert adapter.output_type == MediaType.VIDEO
        assert adapter.compute == ComputeTarget.CLOUD
        assert adapter.lora_format == LoraFormat.DUAL_STAGE

    def test_constraints(self):
        adapter = Wan22CloudI2VAdapter()
        c = adapter.constraints()
        assert c.max_frames == 321
        assert "480p" in c.resolution_presets
        assert "720p" in c.resolution_presets
        assert c.default_steps == 15
        assert c.default_cfg == 3.0
        assert 16 in c.allowed_fps

    @pytest.mark.parametrize("frames,expected_cost", [
        (81, 10),    # 5 base × 2 cloud
        (161, 16),   # 8 base × 2 cloud
        (321, 30),   # 15 base × 2 cloud
    ])
    def test_cost(self, frames, expected_cost):
        adapter = Wan22CloudI2VAdapter()
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="test",
            frames=frames,
        )
        assert adapter.cost(req) == expected_cost

    def test_build_workflow_delegates_to_comfyui(self):
        mock_comfyui = MagicMock()
        mock_comfyui.build_cloud_wan22_i2v_workflow.return_value = {"workflow": True}

        adapter = Wan22CloudI2VAdapter(comfyui_client_fn=lambda: mock_comfyui)
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="a cat",
            input_images=["img.png"],
            frames=81,
            fps=16,
        )
        result = adapter.build_workflow(req)
        assert result == {"workflow": True}
        mock_comfyui.build_cloud_wan22_i2v_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_success(self):
        mock_submit = AsyncMock(return_value={
            "prompt_id": "test-id",
            "runpod_job_id": "rp-123",
        })
        mock_comfyui = MagicMock()
        mock_comfyui.build_cloud_wan22_i2v_workflow.return_value = {"workflow": True}

        adapter = Wan22CloudI2VAdapter(
            submit_to_runpod_fn=mock_submit,
            comfyui_client_fn=lambda: mock_comfyui,
        )
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="test motion",
            input_images=["base64data"],
            frames=81,
        )
        result = await adapter.execute(req)
        assert result.status == "queued_cloud"
        assert result.adapter_name == "wan22-cloud-i2v"
        assert result.runpod_job_id == "rp-123"
        mock_submit.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_no_image_raises(self):
        adapter = Wan22CloudI2VAdapter(submit_to_runpod_fn=AsyncMock())
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="test",
        )
        with pytest.raises(ValueError, match="requires an input image"):
            await adapter.execute(req)

    def test_to_dict(self):
        adapter = Wan22CloudI2VAdapter()
        d = adapter.to_dict()
        assert d["name"] == "wan22-cloud-i2v"
        assert d["compute"] == "cloud"
        assert d["lora_format"] == "dual"


# ── Wan22 Cloud T2V ─────────────────────────────────────────────────


class TestWan22CloudT2V:
    def test_metadata(self):
        adapter = Wan22CloudT2VAdapter()
        assert adapter.name == "wan22-cloud-t2v"
        assert adapter.model_family == "wan2.2"
        assert MediaType.TEXT in adapter.input_types
        assert adapter.output_type == MediaType.VIDEO
        assert adapter.compute == ComputeTarget.CLOUD
        assert adapter.lora_format == LoraFormat.DUAL_STAGE

    def test_constraints(self):
        adapter = Wan22CloudT2VAdapter()
        c = adapter.constraints()
        assert c.max_frames == 161
        assert c.default_steps == 15
        assert "dpmpp_2m" in c.supported_samplers

    @pytest.mark.parametrize("frames,expected_cost", [
        (81, 16),    # 8 base × 2 cloud
        (161, 24),   # 12 base × 2 cloud
    ])
    def test_cost(self, frames, expected_cost):
        adapter = Wan22CloudT2VAdapter()
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="test",
            frames=frames,
        )
        assert adapter.cost(req) == expected_cost

    def test_build_workflow_delegates(self):
        mock_comfyui = MagicMock()
        mock_comfyui.build_cloud_wan22_t2v_workflow.return_value = {"t2v": True}

        adapter = Wan22CloudT2VAdapter(comfyui_client_fn=lambda: mock_comfyui)
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="a dancing robot",
            frames=81,
        )
        result = adapter.build_workflow(req)
        assert result == {"t2v": True}
        mock_comfyui.build_cloud_wan22_t2v_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_success(self):
        mock_submit = AsyncMock(return_value={
            "prompt_id": "t2v-id",
            "runpod_job_id": "rp-t2v",
        })
        mock_comfyui = MagicMock()
        mock_comfyui.build_cloud_wan22_t2v_workflow.return_value = {"t2v": True}

        adapter = Wan22CloudT2VAdapter(
            submit_to_runpod_fn=mock_submit,
            comfyui_client_fn=lambda: mock_comfyui,
        )
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="test",
            frames=81,
        )
        result = await adapter.execute(req)
        assert result.status == "queued_cloud"
        assert result.adapter_name == "wan22-cloud-t2v"

    @pytest.mark.asyncio
    async def test_execute_budget_exceeded_raises(self):
        mock_comfyui = MagicMock()
        mock_comfyui.build_cloud_wan22_t2v_workflow.return_value = {"t2v": True}

        adapter = Wan22CloudT2VAdapter(
            submit_to_runpod_fn=AsyncMock(),
            comfyui_client_fn=lambda: mock_comfyui,
        )
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="test",
            width=1920,
            height=1080,
            frames=161,  # 1920*1080*161 = ~333M > 100M
        )
        with pytest.raises(ValueError, match="exceeds safety budget"):
            await adapter.execute(req)


# ── LTX-2.3 Cloud I2V ──────────────────────────────────────────────


class TestLTX23CloudI2V:
    def test_metadata(self):
        adapter = LTX23CloudI2VAdapter()
        assert adapter.name == "ltx23-cloud-i2v"
        assert adapter.model_family == "ltx"
        assert MediaType.IMAGE in adapter.input_types
        assert adapter.output_type == MediaType.VIDEO
        assert adapter.compute == ComputeTarget.CLOUD
        assert adapter.lora_format == LoraFormat.SINGLE_STAGE

    def test_constraints(self):
        adapter = LTX23CloudI2VAdapter()
        c = adapter.constraints()
        assert c.resolution_step == 32  # LTX requires /32
        assert c.max_frames == 257
        assert 25 in c.allowed_fps
        assert c.default_steps == 20

    @pytest.mark.parametrize("frames,expected_cost", [
        (97, 5),
        (161, 8),
        (257, 15),
    ])
    def test_cost(self, frames, expected_cost):
        adapter = LTX23CloudI2VAdapter()
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="test",
            frames=frames,
        )
        assert adapter.cost(req) == expected_cost

    def test_build_workflow_delegates(self):
        mock_comfyui = MagicMock()
        mock_comfyui.build_cloud_ltx23_i2v_workflow.return_value = {"ltx_i2v": True}

        adapter = LTX23CloudI2VAdapter(comfyui_client_fn=lambda: mock_comfyui)
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="test",
            input_images=["img.png"],
            frames=97,
        )
        result = adapter.build_workflow(req)
        assert result == {"ltx_i2v": True}
        mock_comfyui.build_cloud_ltx23_i2v_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_success(self):
        mock_submit = AsyncMock(return_value={
            "prompt_id": "ltx-id",
            "runpod_job_id": "rp-ltx",
        })
        mock_comfyui = MagicMock()
        mock_comfyui.build_cloud_ltx23_i2v_workflow.return_value = {"ltx_i2v": True}

        adapter = LTX23CloudI2VAdapter(
            submit_to_runpod_fn=mock_submit,
            comfyui_client_fn=lambda: mock_comfyui,
        )
        with patch.dict(os.environ, {"RUNPOD_LTX23_ENDPOINT_ID": "test-endpoint"}):
            req = GenerationRequest(
                operation=Operation.GENERATE,
                target_type=MediaType.VIDEO,
                prompt="test",
                input_images=["base64data"],
                frames=97,
            )
            result = await adapter.execute(req)
            assert result.status == "queued_cloud"
            assert result.adapter_name == "ltx23-cloud-i2v"

    @pytest.mark.asyncio
    async def test_execute_no_endpoint_raises(self):
        adapter = LTX23CloudI2VAdapter(submit_to_runpod_fn=AsyncMock())
        with patch.dict(os.environ, {}, clear=True):
            # Ensure the env var is not set
            os.environ.pop("RUNPOD_LTX23_ENDPOINT_ID", None)
            req = GenerationRequest(
                operation=Operation.GENERATE,
                target_type=MediaType.VIDEO,
                prompt="test",
                input_images=["base64data"],
            )
            with pytest.raises(RuntimeError, match="RUNPOD_LTX23_ENDPOINT_ID"):
                await adapter.execute(req)

    @pytest.mark.asyncio
    async def test_execute_no_image_raises(self):
        adapter = LTX23CloudI2VAdapter(submit_to_runpod_fn=AsyncMock())
        with patch.dict(os.environ, {"RUNPOD_LTX23_ENDPOINT_ID": "test"}):
            req = GenerationRequest(
                operation=Operation.GENERATE,
                target_type=MediaType.VIDEO,
                prompt="test",
            )
            with pytest.raises(ValueError, match="requires an input image"):
                await adapter.execute(req)


# ── LTX-2.3 Cloud T2V ──────────────────────────────────────────────


class TestLTX23CloudT2V:
    def test_metadata(self):
        adapter = LTX23CloudT2VAdapter()
        assert adapter.name == "ltx23-cloud-t2v"
        assert adapter.model_family == "ltx"
        assert MediaType.TEXT in adapter.input_types
        assert adapter.output_type == MediaType.VIDEO
        assert adapter.compute == ComputeTarget.CLOUD
        assert adapter.lora_format == LoraFormat.SINGLE_STAGE

    def test_constraints(self):
        adapter = LTX23CloudT2VAdapter()
        c = adapter.constraints()
        assert c.resolution_step == 32
        assert "21:9" in c.aspect_ratios

    @pytest.mark.parametrize("frames,expected_cost", [
        (97, 8),
        (161, 12),
        (257, 15),
    ])
    def test_cost(self, frames, expected_cost):
        adapter = LTX23CloudT2VAdapter()
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="test",
            frames=frames,
        )
        assert adapter.cost(req) == expected_cost

    def test_build_workflow_delegates(self):
        mock_comfyui = MagicMock()
        mock_comfyui.build_cloud_ltx23_t2v_workflow.return_value = {"ltx_t2v": True}

        adapter = LTX23CloudT2VAdapter(comfyui_client_fn=lambda: mock_comfyui)
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="ocean waves crashing",
            frames=97,
        )
        result = adapter.build_workflow(req)
        assert result == {"ltx_t2v": True}
        mock_comfyui.build_cloud_ltx23_t2v_workflow.assert_called_once()

    def test_build_workflow_with_audio_prompt(self):
        mock_comfyui = MagicMock()
        mock_comfyui.build_cloud_ltx23_t2v_workflow.return_value = {"ltx_t2v": True}

        adapter = LTX23CloudT2VAdapter(comfyui_client_fn=lambda: mock_comfyui)
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="ocean waves",
            audio_prompt="waves crashing on beach",
            frames=97,
        )
        adapter.build_workflow(req)
        call_kwargs = mock_comfyui.build_cloud_ltx23_t2v_workflow.call_args[1]
        assert call_kwargs["audio_prompt"] == "waves crashing on beach"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        mock_submit = AsyncMock(return_value={
            "prompt_id": "ltx-t2v-id",
            "runpod_job_id": "rp-ltx-t2v",
        })
        mock_comfyui = MagicMock()
        mock_comfyui.build_cloud_ltx23_t2v_workflow.return_value = {"ltx_t2v": True}

        adapter = LTX23CloudT2VAdapter(
            submit_to_runpod_fn=mock_submit,
            comfyui_client_fn=lambda: mock_comfyui,
        )
        with patch.dict(os.environ, {"RUNPOD_LTX23_ENDPOINT_ID": "test-endpoint"}):
            req = GenerationRequest(
                operation=Operation.GENERATE,
                target_type=MediaType.VIDEO,
                prompt="ocean waves",
                frames=97,
            )
            result = await adapter.execute(req)
            assert result.status == "queued_cloud"
            assert result.adapter_name == "ltx23-cloud-t2v"

    @pytest.mark.asyncio
    async def test_execute_no_endpoint_raises(self):
        adapter = LTX23CloudT2VAdapter(submit_to_runpod_fn=AsyncMock())
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("RUNPOD_LTX23_ENDPOINT_ID", None)
            req = GenerationRequest(
                operation=Operation.GENERATE,
                target_type=MediaType.VIDEO,
                prompt="test",
            )
            with pytest.raises(RuntimeError, match="RUNPOD_LTX23_ENDPOINT_ID"):
                await adapter.execute(req)


# ── Registry integration ────────────────────────────────────────────


class TestCloudAdaptersRegistry:
    def test_all_cloud_adapters_register(self):
        from generation.registry import AdapterRegistry

        registry = AdapterRegistry()
        registry.register(Wan22CloudI2VAdapter())
        registry.register(Wan22CloudT2VAdapter())
        registry.register(LTX23CloudI2VAdapter())
        registry.register(LTX23CloudT2VAdapter())

        assert len(registry) == 4
        assert "wan22-cloud-i2v" in registry
        assert "wan22-cloud-t2v" in registry
        assert "ltx23-cloud-i2v" in registry
        assert "ltx23-cloud-t2v" in registry

    def test_find_cloud_i2v_adapters(self):
        from generation.registry import AdapterRegistry

        registry = AdapterRegistry()
        registry.register(Wan22CloudI2VAdapter())
        registry.register(LTX23CloudI2VAdapter())

        results = registry.find(
            operation=Operation.GENERATE,
            input_type=MediaType.IMAGE,
            target_type=MediaType.VIDEO,
            compute=ComputeTarget.CLOUD,
        )
        assert len(results) == 2

    def test_find_cloud_t2v_adapters(self):
        from generation.registry import AdapterRegistry

        registry = AdapterRegistry()
        registry.register(Wan22CloudT2VAdapter())
        registry.register(LTX23CloudT2VAdapter())

        results = registry.find(
            operation=Operation.GENERATE,
            input_type=MediaType.TEXT,
            target_type=MediaType.VIDEO,
            compute=ComputeTarget.CLOUD,
        )
        assert len(results) == 2
