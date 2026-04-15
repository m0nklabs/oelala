"""
Tests for Phase 5 local video adapters: Wan22 I2V variants + T2V.

Tests cover:
- All 4 I2V variants (Q6, DisTorch2, BlockSwap, Ultra) via shared base
- T2V Q6
- Metadata, constraints, cost, workflow delegation, execute
"""

import os
import sys

import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "backend"))

from generation.types import (
    ComputeTarget,
    GenerationRequest,
    LoraFormat,
    LoraStackItem,
    MediaType,
    Operation,
)
from generation.adapters.local.i2v_wan22 import (
    Wan22LocalI2VQ6Adapter,
    Wan22LocalI2VDisTorch2Adapter,
    Wan22LocalI2VBlockSwapAdapter,
    Wan22LocalI2VUltraAdapter,
)
from generation.adapters.local.t2v_wan22 import Wan22LocalT2VQ6Adapter


# ── Wan22 I2V Q6 ───────────────────────────────────────────────────


class TestWan22I2VQ6:
    def test_metadata(self):
        a = Wan22LocalI2VQ6Adapter()
        assert a.name == "wan22-local-i2v-q6"
        assert a.model_family == "wan2.2"
        assert Operation.GENERATE in a.supported_ops
        assert MediaType.IMAGE in a.input_types
        assert a.output_type == MediaType.VIDEO
        assert a.compute == ComputeTarget.LOCAL
        assert a.lora_format == LoraFormat.DUAL_STAGE

    def test_constraints(self):
        c = Wan22LocalI2VQ6Adapter().constraints()
        assert c.max_frames == 321
        assert c.default_steps == 6
        assert c.default_cfg == 1.0
        assert "480p" in c.resolution_presets

    @pytest.mark.parametrize("frames,expected", [
        (81, 5), (161, 8), (321, 15),
    ])
    def test_cost(self, frames, expected):
        a = Wan22LocalI2VQ6Adapter()
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="test",
            frames=frames,
        )
        assert a.cost(req) == expected

    def test_build_workflow_delegates(self):
        mock = MagicMock()
        mock.build_q6_workflow.return_value = {"q6": True}

        a = Wan22LocalI2VQ6Adapter(comfyui_client_fn=lambda: mock)
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="cat walking",
            input_images=["img.png"],
            frames=81,
        )
        result = a.build_workflow(req)
        assert result == {"q6": True}
        mock.build_q6_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_success(self):
        mock = MagicMock()
        mock.build_q6_workflow.return_value = {"q6": True}
        mock.queue_prompt.return_value = "q6-prompt-id"

        a = Wan22LocalI2VQ6Adapter(comfyui_client_fn=lambda: mock)
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="test",
            input_images=["img.png"],
            frames=81,
        )
        result = await a.execute(req)
        assert result.status == "queued_local"
        assert result.prompt_id == "q6-prompt-id"
        assert result.adapter_name == "wan22-local-i2v-q6"

    @pytest.mark.asyncio
    async def test_execute_no_image_raises(self):
        mock = MagicMock()
        a = Wan22LocalI2VQ6Adapter(comfyui_client_fn=lambda: mock)
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="test",
        )
        with pytest.raises(ValueError, match="requires an input image"):
            await a.execute(req)


# ── Wan22 I2V DisTorch2 ────────────────────────────────────────────


class TestWan22I2VDisTorch2:
    def test_metadata(self):
        a = Wan22LocalI2VDisTorch2Adapter()
        assert a.name == "wan22-local-i2v-distorch2"
        assert a.model_family == "wan2.2"

    def test_quant_config(self):
        a = Wan22LocalI2VDisTorch2Adapter()
        qc = a._get_quant_config()
        assert qc.builder_method == "build_distorch2_q8_workflow"

    def test_build_workflow_delegates(self):
        mock = MagicMock()
        mock.build_distorch2_q8_workflow.return_value = {"distorch2": True}

        a = Wan22LocalI2VDisTorch2Adapter(comfyui_client_fn=lambda: mock)
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="test",
            input_images=["img.png"],
        )
        result = a.build_workflow(req)
        assert result == {"distorch2": True}


# ── Wan22 I2V BlockSwap ────────────────────────────────────────────


class TestWan22I2VBlockSwap:
    def test_metadata(self):
        a = Wan22LocalI2VBlockSwapAdapter()
        assert a.name == "wan22-local-i2v-blockswap"

    def test_constraints_lower_max_frames(self):
        c = Wan22LocalI2VBlockSwapAdapter().constraints()
        assert c.max_frames == 161  # BlockSwap is more conservative

    def test_build_workflow_delegates(self):
        mock = MagicMock()
        mock.build_blockswap_q8_workflow.return_value = {"blockswap": True}

        a = Wan22LocalI2VBlockSwapAdapter(comfyui_client_fn=lambda: mock)
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="test",
            input_images=["img.png"],
        )
        result = a.build_workflow(req)
        assert result == {"blockswap": True}


# ── Wan22 I2V Ultra ────────────────────────────────────────────────


class TestWan22I2VUltra:
    def test_metadata(self):
        a = Wan22LocalI2VUltraAdapter()
        assert a.name == "wan22-local-i2v-ultra"

    def test_build_workflow_delegates(self):
        mock = MagicMock()
        mock.build_ultra_q8_workflow.return_value = {"ultra": True}

        a = Wan22LocalI2VUltraAdapter(comfyui_client_fn=lambda: mock)
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="test",
            input_images=["img.png"],
        )
        result = a.build_workflow(req)
        assert result == {"ultra": True}


# ── Wan22 T2V Q6 ───────────────────────────────────────────────────


class TestWan22T2VQ6:
    def test_metadata(self):
        a = Wan22LocalT2VQ6Adapter()
        assert a.name == "wan22-local-t2v-q6"
        assert a.model_family == "wan2.2"
        assert MediaType.TEXT in a.input_types
        assert a.output_type == MediaType.VIDEO
        assert a.compute == ComputeTarget.LOCAL

    def test_constraints(self):
        c = Wan22LocalT2VQ6Adapter().constraints()
        assert c.max_frames == 321
        assert c.default_steps == 6
        assert 16 in c.allowed_fps

    @pytest.mark.parametrize("frames,expected", [
        (81, 8), (161, 12), (321, 15),
    ])
    def test_cost(self, frames, expected):
        a = Wan22LocalT2VQ6Adapter()
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="test",
            frames=frames,
        )
        assert a.cost(req) == expected

    def test_build_workflow_delegates(self):
        mock = MagicMock()
        mock.build_t2v_q6_workflow.return_value = {"t2v": True}

        a = Wan22LocalT2VQ6Adapter(comfyui_client_fn=lambda: mock)
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="dancing robot",
            frames=81,
        )
        result = a.build_workflow(req)
        assert result == {"t2v": True}
        mock.build_t2v_q6_workflow.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_success(self):
        mock = MagicMock()
        mock.build_t2v_q6_workflow.return_value = {"t2v": True}
        mock.queue_prompt.return_value = "t2v-prompt-id"

        a = Wan22LocalT2VQ6Adapter(comfyui_client_fn=lambda: mock)
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="test",
            frames=81,
        )
        result = await a.execute(req)
        assert result.status == "queued_local"
        assert result.adapter_name == "wan22-local-t2v-q6"


# ── Registry integration ────────────────────────────────────────────


class TestLocalVideoRegistry:
    def test_all_video_adapters_register(self):
        from generation.registry import AdapterRegistry

        registry = AdapterRegistry()
        for cls in [
            Wan22LocalI2VQ6Adapter,
            Wan22LocalI2VDisTorch2Adapter,
            Wan22LocalI2VBlockSwapAdapter,
            Wan22LocalI2VUltraAdapter,
            Wan22LocalT2VQ6Adapter,
        ]:
            registry.register(cls())

        assert len(registry) == 5

    def test_find_local_i2v(self):
        from generation.registry import AdapterRegistry

        registry = AdapterRegistry()
        for cls in [
            Wan22LocalI2VQ6Adapter,
            Wan22LocalI2VDisTorch2Adapter,
            Wan22LocalI2VBlockSwapAdapter,
            Wan22LocalI2VUltraAdapter,
        ]:
            registry.register(cls())

        results = registry.find(
            operation=Operation.GENERATE,
            input_type=MediaType.IMAGE,
            target_type=MediaType.VIDEO,
            compute=ComputeTarget.LOCAL,
        )
        assert len(results) == 4

    def test_find_local_t2v(self):
        from generation.registry import AdapterRegistry

        registry = AdapterRegistry()
        registry.register(Wan22LocalT2VQ6Adapter())

        results = registry.find(
            operation=Operation.GENERATE,
            input_type=MediaType.TEXT,
            target_type=MediaType.VIDEO,
            compute=ComputeTarget.LOCAL,
        )
        assert len(results) == 1
