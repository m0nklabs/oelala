"""
Tests for local MiniMax-H3 adapters (Windows PC ComfyUI).

These adapters route MiniMax-H3 generation to the user's Windows PC ComfyUI
server (get_windows_comfyui_client) instead of the default ai-kvm2 ComfyUI.

Covers: metadata, constraints, cost, workflow delegation, execute (t2v + i2v,
including the adapter-level image upload for i2v).
"""

import os
import sys
import base64

import pytest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "backend"))

from generation.types import (
    ComputeTarget,
    GenerationRequest,
    LoraFormat,
    MediaType,
    Operation,
)
from generation.adapters.local.minimax_h3_t2v import MiniMaxH3LocalT2VAdapter
from generation.adapters.local.minimax_h3_i2v import MiniMaxH3LocalI2VAdapter


def _png_b64() -> str:
    # 1x1 transparent PNG (valid, decodes fine)
    return base64.b64encode(
        bytes.fromhex(
            "89504e470d0a1a0a0000000d494844520000000100000001080600000"
            "01f15c4890000000d49444154789c626001000000ffff030000060005"
            "57bfabd40000000049454e44ae426082"
        )
    ).decode()


def _req(**kw):
    base = {"operation": Operation.GENERATE, "target_type": MediaType.VIDEO, "prompt": "test"}
    base.update(kw)
    return GenerationRequest(**base)


class TestMiniMaxH3LocalT2V:
    def test_metadata(self):
        a = MiniMaxH3LocalT2VAdapter()
        assert a.name == "minimax-h3-local-t2v"
        assert a.model_family == "minimax_h3"
        assert Operation.GENERATE in a.supported_ops
        assert MediaType.TEXT in a.input_types
        assert a.output_type == MediaType.VIDEO
        assert a.compute == ComputeTarget.LOCAL
        assert a.lora_format == LoraFormat.SINGLE_STAGE

    def test_constraints(self):
        c = MiniMaxH3LocalT2VAdapter().constraints()
        assert c.max_frames == 362
        assert c.default_steps == 20
        assert c.default_cfg == 1.0
        assert c.allowed_fps == [24]
        assert c.supports_negative_prompt is False
        assert "16:9" in c.aspect_ratios

    @pytest.mark.parametrize("frames,expected", [
        (124, 8), (210, 12), (362, 15),
    ])
    def test_cost(self, frames, expected):
        assert MiniMaxH3LocalT2VAdapter().cost(_req(frames=frames)) == expected

    def test_build_workflow_delegates_to_local_builder(self):
        mock = MagicMock()
        mock.build_local_minimax_h3_t2v_workflow.return_value = {"h3": "wf"}
        a = MiniMaxH3LocalT2VAdapter(comfyui_client_fn=lambda: mock)
        req = _req(frames=124, fps=24, seed=5, steps=20, aspect_ratio="16:9", megapixels=0.98)
        wf = a.build_workflow(req)
        assert wf == {"h3": "wf"}
        mock.build_local_minimax_h3_t2v_workflow.assert_called_once()
        kwargs = mock.build_local_minimax_h3_t2v_workflow.call_args[1]
        assert kwargs["prompt"] == "test"
        assert kwargs["num_frames"] == 124
        assert kwargs["aspect_ratio"] == "16:9"
        assert kwargs["megapixels"] == 0.98

    @pytest.mark.asyncio
    async def test_execute_queues_to_client(self):
        mock = MagicMock()
        mock.is_available.return_value = True
        mock.queue_prompt.return_value = "p123"
        mock.host = "192.168.1.245"
        mock.port = 8188
        a = MiniMaxH3LocalT2VAdapter(comfyui_client_fn=lambda: mock)
        req = _req(frames=124)
        res = await a.execute(req)
        assert res.status == "queued_local"
        assert res.compute_target == ComputeTarget.LOCAL
        assert res.prompt_id == "p123"
        mock.queue_prompt.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_raises_when_server_down(self):
        mock = MagicMock()
        mock.is_available.return_value = False
        a = MiniMaxH3LocalT2VAdapter(comfyui_client_fn=lambda: mock)
        with pytest.raises(RuntimeError):
            await a.execute(_req())


class TestMiniMaxH3LocalI2V:
    def test_metadata(self):
        a = MiniMaxH3LocalI2VAdapter()
        assert a.name == "minimax-h3-local-i2v"
        assert a.compute == ComputeTarget.LOCAL
        assert MediaType.IMAGE in a.input_types
        assert a.output_type == MediaType.VIDEO
        assert a.handles_own_image_upload is True

    def test_constraints(self):
        c = MiniMaxH3LocalI2VAdapter().constraints()
        assert c.max_input_images == 1
        assert c.allowed_fps == [24]
        assert c.supports_negative_prompt is False

    @pytest.mark.parametrize("frames,expected", [
        (124, 5), (210, 8), (362, 15),
    ])
    def test_cost(self, frames, expected):
        assert MiniMaxH3LocalI2VAdapter().cost(_req(frames=frames)) == expected

    @pytest.mark.asyncio
    async def test_execute_uploads_to_windows_client(self):
        mock = MagicMock()
        mock.is_available.return_value = True
        mock.upload_image_from_bytes.return_value = "v2_minimax_i2v_input.png"
        mock.queue_prompt.return_value = "p456"
        mock.host = "192.168.1.245"
        mock.port = 8188
        a = MiniMaxH3LocalI2VAdapter(comfyui_client_fn=lambda: mock)
        req = _req(input_images=[_png_b64()], frames=124)
        res = await a.execute(req)
        assert res.status == "queued_local"
        assert res.prompt_id == "p456"
        # image must be uploaded to the windows client itself
        mock.upload_image_from_bytes.assert_called_once()
        # and the workflow built with the returned filename
        mock.build_local_minimax_h3_i2v_workflow.assert_called_once()
        assert mock.build_local_minimax_h3_i2v_workflow.call_args[1]["image_name"] == "v2_minimax_i2v_input.png"

    @pytest.mark.asyncio
    async def test_execute_requires_image(self):
        a = MiniMaxH3LocalI2VAdapter(comfyui_client_fn=lambda: MagicMock())
        with pytest.raises(ValueError):
            await a.execute(_req())

    @pytest.mark.asyncio
    async def test_execute_raises_when_upload_fails(self):
        mock = MagicMock()
        mock.is_available.return_value = True
        mock.upload_image_from_bytes.return_value = None
        a = MiniMaxH3LocalI2VAdapter(comfyui_client_fn=lambda: mock)
        req = _req(input_images=[_png_b64()])
        with pytest.raises(RuntimeError):
            await a.execute(req)
