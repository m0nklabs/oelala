"""
Tests for local T2I adapters: SDXL-Pony and Flux.

Tests cover:
- Adapter metadata (name, model_family, ops, types, compute, LoRA format)
- Constraints
- Cost calculations
- Build workflow structure (LoRA slots, nodes)
- Execute with mocked ComfyUI client
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
from generation.adapters.local.t2i_sdxl import SDXLLocalT2IAdapter
from generation.adapters.local.t2i_flux import FluxLocalT2IAdapter
from generation.adapters.local.t2i_krea2 import Krea2LocalT2IAdapter
from generation.adapters.local.t2i_flux2 import Flux2LocalT2IAdapter


# ── SDXL ────────────────────────────────────────────────────────────


class TestSDXLLocalT2I:
    def test_metadata(self):
        a = SDXLLocalT2IAdapter()
        assert a.name == "sdxl-local-t2i"
        assert a.model_family == "sdxl"
        assert Operation.GENERATE in a.supported_ops
        assert MediaType.TEXT in a.input_types
        assert a.output_type == MediaType.IMAGE
        assert a.compute == ComputeTarget.LOCAL
        assert a.lora_format == LoraFormat.SINGLE_STAGE

    def test_constraints(self):
        c = SDXLLocalT2IAdapter().constraints()
        assert c.max_loras == 3
        assert c.default_steps == 30
        assert c.default_cfg == 7.5
        assert "1:1" in c.aspect_ratios
        assert c.supports_negative_prompt is True

    def test_cost_standard(self):
        a = SDXLLocalT2IAdapter()
        req = GenerationRequest(
            operation=Operation.GENERATE, target_type=MediaType.IMAGE, prompt="t"
        )
        assert a.cost(req) == 1

    def test_cost_hd(self):
        a = SDXLLocalT2IAdapter()
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            prompt="t",
            width=1344,
            height=768,
        )
        # 1344*768 = 1,032,192 < 1,048,576 (1024*1024) → standard cost
        assert a.cost(req) == 1

    def test_build_workflow_basic(self):
        a = SDXLLocalT2IAdapter()
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            prompt="a cat",
            negative_prompt="bad",
            seed=42,
            steps=30,
            cfg=7.5,
            aspect_ratio="1:1",
        )
        wf = a.build_workflow(req)
        assert "1" in wf  # CheckpointLoader
        assert "5" in wf  # KSampler
        assert "9" in wf  # Power LoRA Loader
        assert wf["5"]["inputs"]["seed"] == 42
        assert wf["5"]["inputs"]["steps"] == 30
        assert wf["2"]["inputs"]["text"] == "a cat"

    def test_build_workflow_with_loras(self):
        a = SDXLLocalT2IAdapter()
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            prompt="test",
            loras=[
                LoraStackItem(name="detail.safetensors", strength=0.8),
                LoraStackItem(name="style.safetensors", strength=1.0),
            ],
        )
        wf = a.build_workflow(req)
        assert wf["9"]["inputs"]["lora_1"]["on"] is True
        assert wf["9"]["inputs"]["lora_1"]["lora"] == "detail.safetensors"
        assert wf["9"]["inputs"]["lora_2"]["on"] is True

    @pytest.mark.asyncio
    async def test_execute_success(self):
        mock = MagicMock()
        mock.queue_prompt.return_value = "sdxl-prompt-id"

        a = SDXLLocalT2IAdapter(comfyui_client_fn=lambda: mock)
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            prompt="test",
        )
        result = await a.execute(req)
        assert result.status == "queued_local"
        assert result.prompt_id == "sdxl-prompt-id"
        assert result.adapter_name == "sdxl-local-t2i"
        mock.queue_prompt.assert_called_once()


# ── Flux ────────────────────────────────────────────────────────────


class TestFluxLocalT2I:
    def test_metadata(self):
        a = FluxLocalT2IAdapter()
        assert a.name == "flux-local-t2i"
        assert a.model_family == "flux"
        assert a.lora_format == LoraFormat.SINGLE_STAGE
        assert a.compute == ComputeTarget.LOCAL

    def test_constraints(self):
        c = FluxLocalT2IAdapter().constraints()
        assert c.max_loras == 4
        assert c.default_steps == 20
        assert c.default_cfg == 3.5
        assert c.supports_negative_prompt is False

    def test_cost(self):
        a = FluxLocalT2IAdapter()
        req = GenerationRequest(
            operation=Operation.GENERATE, target_type=MediaType.IMAGE, prompt="t"
        )
        assert a.cost(req) == 2

    def test_cost_hd(self):
        a = FluxLocalT2IAdapter()
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            prompt="t",
            width=2048,
            height=2048,
        )
        assert a.cost(req) == 3

    def test_build_workflow_no_negative(self):
        a = FluxLocalT2IAdapter()
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            prompt="a cat",
        )
        wf = a.build_workflow(req)
        # Flux uses FluxGuidance, not CLIPTextEncode for negative
        assert "12" in wf  # FluxGuidance
        assert wf["3"]["inputs"]["text"] == "a cat"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        mock = MagicMock()
        mock.queue_prompt.return_value = "flux-prompt-id"

        a = FluxLocalT2IAdapter(comfyui_client_fn=lambda: mock)
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            prompt="test",
        )
        result = await a.execute(req)
        assert result.status == "queued_local"
        assert result.adapter_name == "flux-local-t2i"


# ── Krea 2 ─────────────────────────────────────────────────────────


class TestKrea2LocalT2I:
    def test_metadata(self):
        a = Krea2LocalT2IAdapter()
        assert a.name == "krea2-local-t2i"
        assert a.model_family == "krea2"
        assert a.compute == ComputeTarget.LOCAL

    def test_constraints(self):
        c = Krea2LocalT2IAdapter().constraints()
        assert c.default_steps == 8
        assert c.default_cfg == 1.0
        assert c.supports_negative_prompt is False
        assert "euler" in c.supported_samplers

    def test_cost(self):
        a = Krea2LocalT2IAdapter()
        req = GenerationRequest(
            operation=Operation.GENERATE, target_type=MediaType.IMAGE, prompt="t"
        )
        assert a.cost(req) == 1

    def test_build_workflow(self):
        a = Krea2LocalT2IAdapter()
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            prompt="a cat",
        )
        wf = a.build_workflow(req)
        # CLIPLoader type must be krea2, UNETLoader krea2_turbo_int8_convrot
        assert wf["2"]["inputs"]["type"] == "krea2"
        assert wf["2"]["inputs"]["clip_name"] == "qwen3vl_4b_bf16.safetensors"
        assert wf["1"]["inputs"]["unet_name"] == "krea2_turbo_int8_convrot.safetensors"
        assert wf["5"]["inputs"]["cfg"] == 1.0
        assert wf["5"]["inputs"]["steps"] == 8

    @pytest.mark.asyncio
    async def test_execute_success(self):
        mock = MagicMock()
        mock.queue_prompt.return_value = "krea2-prompt-id"

        a = Krea2LocalT2IAdapter(comfyui_client_fn=lambda: mock)
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            prompt="test",
        )
        result = await a.execute(req)
        assert result.status == "queued_local"
        assert result.adapter_name == "krea2-local-t2i"


# ── Registry integration ────────────────────────────────────────────


class TestFlux2LocalT2I:
    def test_metadata(self):
        a = Flux2LocalT2IAdapter()
        assert a.name == "flux2-local-t2i"
        assert a.model_family == "flux2"
        assert a.compute == ComputeTarget.LOCAL

    def test_constraints(self):
        c = Flux2LocalT2IAdapter().constraints()
        assert c.default_steps == 20
        assert c.default_cfg == 4.0
        assert c.supports_negative_prompt is False
        assert "euler" in c.supported_samplers

    def test_cost(self):
        a = Flux2LocalT2IAdapter()
        req = GenerationRequest(
            operation=Operation.GENERATE, target_type=MediaType.IMAGE, prompt="t"
        )
        assert a.cost(req) == 3
        req_hd = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            prompt="t",
            width=1536,
            height=1024,
        )
        assert a.cost(req_hd) == 4

    def test_build_workflow(self):
        a = Flux2LocalT2IAdapter()
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            prompt="a cat",
        )
        wf = a.build_workflow(req)
        # UnetLoaderGGUFDisTorch2MultiGPU with flux2 GGUF + multi-GPU alloc
        assert wf["1"]["class_type"] == "UnetLoaderGGUFDisTorch2MultiGPU"
        assert wf["1"]["inputs"]["unet_name"] == "flux2-dev-Q4_K_M.gguf"
        assert "expert_mode_allocations" in wf["1"]["inputs"]
        assert "cuda:1" in wf["1"]["inputs"]["expert_mode_allocations"]
        # CLIPLoader type flux2 with Mistral3 encoder, device cpu
        assert wf["2"]["inputs"]["type"] == "flux2"
        assert wf["2"]["inputs"]["clip_name"] == "mistral_3_small_flux2_fp8.safetensors"
        assert wf["2"]["inputs"]["device"] == "cpu"
        # FLUX.2-specific nodes
        assert wf["4"]["class_type"] == "EmptyFlux2LatentImage"
        assert wf["6"]["class_type"] == "Flux2Scheduler"
        assert wf["11"]["class_type"] == "FluxGuidance"
        # VAE is flux2-vae
        assert wf["5"]["inputs"]["vae_name"] == "flux2-vae.safetensors"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        mock = MagicMock()
        mock.queue_prompt.return_value = "flux2-prompt-id"

        a = Flux2LocalT2IAdapter(comfyui_client_fn=lambda: mock)
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            prompt="test",
        )
        result = await a.execute(req)
        assert result.status == "queued_local"
        assert result.adapter_name == "flux2-local-t2i"


class TestLocalT2IRegistry:
    def test_all_t2i_adapters_register(self):
        from generation.registry import AdapterRegistry

        registry = AdapterRegistry()
        registry.register(SDXLLocalT2IAdapter())
        registry.register(FluxLocalT2IAdapter())
        registry.register(Krea2LocalT2IAdapter())
        registry.register(Flux2LocalT2IAdapter())

        assert len(registry) == 4

    def test_find_local_t2i(self):
        from generation.registry import AdapterRegistry

        registry = AdapterRegistry()
        registry.register(SDXLLocalT2IAdapter())
        registry.register(FluxLocalT2IAdapter())
        registry.register(Krea2LocalT2IAdapter())
        registry.register(Flux2LocalT2IAdapter())

        results = registry.find(
            operation=Operation.GENERATE,
            input_type=MediaType.TEXT,
            target_type=MediaType.IMAGE,
            compute=ComputeTarget.LOCAL,
        )
        assert len(results) == 4
