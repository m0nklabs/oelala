"""
Tests for the Unified Generation Core — Phase 1 (Foundation) + Phase 2 (Qwen Edit PoC).

Tests cover:
- types.py: Enum values, Pydantic model serialization, defaults
- registry.py: Register, get, find, list, duplicate handling
- router.py: Adapter resolution, control validation, LoRA filtering, dispatch
- lora_utils.py: Sanitization, compatibility filtering
- adapters/cloud/qwen_edit.py: Workflow builder, cost calculation, constraints
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os

# Add src/backend to path so generation package is importable
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
from generation.adapter import GenerationAdapter
from generation.registry import AdapterRegistry
from generation.router import GenerationRouter


# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


class FakeAdapter(GenerationAdapter):
    """Minimal concrete adapter for testing."""

    name = "fake-adapter"
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
            min_width=256,
            min_height=256,
            default_steps=20,
            default_cfg=7.0,
            max_loras=3,
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        return {"fake": True}

    def cost(self, req: GenerationRequest) -> int:
        return 5

    async def execute(self, req, progress_callback=None):
        return GenerationResult(
            prompt_id="fake-prompt-id",
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=5,
            adapter_name=self.name,
        )


class FakeCloudAdapter(GenerationAdapter):
    """Minimal cloud adapter for testing."""

    name = "fake-cloud-adapter"
    model_family = "test"
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
        )

    def build_workflow(self, req: GenerationRequest) -> dict:
        return {"cloud": True}

    def cost(self, req: GenerationRequest) -> int:
        return 15

    async def execute(self, req, progress_callback=None):
        return GenerationResult(
            prompt_id="cloud-prompt-id",
            status="queued_cloud",
            compute_target=ComputeTarget.CLOUD,
            credits_used=15,
            adapter_name=self.name,
        )


@pytest.fixture
def registry():
    return AdapterRegistry()


@pytest.fixture
def fake_adapter():
    return FakeAdapter()


@pytest.fixture
def fake_cloud_adapter():
    return FakeCloudAdapter()


@pytest.fixture
def populated_registry(registry, fake_adapter, fake_cloud_adapter):
    registry.register(fake_adapter)
    registry.register(fake_cloud_adapter)
    return registry


@pytest.fixture
def router(populated_registry):
    return GenerationRouter(populated_registry)


# ═══════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════


class TestEnums:
    def test_media_type_values(self):
        assert MediaType.IMAGE == "image"
        assert MediaType.VIDEO == "video"
        assert MediaType.AUDIO == "audio"
        assert MediaType.TEXT == "text"

    def test_operation_values(self):
        assert Operation.GENERATE == "generate"
        assert Operation.TRANSFORM == "transform"
        assert Operation.EDIT == "edit"
        assert Operation.UPSCALE == "upscale"
        assert Operation.INTERPOLATE == "interpolate"
        assert Operation.SWAP == "swap"
        assert Operation.CAPTION == "caption"
        assert Operation.LIPSYNC == "lipsync"
        assert Operation.INPAINT == "inpaint"

    def test_compute_target_values(self):
        assert ComputeTarget.LOCAL == "local"
        assert ComputeTarget.CLOUD == "cloud"
        assert ComputeTarget.AUTO == "auto"

    def test_lora_format_values(self):
        assert LoraFormat.NONE == "none"
        assert LoraFormat.SINGLE_STAGE == "single"
        assert LoraFormat.DUAL_STAGE == "dual"


class TestLoraStackItem:
    def test_defaults(self):
        item = LoraStackItem(name="test.safetensors")
        assert item.name == "test.safetensors"
        assert item.strength == 1.0
        assert item.high is None
        assert item.low is None

    def test_dual_stage(self):
        item = LoraStackItem(
            name="", high="test_high.gguf", low="test_low.gguf", strength=0.8
        )
        assert item.high == "test_high.gguf"
        assert item.low == "test_low.gguf"
        assert item.strength == 0.8

    def test_serialization(self):
        item = LoraStackItem(name="lora.safetensors", strength=0.5)
        d = item.model_dump()
        assert d["name"] == "lora.safetensors"
        assert d["strength"] == 0.5


class TestAdapterConstraints:
    def test_defaults(self):
        c = AdapterConstraints()
        assert c.max_width == 2048
        assert c.max_height == 2048
        assert c.min_width == 256
        assert c.min_height == 256
        assert c.resolution_step == 16
        assert c.default_steps == 20
        assert c.default_cfg == 7.0
        assert c.max_loras == 5
        assert c.supports_lightning is False
        assert c.supports_negative_prompt is True

    def test_custom_values(self):
        c = AdapterConstraints(
            max_width=1024,
            min_steps=4,
            supports_lightning=True,
            resolution_presets=["480p", "720p"],
        )
        assert c.max_width == 1024
        assert c.min_steps == 4
        assert c.supports_lightning is True
        assert c.resolution_presets == ["480p", "720p"]


class TestGenerationRequest:
    def test_minimal_request(self):
        req = GenerationRequest(
            operation=Operation.GENERATE, target_type=MediaType.IMAGE
        )
        assert req.operation == Operation.GENERATE
        assert req.target_type == MediaType.IMAGE
        assert req.prompt == ""
        assert req.seed == -1
        assert req.loras == []
        assert req.lightning is False

    def test_full_request(self):
        req = GenerationRequest(
            operation=Operation.EDIT,
            target_type=MediaType.IMAGE,
            prompt="test prompt",
            negative_prompt="bad quality",
            seed=42,
            steps=40,
            cfg=4.0,
            width=1024,
            height=1024,
            instruction="remove background",
            lightning=True,
            loras=[LoraStackItem(name="test.safetensors", strength=0.8)],
            input_images=["base64data"],
        )
        assert req.instruction == "remove background"
        assert req.lightning is True
        assert len(req.loras) == 1
        assert len(req.input_images) == 1

    def test_serialization_roundtrip(self):
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            prompt="a cat",
            steps=30,
        )
        d = req.model_dump()
        req2 = GenerationRequest(**d)
        assert req2.operation == req.operation
        assert req2.prompt == req.prompt
        assert req2.steps == req.steps


class TestGenerationResult:
    def test_minimal_result(self):
        res = GenerationResult(
            prompt_id="abc-123",
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=5,
            adapter_name="test-adapter",
        )
        assert res.prompt_id == "abc-123"
        assert res.runpod_job_id is None
        assert res.meta == {}

    def test_cloud_result(self):
        res = GenerationResult(
            prompt_id="def-456",
            status="queued_cloud",
            compute_target=ComputeTarget.CLOUD,
            credits_used=15,
            adapter_name="qwen-cloud-edit",
            runpod_job_id="rp-job-789",
            meta={"instruction": "test"},
        )
        assert res.runpod_job_id == "rp-job-789"
        assert res.meta["instruction"] == "test"


# ═══════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════


class TestAdapterRegistry:
    def test_register_and_get(self, registry, fake_adapter):
        registry.register(fake_adapter)
        assert registry.get("fake-adapter") is fake_adapter

    def test_get_nonexistent(self, registry):
        assert registry.get("nonexistent") is None

    def test_duplicate_raises(self, registry, fake_adapter):
        registry.register(fake_adapter)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(fake_adapter)

    def test_find_by_operation(self, populated_registry):
        results = populated_registry.find(operation=Operation.GENERATE)
        assert len(results) == 1
        assert results[0].name == "fake-adapter"

    def test_find_by_target_type(self, populated_registry):
        results = populated_registry.find(target_type=MediaType.IMAGE)
        assert len(results) == 2  # both fake adapters output images

    def test_find_by_compute(self, populated_registry):
        results = populated_registry.find(compute=ComputeTarget.LOCAL)
        assert len(results) == 1
        assert results[0].name == "fake-adapter"

    def test_find_no_match(self, populated_registry):
        results = populated_registry.find(target_type=MediaType.VIDEO)
        assert len(results) == 0

    def test_find_multiple_criteria(self, populated_registry):
        results = populated_registry.find(
            operation=Operation.EDIT,
            target_type=MediaType.IMAGE,
            compute=ComputeTarget.CLOUD,
        )
        assert len(results) == 1
        assert results[0].name == "fake-cloud-adapter"

    def test_list_all(self, populated_registry):
        all_adapters = populated_registry.list_all()
        assert len(all_adapters) == 2

    def test_len(self, populated_registry):
        assert len(populated_registry) == 2

    def test_contains(self, populated_registry):
        assert "fake-adapter" in populated_registry
        assert "nonexistent" not in populated_registry


# ═══════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════


class TestRouterResolve:
    def test_resolve_by_hint(self, router):
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            adapter_hint="fake-adapter",
        )
        adapter = router.resolve_adapter(req)
        assert adapter.name == "fake-adapter"

    def test_resolve_by_hint_not_found(self, router):
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            adapter_hint="nonexistent",
        )
        with pytest.raises(ValueError, match="not found"):
            router.resolve_adapter(req)

    def test_resolve_auto_text_input(self, router):
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            prompt="test",
        )
        adapter = router.resolve_adapter(req)
        assert adapter.name == "fake-adapter"

    def test_resolve_auto_image_input(self, router):
        req = GenerationRequest(
            operation=Operation.EDIT,
            target_type=MediaType.IMAGE,
            input_images=["base64data"],
        )
        adapter = router.resolve_adapter(req)
        assert adapter.name == "fake-cloud-adapter"

    def test_resolve_no_match(self, router):
        req = GenerationRequest(
            operation=Operation.UPSCALE,
            target_type=MediaType.VIDEO,
        )
        with pytest.raises(ValueError, match="No adapter found"):
            router.resolve_adapter(req)


class TestRouterValidation:
    def test_apply_defaults(self, router):
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
        )
        adapter = router.registry.get("fake-adapter")
        validated = router.validate_controls(req, adapter)
        assert validated.steps == 20
        assert validated.cfg == 7.0

    def test_clamp_resolution(self, router):
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            width=5000,
            height=100,
        )
        adapter = router.registry.get("fake-adapter")
        validated = router.validate_controls(req, adapter)
        assert validated.width == 1024  # clamped to max
        assert validated.height == 256  # clamped to min

    def test_resolution_step_alignment(self, router):
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            width=517,
            height=519,
        )
        adapter = router.registry.get("fake-adapter")
        validated = router.validate_controls(req, adapter)
        assert validated.width % 16 == 0
        assert validated.height % 16 == 0
        assert validated.width == 512
        assert validated.height == 512

    def test_random_seed(self, router):
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            seed=-1,
        )
        adapter = router.registry.get("fake-adapter")
        validated = router.validate_controls(req, adapter)
        assert validated.seed != -1
        assert 0 <= validated.seed <= 2**32 - 1

    def test_fixed_seed_preserved(self, router):
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            seed=42,
        )
        adapter = router.registry.get("fake-adapter")
        validated = router.validate_controls(req, adapter)
        assert validated.seed == 42


class TestRouterLoraFiltering:
    def test_empty_loras_passthrough(self, router):
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
        )
        adapter = router.registry.get("fake-adapter")
        result = router.filter_loras(req, adapter)
        assert result.loras == []

    def test_max_loras_enforced(self, router):
        loras = [LoraStackItem(name=f"lora_{i}.safetensors") for i in range(10)]
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            loras=loras,
        )
        adapter = router.registry.get("fake-adapter")
        # Mock out the external lora_scanner dependency
        with patch("generation.lora_utils.filter_loras_by_model_compat", side_effect=lambda x, _: x):
            result = router.filter_loras(req, adapter)
        assert len(result.loras) == 3  # fake adapter max_loras=3


class TestRouterDispatch:
    @pytest.mark.asyncio
    async def test_dispatch_success(self, router):
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            prompt="test prompt",
            adapter_hint="fake-adapter",
        )

        mock_check = AsyncMock()
        mock_deduct = AsyncMock(return_value=True)
        mock_user = MagicMock(id="user-123")

        result = await router.dispatch(
            req,
            mock_user,
            check_credits_fn=mock_check,
            deduct_credits_fn=mock_deduct,
        )

        assert result.status == "queued_local"
        assert result.adapter_name == "fake-adapter"
        assert result.credits_used == 5
        mock_check.assert_called_once_with(mock_user, 5)
        mock_deduct.assert_called_once_with(mock_user, 5, "fake-prompt-id", "fake-adapter")

    @pytest.mark.asyncio
    async def test_dispatch_without_credit_fns(self, router):
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            adapter_hint="fake-adapter",
        )

        result = await router.dispatch(req, MagicMock(id="user-123"))
        assert result.status == "queued_local"


# ═══════════════════════════════════════════════════════════════════
# LoRA Utils
# ═══════════════════════════════════════════════════════════════════


class TestLoraUtils:
    def test_sanitize_single_stage_passthrough(self):
        from generation.lora_utils import sanitize_lora_configs_for_single_stage

        configs = [{"name": "test.safetensors", "strength": 0.8}]
        result = sanitize_lora_configs_for_single_stage(configs)
        assert result == configs

    def test_sanitize_dual_to_single(self):
        from generation.lora_utils import sanitize_lora_configs_for_single_stage

        configs = [{"high": "high.gguf", "low": "low.gguf", "strength": 0.7}]
        result = sanitize_lora_configs_for_single_stage(configs)
        assert len(result) == 1
        assert result[0]["name"] == "high.gguf"
        assert result[0]["strength"] == 0.7
        assert "high" not in result[0]
        assert "low" not in result[0]

    def test_sanitize_skip_empty(self):
        from generation.lora_utils import sanitize_lora_configs_for_single_stage

        configs = [{"strength": 0.5}]  # no name or high
        result = sanitize_lora_configs_for_single_stage(configs)
        assert len(result) == 0

    def test_filter_compat_with_mock(self):
        from generation.lora_utils import filter_loras_by_model_compat

        configs = [
            {"name": "wan22_lora.safetensors", "strength": 1.0},
            {"name": "sdxl_lora.safetensors", "strength": 1.0},
        ]

        mock_derive = MagicMock(side_effect=lambda name: (
            "wan2.2" if "wan22" in name else "sdxl"
        ))
        with patch.dict("sys.modules", {"lora_scanner": MagicMock(_derive_base_model=mock_derive)}):
            # Re-import to pick up the mocked module
            import importlib
            import generation.lora_utils as lu
            importlib.reload(lu)
            result = lu.filter_loras_by_model_compat(configs, "wan2.2")

        assert len(result) == 1
        assert result[0]["name"] == "wan22_lora.safetensors"

    def test_filter_compat_generic_passes(self):
        from generation.lora_utils import filter_loras_by_model_compat

        configs = [{"name": "generic_lora.safetensors", "strength": 1.0}]

        mock_derive = MagicMock(return_value="")  # generic/unknown
        with patch.dict("sys.modules", {"lora_scanner": MagicMock(_derive_base_model=mock_derive)}):
            import importlib
            import generation.lora_utils as lu
            importlib.reload(lu)
            result = lu.filter_loras_by_model_compat(configs, "wan2.2")

        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════
# Adapter — Qwen Edit
# ═══════════════════════════════════════════════════════════════════


class TestQwenEditWorkflowBuilder:
    def test_basic_workflow(self):
        from generation.adapters.cloud.qwen_edit import build_qwen_edit_workflow

        workflow = build_qwen_edit_workflow(
            image_filename="test.png",
            instruction="remove background",
            width=1024,
            height=1024,
            steps=40,
            cfg=4.0,
            seed=42,
        )

        # Check core nodes exist
        assert "1" in workflow  # UNET
        assert "2" in workflow  # CLIP
        assert "3" in workflow  # VAE
        assert "4" in workflow  # LoadImage
        assert "5" in workflow  # EmptySD3LatentImage
        assert "9" in workflow  # KSampler
        assert "11" in workflow  # SaveImage

        # Check image is wired
        assert workflow["4"]["inputs"]["image"] == "test.png"

        # Check prompt is wired
        assert workflow["7"]["inputs"]["prompt"] == "remove background"

        # Check sampler settings
        assert workflow["9"]["inputs"]["seed"] == 42
        assert workflow["9"]["inputs"]["steps"] == 40
        assert workflow["9"]["inputs"]["cfg"] == 4.0

    def test_lightning_mode(self):
        from generation.adapters.cloud.qwen_edit import build_qwen_edit_workflow

        workflow = build_qwen_edit_workflow(
            image_filename="test.png",
            instruction="edit",
            lightning=True,
            seed=42,
        )

        # Lightning should override steps/cfg
        assert workflow["9"]["inputs"]["steps"] == 4
        assert workflow["9"]["inputs"]["cfg"] == 1.0

        # Lightning LoRA node should exist (node 20)
        assert "20" in workflow
        assert (
            workflow["20"]["inputs"]["lora_name"]
            == "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
        )

    def test_lora_chain(self):
        from generation.adapters.cloud.qwen_edit import build_qwen_edit_workflow

        workflow = build_qwen_edit_workflow(
            image_filename="test.png",
            instruction="edit",
            seed=42,
            lora_configs=[
                {"name": "lora1.safetensors", "strength": 0.5},
                {"name": "lora2.safetensors", "strength": 0.8},
            ],
        )

        # Two LoRA nodes should be added
        assert "20" in workflow
        assert "21" in workflow
        assert workflow["20"]["inputs"]["lora_name"] == "lora1.safetensors"
        assert workflow["20"]["inputs"]["strength_model"] == 0.5
        assert workflow["21"]["inputs"]["lora_name"] == "lora2.safetensors"

        # Chain: UNET(1) → LoRA(20) → LoRA(21) → ModelSamplingAuraFlow(6)
        assert workflow["20"]["inputs"]["model"] == ["1", 0]
        assert workflow["21"]["inputs"]["model"] == ["20", 0]
        assert workflow["6"]["inputs"]["model"] == ["21", 0]

    def test_lora_plus_lightning(self):
        from generation.adapters.cloud.qwen_edit import build_qwen_edit_workflow

        workflow = build_qwen_edit_workflow(
            image_filename="test.png",
            instruction="edit",
            seed=42,
            lightning=True,
            lora_configs=[{"name": "custom.safetensors", "strength": 0.5}],
        )

        # Custom LoRA at 20, Lightning at 21
        assert "20" in workflow
        assert "21" in workflow
        assert workflow["20"]["inputs"]["lora_name"] == "custom.safetensors"
        assert (
            workflow["21"]["inputs"]["lora_name"]
            == "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
        )
        # ModelSamplingAuraFlow should point to lightning (last in chain)
        assert workflow["6"]["inputs"]["model"] == ["21", 0]


class TestQwenEditAdapter:
    def test_constraints(self):
        from generation.adapters.cloud.qwen_edit import QwenEditCloudAdapter

        adapter = QwenEditCloudAdapter()
        c = adapter.constraints()
        assert c.min_width == 512
        assert c.max_width == 2048
        assert c.default_steps == 40
        assert c.default_cfg == 4.0
        assert c.supports_lightning is True

    @pytest.mark.parametrize(
        "lightning,lora_count,expected_cost",
        [
            (False, 0, 20),  # 15 base + 5 full quality
            (True, 0, 15),  # 15 base (lightning)
            (False, 2, 24),  # 15 + 5 + 2*2
            (True, 3, 21),  # 15 + 3*2
        ],
    )
    def test_cost(self, lightning, lora_count, expected_cost):
        from generation.adapters.cloud.qwen_edit import QwenEditCloudAdapter

        adapter = QwenEditCloudAdapter()
        loras = [LoraStackItem(name=f"l{i}.safetensors") for i in range(lora_count)]
        req = GenerationRequest(
            operation=Operation.EDIT,
            target_type=MediaType.IMAGE,
            lightning=lightning,
            loras=loras,
        )
        assert adapter.cost(req) == expected_cost

    def test_to_dict(self):
        from generation.adapters.cloud.qwen_edit import QwenEditCloudAdapter

        adapter = QwenEditCloudAdapter()
        d = adapter.to_dict()
        assert d["name"] == "qwen-cloud-edit"
        assert d["model_family"] == "qwen_image_edit"
        assert d["compute"] == "cloud"
        assert d["lora_format"] == "single"
        assert "edit" in d["supported_ops"]
        assert "constraints" in d
