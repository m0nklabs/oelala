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

    def test_resolve_never_falls_back_to_cloud_when_local_backend_unavailable(self):
        class LocalUnavailableAdapter(FakeAdapter):
            name = "local-unavailable"

            def __init__(self):
                self._get_comfyui = lambda: None

        class CloudSameRouteAdapter(FakeAdapter):
            name = "cloud-same-route"
            compute = ComputeTarget.CLOUD

            async def execute(self, req, progress_callback=None):
                return GenerationResult(
                    prompt_id="cloud-fallback",
                    status="queued_cloud",
                    compute_target=ComputeTarget.CLOUD,
                    credits_used=5,
                    adapter_name=self.name,
                )

        reg = AdapterRegistry()
        reg.register(LocalUnavailableAdapter())
        reg.register(CloudSameRouteAdapter())
        r = GenerationRouter(reg)
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            prompt="fallback",
        )
        # Local-first routing is preserved even when the local backend is
        # unavailable: a local workflow is NEVER silently re-routed to (paid)
        # cloud compute. The local adapter stays selected and fails explicitly
        # on execution instead.
        adapter = r.resolve_adapter(req)
        assert adapter.name == "local-unavailable"

    def test_resolve_selects_local_adapter_without_backend(self):
        # When only an unavailable local adapter matches, resolve still returns
        # it (no "No adapter found"): the failure surfaces at execution time as
        # an explicit error, not as a silent cloud fallback.
        class LocalUnavailableAdapter(FakeAdapter):
            name = "local-unavailable-only"

            def __init__(self):
                self._get_comfyui = lambda: None

        reg = AdapterRegistry()
        reg.register(LocalUnavailableAdapter())
        r = GenerationRouter(reg)
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            prompt="fallback",
        )
        adapter = r.resolve_adapter(req)
        assert adapter.name == "local-unavailable-only"


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


# ═══════════════════════════════════════════════════════════════════
# Router Enhancements (#132)
# ═══════════════════════════════════════════════════════════════════


from generation.router import resolve_resolution, normalize_frame_count, _is_base64_image


class FakeWanAdapter(GenerationAdapter):
    """Adapter with model_family='wan2.2' for frame normalization tests."""

    name = "fake-wan-adapter"
    model_family = "wan2.2"
    supported_ops = {Operation.GENERATE}
    input_types = {MediaType.TEXT}
    output_type = MediaType.VIDEO
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.SINGLE_STAGE

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_width=1280,
            max_height=720,
            max_frames=161,
            allowed_fps=[8, 16, 24],
        )

    def build_workflow(self, req):
        return {}

    def cost(self, req):
        return 10

    async def execute(self, req, progress_callback=None):
        return GenerationResult(
            prompt_id="wan-id",
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=10,
            adapter_name=self.name,
        )


class TestResolveResolution:
    """Tests for standalone resolve_resolution() helper."""

    def test_none_resolution_returns_none(self):
        assert resolve_resolution(None, None) is None

    def test_480p_16_9(self):
        w, h = resolve_resolution("480p", "16:9")
        assert (w, h) == (848, 480)
        assert w % 8 == 0
        assert h % 8 == 0

    def test_480p_9_16_portrait(self):
        w, h = resolve_resolution("480p", "9:16")
        assert w < h  # portrait
        assert w % 8 == 0
        assert h % 8 == 0

    def test_720p_1_1_square(self):
        w, h = resolve_resolution("720p", "1:1")
        assert w == h == 720

    def test_1080p_16_9(self):
        w, h = resolve_resolution("1080p", "16:9")
        assert h == 1080
        assert w > h  # landscape
        assert w % 8 == 0

    def test_default_aspect_is_square(self):
        w, h = resolve_resolution("480p", None)
        assert w == h

    def test_unknown_resolution_defaults_480(self):
        w, h = resolve_resolution("unknown", "16:9")
        assert h == 480

    def test_unknown_aspect_defaults_square(self):
        w, h = resolve_resolution("720p", "weird:ratio")
        assert w == h == 720

    def test_all_resolutions_are_multiples_of_8(self):
        for res in ("480p", "576p", "720p", "1080p"):
            for ar in ("16:9", "9:16", "1:1", "4:3", "3:4", "21:9"):
                w, h = resolve_resolution(res, ar)
                assert w % 8 == 0, f"{res} {ar}: width {w} not multiple of 8"
                assert h % 8 == 0, f"{res} {ar}: height {h} not multiple of 8"


class TestNormalizeFrameCount:
    """Tests for normalize_frame_count() — 4k+1 snapping."""

    @pytest.mark.parametrize(
        "input_frames,expected",
        [
            (5, 5),      # already valid (k=1)
            (9, 9),      # already valid (k=2)
            (81, 81),    # already valid (k=20)
            (321, 321),  # already valid (k=80)
            (80, 81),    # round up
            (82, 81),    # round down
            (83, 81),    # round to nearest (k=20.5 → 20 via banker's rounding)
            (1, 5),      # clamp to minimum
            (2, 5),      # clamp to minimum
            (3, 5),      # clamp to minimum (k=0.5 rounds to 1)
            (4, 5),      # clamp to minimum
            (6, 5),      # round down (k=1.25 rounds to 1)
            (7, 9),      # round up (k=1.5 rounds to 2)
            (100, 101),  # (99/4=24.75, rounds to 25, 4*25+1=101)
        ],
    )
    def test_normalize(self, input_frames, expected):
        assert normalize_frame_count(input_frames) == expected

    def test_result_is_always_4k_plus_1(self):
        for f in range(1, 400):
            result = normalize_frame_count(f)
            assert (result - 1) % 4 == 0, f"normalize({f})={result} not 4k+1"
            assert result >= 5, f"normalize({f})={result} below minimum"


class TestIsBase64Image:
    """Tests for _is_base64_image() heuristic."""

    def test_short_filename_is_not_base64(self):
        assert _is_base64_image("input_image_00001.png") is False

    def test_long_string_is_base64(self):
        assert _is_base64_image("x" * 300) is True

    def test_data_uri_prefix(self):
        assert _is_base64_image("data:image/png;base64,iVBOR...") is True

    def test_jpeg_magic(self):
        assert _is_base64_image("/9j/4AAQSkZJRg==") is True

    def test_png_magic(self):
        assert _is_base64_image("iVBORw0KGgoAAAANSUhE") is True


class TestRouterResolveResolutionFields:
    """Tests for router.resolve_resolution_fields()."""

    def test_explicit_dimensions_not_overwritten(self, router):
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            width=512, height=768,
            resolution="1080p", aspect_ratio="16:9",
        )
        result = router.resolve_resolution_fields(req)
        assert result.width == 512
        assert result.height == 768

    def test_resolution_populates_dimensions(self, router):
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            resolution="480p", aspect_ratio="16:9",
        )
        result = router.resolve_resolution_fields(req)
        assert result.width == 848
        assert result.height == 480

    def test_no_resolution_leaves_none(self, router):
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
        )
        result = router.resolve_resolution_fields(req)
        assert result.width is None
        assert result.height is None


class TestRouterNormalizeFrames:
    """Tests for router.normalize_frames() with Wan2.2 adapters."""

    @pytest.fixture
    def wan_adapter(self):
        return FakeWanAdapter()

    def test_wan_adapter_normalizes(self, router, wan_adapter):
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            frames=80,
        )
        result = router.normalize_frames(req, wan_adapter)
        assert result.frames == 81

    def test_non_wan_adapter_skips(self, router, fake_adapter):
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            frames=80,
        )
        result = router.normalize_frames(req, fake_adapter)
        assert result.frames == 80  # unchanged — model_family is "test"

    def test_none_frames_skips(self, router, wan_adapter):
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
        )
        result = router.normalize_frames(req, wan_adapter)
        assert result.frames is None

    def test_validate_clamps_frames_and_fps(self, router, wan_adapter):
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            frames=999,
            fps=15,
        )
        result = router.validate_controls(req, wan_adapter)
        assert result.frames == 161
        assert result.fps == 16


class TestRouterUploadLocalImages:
    """Tests for router.upload_local_images()."""

    @pytest.mark.asyncio
    async def test_uploads_base64_for_local_adapter(self):
        mock_upload = AsyncMock(return_value="uploaded_image.png")
        registry = AdapterRegistry()
        registry.register(FakeAdapter())
        r = GenerationRouter(registry, comfyui_upload_fn=mock_upload)

        adapter = registry.get("fake-adapter")
        b64_data = "iVBORw0KGgoAAAANSUhEUgAA" + "A" * 300  # long enough

        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            input_images=[b64_data],
        )
        result = await r.upload_local_images(req, adapter)
        assert result.input_images == ["uploaded_image.png"]
        mock_upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_skips_cloud_adapter(self):
        mock_upload = AsyncMock(return_value="should_not_be_called.png")
        registry = AdapterRegistry()
        registry.register(FakeCloudAdapter())
        r = GenerationRouter(registry, comfyui_upload_fn=mock_upload)

        adapter = registry.get("fake-cloud-adapter")
        req = GenerationRequest(
            operation=Operation.EDIT,
            target_type=MediaType.IMAGE,
            input_images=["data:image/png;base64,big_data" + "A" * 300],
        )
        result = await r.upload_local_images(req, adapter)
        # Cloud adapter — no upload, images passed through
        mock_upload.assert_not_called()
        assert result.input_images == req.input_images

    @pytest.mark.asyncio
    async def test_preserves_comfyui_filenames(self):
        mock_upload = AsyncMock()
        registry = AdapterRegistry()
        registry.register(FakeAdapter())
        r = GenerationRouter(registry, comfyui_upload_fn=mock_upload)

        adapter = registry.get("fake-adapter")
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            input_images=["input_image_00001.png"],
        )
        result = await r.upload_local_images(req, adapter)
        assert result.input_images == ["input_image_00001.png"]
        mock_upload.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_upload_fn_passthrough(self):
        registry = AdapterRegistry()
        registry.register(FakeAdapter())
        r = GenerationRouter(registry)  # no comfyui_upload_fn

        adapter = registry.get("fake-adapter")
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            input_images=["data:image/png;base64," + "A" * 500],
        )
        result = await r.upload_local_images(req, adapter)
        assert result.input_images == req.input_images  # unchanged

    @pytest.mark.asyncio
    async def test_strips_data_uri_prefix(self):
        captured_calls = []

        async def capture_upload(b64_data: str, filename: str) -> str:
            captured_calls.append(b64_data)
            return "result.png"

        registry = AdapterRegistry()
        registry.register(FakeAdapter())
        r = GenerationRouter(registry, comfyui_upload_fn=capture_upload)

        adapter = registry.get("fake-adapter")
        req = GenerationRequest(
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            input_images=["data:image/png;base64,AAAA" + "B" * 300],
        )
        result = await r.upload_local_images(req, adapter)
        assert result.input_images == ["result.png"]
        # The data URI prefix should be stripped
        assert not captured_calls[0].startswith("data:")


# ═══════════════════════════════════════════════════════════════════
# V1 → V2 Compatibility Layer Tests
# ═══════════════════════════════════════════════════════════════════


from generation.v1_compat import (
    form_to_generation_request,
    generation_result_to_v1_response,
    _parse_loras,
)


class TestParseLoras:
    """Test LoRA config parsing from V1 JSON string format."""

    def test_parse_json_string(self):
        result = _parse_loras('[{"name": "my_lora.safetensors", "strength": 0.8}]')
        assert len(result) == 1
        assert result[0].name == "my_lora.safetensors"
        assert result[0].strength == 0.8

    def test_parse_list_of_dicts(self):
        result = _parse_loras([{"name": "a.safetensors"}, {"name": "b.safetensors", "strength": 0.5}])
        assert len(result) == 2
        assert result[1].strength == 0.5

    def test_parse_empty_string(self):
        assert _parse_loras("") == []
        assert _parse_loras("[]") == []

    def test_parse_none(self):
        assert _parse_loras(None) == []

    def test_parse_invalid_json(self):
        assert _parse_loras("not-json{") == []

    def test_skip_entries_without_name(self):
        result = _parse_loras('[{"name": ""}, {"name": "valid.safetensors"}]')
        assert len(result) == 1
        assert result[0].name == "valid.safetensors"

    def test_dual_stage_fields(self):
        result = _parse_loras('[{"name": "wan_lora", "strength": 0.7, "high": "high.safetensors", "low": "low.safetensors"}]')
        assert result[0].high == "high.safetensors"
        assert result[0].low == "low.safetensors"


class TestFormToGenerationRequest:
    """Test V1 form → V2 GenerationRequest conversion."""

    @pytest.mark.asyncio
    async def test_basic_t2i_form(self):
        req = await form_to_generation_request(
            form={
                "prompt": "a sunset over mountains",
                "negative_prompt": "ugly, blurry",
                "steps": 30,
                "cfg": 7.5,
                "seed": 42,
                "aspect_ratio": "1:1",
                "checkpoint": "model.safetensors",
            },
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            adapter_hint="sdxl-local-t2i",
        )
        assert req.operation == Operation.GENERATE
        assert req.target_type == MediaType.IMAGE
        assert req.adapter_hint == "sdxl-local-t2i"
        assert req.prompt == "a sunset over mountains"
        assert req.negative_prompt == "ugly, blurry"
        assert req.steps == 30
        assert req.cfg == 7.5
        assert req.seed == 42
        assert req.checkpoint == "model.safetensors"

    @pytest.mark.asyncio
    async def test_field_aliases(self):
        """V1 field names like num_frames, sampler_name get mapped to V2 names."""
        req = await form_to_generation_request(
            form={
                "prompt": "test",
                "num_frames": 81,
                "sampler_name": "euler",
                "guidance_scale": 4.5,
            },
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
        )
        assert req.frames == 81
        assert req.sampler == "euler"
        assert req.cfg == 4.5

    @pytest.mark.asyncio
    async def test_lora_configs_json_string(self):
        req = await form_to_generation_request(
            form={
                "prompt": "test",
                "lora_configs": '[{"name": "style.safetensors", "strength": 0.6}]',
            },
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
        )
        assert len(req.loras) == 1
        assert req.loras[0].name == "style.safetensors"
        assert req.loras[0].strength == 0.6

    @pytest.mark.asyncio
    async def test_audio_form(self):
        req = await form_to_generation_request(
            form={
                "text": "Hello world",
                "mode": "tts",
                "voice": "nova",
                "duration": 10,
            },
            operation=Operation.GENERATE,
            target_type=MediaType.AUDIO,
            adapter_hint="local-mmaudio",
        )
        assert req.prompt == "Hello world"
        assert req.audio_mode == "tts"
        assert req.voice == "nova"
        assert req.duration == 10

    @pytest.mark.asyncio
    async def test_none_values_skipped(self):
        req = await form_to_generation_request(
            form={"prompt": "test", "steps": None, "cfg": None},
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
        )
        assert req.steps is None
        assert req.cfg is None

    @pytest.mark.asyncio
    async def test_unknown_fields_ignored(self):
        req = await form_to_generation_request(
            form={"prompt": "test", "unknown_field_xyz": "value"},
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
        )
        assert req.prompt == "test"

    @pytest.mark.asyncio
    async def test_file_upload_to_input_images(self):
        """Uploaded image files become base64 input_images."""
        mock_file = AsyncMock(spec=UploadFile)
        mock_file.read.return_value = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        req = await form_to_generation_request(
            form={"prompt": "test"},
            files={"image": mock_file},
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
        )
        assert len(req.input_images) == 1
        assert len(req.input_images[0]) > 10  # base64 string

    @pytest.mark.asyncio
    async def test_video_upload(self):
        mock_file = AsyncMock(spec=UploadFile)
        mock_file.read.return_value = b"\x00" * 50
        req = await form_to_generation_request(
            form={"prompt": "test"},
            files={"video": mock_file},
            operation=Operation.TRANSFORM,
            target_type=MediaType.VIDEO,
        )
        assert req.input_video is not None
        assert len(req.input_video) > 5


class TestGenerationResultToV1Response:
    """Test V2 GenerationResult → V1 response dict conversion."""

    def test_standard_queued_response(self):
        result = GenerationResult(
            prompt_id="abc123",
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=5,
            adapter_name="sdxl-local-t2i",
            meta={"width": 1024, "height": 1024},
        )
        resp = generation_result_to_v1_response(result)
        assert resp["status"] == "queued"
        assert resp["prompt_id"] == "abc123"
        assert resp["job_id"] == "abc123"
        assert resp["credits_used"] == 5
        assert resp["meta"]["width"] == 1024

    def test_cloud_response_includes_runpod_id(self):
        result = GenerationResult(
            prompt_id="xyz789",
            status="queued_cloud",
            compute_target=ComputeTarget.CLOUD,
            credits_used=10,
            runpod_job_id="rp-abc-123",
            adapter_name="wan22-cloud-i2v",
            meta={},
        )
        resp = generation_result_to_v1_response(result, v1_format="cloud")
        assert resp["status"] == "queued_cloud"
        assert resp["runpod_job_id"] == "rp-abc-123"

    def test_completed_status_mapping(self):
        result = GenerationResult(
            prompt_id="done1",
            status="completed",
            compute_target=ComputeTarget.LOCAL,
            credits_used=3,
            adapter_name="local-mmaudio",
        )
        resp = generation_result_to_v1_response(result)
        assert resp["status"] == "completed"

    def test_no_meta(self):
        result = GenerationResult(
            prompt_id="p1",
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=2,
            adapter_name="test",
        )
        resp = generation_result_to_v1_response(result)
        # Empty meta dict still present
        assert "meta" not in resp or resp["meta"] == {}


class TestDispatchV1:
    """Test the all-in-one dispatch_v1 helper."""

    @pytest.mark.asyncio
    async def test_dispatch_v1_calls_router(self):
        """dispatch_v1 should convert form, dispatch, and return V1 response."""
        from generation.v1_compat import dispatch_v1, init_v1_compat

        mock_router = AsyncMock()
        mock_router.dispatch.return_value = GenerationResult(
            prompt_id="test-123",
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=5,
            adapter_name="sdxl-local-t2i",
            meta={"width": 1024, "height": 1024},
        )
        mock_check = AsyncMock()
        mock_deduct = AsyncMock()

        init_v1_compat(
            router=mock_router,
            check_credits=mock_check,
            deduct_credits=mock_deduct,
        )

        mock_user = MagicMock()
        mock_user.id = "user-1"

        result = await dispatch_v1(
            form={"prompt": "test", "steps": 30},
            files={},
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            adapter_hint="sdxl-local-t2i",
            user=mock_user,
        )

        # Should call router.dispatch with a GenerationRequest
        assert mock_router.dispatch.called
        call_args = mock_router.dispatch.call_args
        gen_req = call_args[0][0]
        assert gen_req.prompt == "test"
        assert gen_req.steps == 30
        assert gen_req.adapter_hint == "sdxl-local-t2i"

        # Should return V1 response format
        assert result["status"] == "queued"
        assert result["prompt_id"] == "test-123"
        assert result["credits_used"] == 5

    @pytest.mark.asyncio
    async def test_dispatch_v1_registers_job(self):
        """dispatch_v1 should call register_job when settings provided."""
        from generation.v1_compat import dispatch_v1, init_v1_compat

        mock_router = AsyncMock()
        mock_router.dispatch.return_value = GenerationResult(
            prompt_id="job-456",
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=3,
            adapter_name="test",
        )
        mock_client = MagicMock()
        mock_get_client = MagicMock(return_value=mock_client)

        init_v1_compat(
            router=mock_router,
            check_credits=AsyncMock(),
            deduct_credits=AsyncMock(),
            get_comfyui_client=mock_get_client,
        )

        mock_user = MagicMock()
        mock_user.id = "user-2"

        await dispatch_v1(
            form={"prompt": "register test"},
            files={},
            operation=Operation.GENERATE,
            target_type=MediaType.IMAGE,
            adapter_hint="test",
            user=mock_user,
            register_job_settings={"job_type": "t2i"},
        )

        # Should register the job
        mock_client.register_job.assert_called_once()
        call_kwargs = mock_client.register_job.call_args
        assert call_kwargs[1]["prompt_id"] == "job-456"
        assert call_kwargs[1]["user_id"] == "user-2"

    @pytest.mark.asyncio
    async def test_dispatch_v1_cloud_format(self):
        """Cloud dispatch should include runpod_job_id in response."""
        from generation.v1_compat import dispatch_v1, init_v1_compat

        mock_router = AsyncMock()
        mock_router.dispatch.return_value = GenerationResult(
            prompt_id="cloud-789",
            status="queued_cloud",
            compute_target=ComputeTarget.CLOUD,
            credits_used=15,
            runpod_job_id="rp-test-id",
            adapter_name="wan22-cloud-i2v",
        )

        init_v1_compat(
            router=mock_router,
            check_credits=AsyncMock(),
            deduct_credits=AsyncMock(),
        )

        result = await dispatch_v1(
            form={"prompt": "cloud test"},
            files={},
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            adapter_hint="wan22-cloud-i2v",
            user=MagicMock(id="user-3"),
            v1_format="cloud",
        )

        assert result["status"] == "queued_cloud"
        assert result["runpod_job_id"] == "rp-test-id"


from fastapi import UploadFile
