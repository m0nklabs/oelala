"""Unit tests for RunPod worker cached-model linking."""

import builtins
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


HANDLER_PATH = Path(__file__).parent.parent / "deploy" / "runpod" / "handler.py"


def _load_handler_module(monkeypatch):
    """Load the RunPod worker module with a minimal fake runpod dependency."""
    fake_runpod = SimpleNamespace(
        serverless=SimpleNamespace(
            progress_update=lambda *args, **kwargs: None,
            start=lambda *args, **kwargs: None,
        )
    )
    monkeypatch.setitem(sys.modules, "runpod", fake_runpod)

    spec = importlib.util.spec_from_file_location("oelala_runpod_handler_test", HANDLER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_find_cached_model_source_reads_hf_snapshot_layout(monkeypatch, tmp_path):
    """The worker should find models inside Hugging Face snapshot-style cache roots."""
    module = _load_handler_module(monkeypatch)
    model = next(
        item
        for item in module.CLOUD_MAX_MODELS
        if item["filename"] == "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
    )

    cache_root = tmp_path / "hf-cache"
    cached_file = (
        cache_root
        / "models--Comfy-Org--Wan_2.2_ComfyUI_Repackaged"
        / "snapshots"
        / "abcdef"
        / "split_files"
        / "diffusion_models"
        / model["filename"]
    )
    cached_file.parent.mkdir(parents=True, exist_ok=True)
    cached_file.write_bytes(b"model-bytes")

    monkeypatch.setattr(module, "_candidate_cached_model_roots", lambda: [cache_root])

    assert module._find_cached_model_source(model) == cached_file


def test_download_requested_models_uses_cache_before_hf_download(monkeypatch, tmp_path):
    """Cached model hits should avoid importing or calling huggingface_hub entirely."""
    module = _load_handler_module(monkeypatch)
    model = next(
        item
        for item in module.CLOUD_MAX_MODELS
        if item["filename"] == "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
    )

    cache_root = tmp_path / "hf-cache"
    cached_file = cache_root / Path(model["hf_path"])
    cached_file.parent.mkdir(parents=True, exist_ok=True)
    with cached_file.open("wb") as handle:
        handle.truncate(51 * 1024 * 1024)

    volume_root = tmp_path / "volume" / "models"
    comfy_root = tmp_path / "comfy" / "models"

    monkeypatch.setattr(module, "_candidate_cached_model_roots", lambda: [cache_root])
    monkeypatch.setattr(
        module,
        "_model_destinations",
        lambda current_model: (
            volume_root / current_model["local_dir"] / current_model["filename"],
            comfy_root / current_model["local_dir"] / current_model["filename"],
        ),
    )

    original_import = builtins.__import__

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "huggingface_hub":
            raise AssertionError("huggingface_hub should not be imported on a cache hit")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    prepared = module.download_requested_models([model["filename"]])

    linked_path = comfy_root / model["local_dir"] / model["filename"]
    assert prepared == 1
    assert linked_path.is_symlink()
    assert linked_path.resolve() == cached_file.resolve()


def test_cloud_max_models_always_use_container_storage(monkeypatch, tmp_path):
    """Public/general Cloud Max models must never use the RunPod Network Volume."""
    module = _load_handler_module(monkeypatch)
    model = next(
        item
        for item in module.CLOUD_MAX_MODELS
        if item["filename"] == "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
    )

    volume_root = tmp_path / "volume"
    volume_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(module, "MODEL_VOLUME", str(volume_root))

    target_base = module._target_base_for_model(model)

    assert target_base == Path("/comfyui/models")


def test_writable_volume_models_dir_rejects_read_only_root(monkeypatch, tmp_path):
    """A cache-only /runpod-volume mount must not be treated as writable LoRA storage."""
    module = _load_handler_module(monkeypatch)

    volume_root = tmp_path / "volume"
    volume_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(module, "MODEL_VOLUME", str(volume_root))

    def fake_write_text(self, data, *args, **kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr(module.Path, "write_text", fake_write_text)

    detected = module._writable_volume_models_dir()

    assert detected is None


def test_is_model_present_ignores_volume_for_public_models(monkeypatch, tmp_path):
    """Stale public copies on volume must not suppress local downloads."""
    module = _load_handler_module(monkeypatch)
    model = next(
        item
        for item in module.CLOUD_MAX_MODELS
        if item["filename"] == "umt5_xxl_fp16.safetensors"
    )

    volume_file = tmp_path / "volume" / "models" / model["local_dir"] / model["filename"]
    volume_file.parent.mkdir(parents=True, exist_ok=True)
    volume_file.write_bytes(b"bytes")
    local_file = tmp_path / "comfy" / "models" / model["local_dir"] / model["filename"]

    monkeypatch.setattr(
        module,
        "_model_destinations",
        lambda current_model: (
            volume_file,
            local_file,
        ),
    )

    assert module._is_model_present(model) is False


def test_check_download_capacity_reports_insufficient_local_disk(monkeypatch, tmp_path):
    """The worker should fail fast when the target filesystem cannot hold the required downloads."""
    module = _load_handler_module(monkeypatch)
    models = [
        next(
            item
            for item in module.CLOUD_MAX_MODELS
            if item["filename"] == "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
        ),
        next(
            item
            for item in module.CLOUD_MAX_MODELS
            if item["filename"] == "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
        ),
    ]

    comfy_root = tmp_path / "comfy" / "models"
    staging_root = tmp_path / "staging"
    comfy_root.mkdir(parents=True, exist_ok=True)
    staging_root.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(module, "HF_STAGING_DIR", staging_root)
    monkeypatch.setattr(module, "DOWNLOAD_SAFETY_BUFFER_GB", 2.0)
    monkeypatch.setattr(module, "_target_base_for_model", lambda model: comfy_root)

    class _Usage:
        def __init__(self, free):
            self.free = free

    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: _Usage(free=10 * (1024 ** 3)))

    ok, message = module._check_download_capacity(models)

    assert ok is False
    assert message is not None
    assert "Increase containerDiskInGb" in message
    assert "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" in message


def test_missing_models_required_only_skips_mode_specific_models(monkeypatch, tmp_path):
    """Startup preflight should only demand shared core assets, not I2V/T2V-specific models."""
    module = _load_handler_module(monkeypatch)

    comfy_root = tmp_path / "comfy" / "models"
    volume_root = tmp_path / "volume" / "models"

    monkeypatch.setattr(
        module,
        "_model_destinations",
        lambda current_model: (
            volume_root / current_model["local_dir"] / current_model["filename"],
            comfy_root / current_model["local_dir"] / current_model["filename"],
        ),
    )

    missing = module._missing_models(required_only=True)
    missing_names = {model["filename"] for model in missing}

    assert "umt5_xxl_fp16.safetensors" in missing_names
    assert "wan_2.1_vae.safetensors" in missing_names
    assert "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" not in missing_names
    assert "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" not in missing_names
    assert "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors" not in missing_names
    assert "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors" not in missing_names
    assert "clip_vision_h.safetensors" not in missing_names


def test_ensure_workflow_models_downloads_only_referenced_t2v_models(monkeypatch):
    """A T2V workflow should not trigger I2V-only model preparation."""
    module = _load_handler_module(monkeypatch)

    workflow = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"},
        },
        "2": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"},
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": "umt5_xxl_fp16.safetensors"},
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "wan_2.1_vae.safetensors"},
        },
    }

    requested = []
    monkeypatch.setattr(module, "_is_model_present", lambda model: False)
    monkeypatch.setattr(
        module,
        "download_requested_models",
        lambda filenames: requested.extend(filenames) or len(filenames),
    )
    monkeypatch.setattr(module, "restart_comfyui", lambda: None)

    prepared = module.ensure_workflow_models(workflow)

    assert prepared == 4
    assert set(requested) == {
        "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        "umt5_xxl_fp16.safetensors",
        "wan_2.1_vae.safetensors",
    }
    assert "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" not in requested
    assert "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" not in requested
    assert "clip_vision_h.safetensors" not in requested


def test_detect_workflow_family_distinguishes_t2v_i2v_and_mixed(monkeypatch):
    """Workflow diagnostics should correctly classify referenced model families."""
    module = _load_handler_module(monkeypatch)

    assert module._detect_workflow_family({"wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"}) == "t2v"
    assert module._detect_workflow_family({"wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"}) == "i2v"
    assert module._detect_workflow_family({"clip_vision_h.safetensors"}) == "i2v"
    assert module._detect_workflow_family({
        "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
    }) == "mixed"
    assert module._detect_workflow_family({"umt5_xxl_fp16.safetensors"}) == "shared-core"


def test_startup_models_include_only_shared_core(monkeypatch):
    """Startup diagnostics should report only shared core assets as preload targets."""
    module = _load_handler_module(monkeypatch)

    startup_names = {model["filename"] for model in module._startup_models()}
    deferred_names = {model["filename"] for model in module._deferred_models()}

    assert startup_names == {"umt5_xxl_fp16.safetensors", "wan_2.1_vae.safetensors"}
    assert "clip_vision_h.safetensors" in deferred_names
    assert "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors" in deferred_names
    assert "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" in deferred_names
