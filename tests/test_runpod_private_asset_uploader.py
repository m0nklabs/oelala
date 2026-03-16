"""Tests for the RunPod private asset uploader policy."""

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).parent.parent / "deploy" / "runpod" / "upload_private_assets.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("oelala_runpod_private_uploader", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_blocks_public_model_filename(tmp_path):
    module = _load_module()
    blocked = tmp_path / "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors"
    blocked.write_bytes(b"x")

    try:
        module._validate_candidate(blocked, "models/loras")
    except ValueError as error:
        assert "Blocked public/general model filename" in str(error)
    else:
        raise AssertionError("Expected public model upload to be blocked")


def test_allows_lora_upload_and_builds_lora_key(tmp_path):
    module = _load_module()
    lora_root = tmp_path / "loras"
    lora_path = lora_root / "character" / "my-style.safetensors"
    lora_path.parent.mkdir(parents=True, exist_ok=True)
    lora_path.write_bytes(b"x")
    module.DEFAULT_LORA_ROOTS = (lora_root,)

    module._validate_candidate(lora_path, "models/loras")
    remote_key = module._build_remote_key(lora_path, "models/loras")

    assert remote_key == "models/loras/character/my-style.safetensors"


def test_rejects_invalid_remote_prefix():
    module = _load_module()

    try:
        module._normalize_remote_prefix("models/unet")
    except ValueError as error:
        assert "Remote prefix must be one of" in str(error)
    else:
        raise AssertionError("Expected invalid remote prefix to be rejected")
