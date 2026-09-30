"""On-demand model resolution for the MiniMax-H3 RunPod worker handler.

The worker no longer downloads every checkpoint at startup: it fetches only the
files a job's workflow references. These tests cover the workflow scan and the
registry matching without touching the network or the real ComfyUI tree.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

HANDLER_PATH = (
    Path(__file__).resolve().parents[1] / "deploy" / "runpod-minimax-h3" / "handler.py"
)


@pytest.fixture(scope="module")
def handler_module(tmp_path_factory):
    """Import the worker handler with a stubbed runpod package and fake ComfyUI root."""
    comfy_root = tmp_path_factory.mktemp("comfyui")
    (comfy_root / "models").mkdir()

    stub = types.ModuleType("runpod")
    stub.serverless = types.SimpleNamespace(start=lambda *a, **k: None)
    sys.modules.setdefault("runpod", stub)

    spec = importlib.util.spec_from_file_location("h3_worker_handler", HANDLER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.COMFYUI_PATH = str(comfy_root)
    return module


def test_workflow_model_names_maps_loader_inputs(handler_module):
    workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "a.safetensors"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "b.safetensors"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "c.safetensors"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "d.safetensors"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "not-a-model.safetensors"}},
        "6": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["7", 0]}},
    }
    names = handler_module._workflow_model_names(workflow)
    assert names == {
        "a.safetensors": "diffusion_models",
        "b.safetensors": "loras",
        "c.safetensors": "text_encoders",
        "d.safetensors": "vae",
    }


def test_startup_models_are_only_the_shared_core(handler_module):
    """Checkpoints and turbo LoRAs must not be downloaded at startup."""
    startup = {
        m["filename"] for m in handler_module.MINIMAX_H3_MODELS if m.get("startup_required")
    }
    assert startup == {
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "minimax_h3_video_vae_fp16.safetensors",
        "minimax_h3_audio_vae_fp32.safetensors",
    }
    assert "minimax_h3_fl2va_pruned_int8_convrot.safetensors" not in startup
    assert "h3ErosMax_beta5_3185144.safetensors" not in startup


def test_ensure_workflow_models_downloads_only_referenced_entries(handler_module, monkeypatch):
    """Only registry files the workflow names are fetched; unregistered names are ignored."""
    requested = []

    def fake_download(model):
        requested.append(model["filename"])
        return True

    monkeypatch.setattr(handler_module, "download_model", fake_download)
    workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "h3ErosMax_beta5_3185144.safetensors"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "user/own.safetensors"}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"}},
    }
    downloaded, missing = handler_module.ensure_workflow_models(workflow)

    assert downloaded == [
        "h3ErosMax_beta5_3185144.safetensors",
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    ]
    assert missing == []
    assert requested == downloaded  # official DiT and turbo LoRAs untouched


def test_ensure_workflow_models_reports_failures(handler_module, monkeypatch):
    monkeypatch.setattr(handler_module, "download_model", lambda model: False)
    workflow = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "h3ErosMax_beta5_3185144.safetensors"}}
    }
    downloaded, missing = handler_module.ensure_workflow_models(workflow)
    assert downloaded == []
    assert missing == ["h3ErosMax_beta5_3185144.safetensors"]


def test_ensure_models_skips_on_demand_entries(handler_module, monkeypatch):
    requested = []

    def fake_download(model):
        requested.append(model["filename"])
        return True

    monkeypatch.setattr(handler_module, "download_model", fake_download)
    monkeypatch.setattr(handler_module, "ensure_model_directories", lambda: None)
    monkeypatch.setattr(handler_module, "setup_model_links", lambda: None)

    result = handler_module.ensure_models()

    assert result.models_ready is True
    assert set(requested) == {
        "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "minimax_h3_video_vae_fp16.safetensors",
        "minimax_h3_audio_vae_fp32.safetensors",
    }


def test_hf_local_dir_lands_every_registry_entry_on_target(handler_module, tmp_path):
    """The download must land on the target, so no move (and no nesting) is needed."""
    handler_module.COMFYUI_PATH = str(tmp_path)
    for model in handler_module.MINIMAX_H3_MODELS:
        target = tmp_path / "models" / model["target_dir"] / model["filename"]
        local_dir = handler_module._hf_local_dir(model["hf_path"], target)
        assert local_dir / model["hf_path"] == target, (
            f"{model['filename']}: hf_hub_download would nest it under {local_dir}"
        )


def test_hf_local_dir_without_repo_folder_uses_target_parent(handler_module, tmp_path):
    handler_module.COMFYUI_PATH = str(tmp_path)
    target = tmp_path / "models" / "loras" / "flat.safetensors"
    assert handler_module._hf_local_dir("flat.safetensors", target) == target.parent


def test_prune_empty_parents_stops_at_the_models_root(handler_module, tmp_path):
    """Real sequence: the file was moved into the target dir, the nesting is empty."""
    models_root = tmp_path / "models"
    target_dir = models_root / "diffusion_models"
    nested = target_dir / "diffusion_models"
    nested.mkdir(parents=True)
    (target_dir / "moved.safetensors").write_bytes(b"x")  # the completed move

    handler_module._prune_empty_parents(nested, models_root)

    assert not nested.exists(), "the empty nesting should be removed"
    assert (target_dir / "moved.safetensors").exists(), "the moved file must survive"
    assert target_dir.exists(), "the target folder must survive"
    assert models_root.exists()


def test_prune_empty_parents_keeps_directories_with_content(handler_module, tmp_path):
    models_root = tmp_path / "models"
    nested = models_root / "vae" / "vae"
    nested.mkdir(parents=True)
    (nested / "keep.safetensors").write_bytes(b"x")
    handler_module._prune_empty_parents(nested, models_root)
    assert nested.exists() and (nested / "keep.safetensors").exists()
