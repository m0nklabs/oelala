"""Download-path layout for the LTX-2.3 RunPod worker handler.

hf_hub_download writes to ``local_dir/<hf_path>``. The Gemma text-encoder entry
in ``LTX23_MODELS`` carries a ``split_files/...`` repo path, so passing the
target's own directory nested the file one level too deep
(``models/text_encoders/split_files/text_encoders/x``) and forced a move on
every cold start. These tests cover the local_dir choice and the empty-dir
pruning without touching the network or the real ComfyUI tree.
"""

import importlib.util
import sys
import types
from pathlib import Path

import pytest

HANDLER_PATH = (
    Path(__file__).resolve().parents[1] / "deploy" / "runpod-ltx23" / "handler.py"
)


@pytest.fixture(scope="module")
def handler_module(tmp_path_factory):
    """Import the worker handler with a stubbed runpod package and fake ComfyUI root."""
    comfy_root = tmp_path_factory.mktemp("comfyui")
    (comfy_root / "models").mkdir()

    stub = types.ModuleType("runpod")
    stub.serverless = types.SimpleNamespace(start=lambda *a, **k: None)
    sys.modules.setdefault("runpod", stub)

    spec = importlib.util.spec_from_file_location("ltx23_worker_handler", HANDLER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.COMFYUI_PATH = str(comfy_root)
    return module


def test_flat_repo_paths_download_into_the_target_dir(handler_module, tmp_path):
    """Entries whose hf_path carries no folder download into the target dir.

    A flat hf_path may still rename the file (the audio VAE is
    ``LTX2_audio_vae_bf16.safetensors`` on the Hub, ``ltx2_audio_vae``
    locally); the existing move handles that without extra nesting.
    """
    handler_module.COMFYUI_PATH = str(tmp_path)
    models_root = tmp_path / "models"
    checked = 0
    for model in handler_module.LTX23_MODELS:
        if "hf_path" not in model or "/" in model["hf_path"]:
            continue
        target = models_root / model["target_dir"] / model["filename"]
        local_dir = handler_module._hf_local_dir(model["hf_path"], target)
        landed = local_dir / model["hf_path"]
        assert local_dir == target.parent, (
            f"{model['filename']}: a flat repo path must download into the target dir"
        )
        assert landed.parent == target.parent
        if model["hf_path"] == model["filename"]:
            assert landed == target
        checked += 1
    assert checked >= 2  # the distilled checkpoint and the audio VAE


def test_split_files_repo_paths_never_nest_inside_the_target_dir(handler_module, tmp_path):
    """The defect: local_dir=target.parent put the file under target_dir/split_files/..."""
    handler_module.COMFYUI_PATH = str(tmp_path)
    models_root = tmp_path / "models"
    checked = 0
    for model in handler_module.LTX23_MODELS:
        if "hf_path" not in model or "/" not in model["hf_path"]:
            continue
        target = models_root / model["target_dir"] / model["filename"]
        local_dir = handler_module._hf_local_dir(model["hf_path"], target)
        landed = local_dir / model["hf_path"]
        # A folder-carrying repo path downloads into the models root ...
        assert local_dir == models_root
        assert landed == models_root / model["hf_path"]
        # ... never one level deep inside the target's own folder.
        assert target.parent not in landed.parents
        checked += 1
    assert checked >= 1  # the Gemma text encoder


def test_prune_empty_parents_removes_the_split_files_nesting(handler_module, tmp_path):
    """Real sequence: the file was moved onto its target, split_files/ is empty."""
    models_root = tmp_path / "models"
    target_dir = models_root / "text_encoders"
    nested = models_root / "split_files" / "text_encoders"
    nested.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    (target_dir / "gemma_3_12B_it_fp8_scaled.safetensors").write_bytes(b"x")

    handler_module._prune_empty_parents(nested, models_root)

    assert not nested.exists()
    assert not (models_root / "split_files").exists()
    assert (target_dir / "gemma_3_12B_it_fp8_scaled.safetensors").exists()
    assert target_dir.exists() and models_root.exists()


def test_prune_empty_parents_keeps_directories_with_content(handler_module, tmp_path):
    models_root = tmp_path / "models"
    nested = models_root / "checkpoints" / "split_files" / "checkpoints"
    nested.mkdir(parents=True)
    (nested / "keep.safetensors").write_bytes(b"x")
    handler_module._prune_empty_parents(nested, models_root)
    assert nested.exists() and (nested / "keep.safetensors").exists()
