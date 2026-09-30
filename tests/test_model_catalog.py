"""Model catalog integrity.

`docs/model_catalog.yaml` is generated from the live registries and the local
ComfyUI tree. These tests keep it from drifting silently: every model the worker
registry or the H3 variant map can reference must appear in the catalog.
"""

import importlib.util
import os
import sys
import types
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "backend"))

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "docs" / "model_catalog.yaml"
HANDLER_PATH = PROJECT_ROOT / "deploy" / "runpod-minimax-h3" / "handler.py"


@pytest.fixture(scope="module")
def catalog():
    assert CATALOG_PATH.exists(), "docs/model_catalog.yaml is missing"
    data = yaml.safe_load(CATALOG_PATH.read_text())
    assert data["entries"], "catalog has no entries"
    return data


@pytest.fixture(scope="module")
def by_filename(catalog):
    return {entry["filename"]: entry for entry in catalog["entries"]}


@pytest.fixture(scope="module")
def handler_module():
    stub = types.ModuleType("runpod")
    stub.serverless = types.SimpleNamespace(start=lambda *a, **k: None)
    sys.modules.setdefault("runpod", stub)
    spec = importlib.util.spec_from_file_location("h3_worker_handler_catalog", HANDLER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_catalog_entries_are_well_formed(catalog):
    required = {"filename", "role", "target_dir", "family", "sources", "available_on"}
    for entry in catalog["entries"]:
        missing = required - set(entry)
        assert not missing, f"{entry.get('filename')}: missing {sorted(missing)}"
        assert entry["filename"], "entry without filename"
        assert entry["available_on"], f"{entry['filename']}: no availability recorded"


def test_catalog_counts_match_entries(catalog):
    counts = catalog["meta"]["counts"]
    assert counts["entries"] == len(catalog["entries"])
    assert sum(counts["by_role"].values()) == len(catalog["entries"])


def test_worker_registry_models_are_cataloged(by_filename, handler_module):
    for model in handler_module.MINIMAX_H3_MODELS:
        entry = by_filename.get(model["filename"])
        assert entry, f"worker model not in catalog: {model['filename']}"
        assert entry["target_dir"] == model["target_dir"]
        assert entry["available_on"].get("runpod_h3_worker") is not None


def test_h3_variant_checkpoints_are_cataloged(by_filename):
    from comfyui_client import MINIMAX_H3_VARIANTS

    for variant, spec in MINIMAX_H3_VARIANTS.items():
        filename = spec["checkpoint"]
        entry = by_filename.get(filename)
        assert entry, f"variant '{variant}' checkpoint not in catalog: {filename}"
        assert any(
            variant in str(usage) for usage in entry["used_by"]
        ), f"{filename}: variant '{variant}' not recorded in used_by"


def test_eros_checkpoints_keep_upstream_names(by_filename):
    """Mirrored Civitai files must keep the upstream name (trailing file id)."""
    for filename in (
        "h3ErosMax_beta5_3185144.safetensors",
        "h3ErosMax_beta5_3178732.safetensors",
        "h3ErosMax_beta5_3185154.safetensors",
    ):
        entry = by_filename.get(filename)
        assert entry, f"{filename} missing from catalog"
        sources = entry["sources"]
        assert any(
            "civitai" in str(key).lower() or "civitai" in str(value).lower()
            for key, value in sources.items()
        ), f"{filename}: no Civitai source recorded"
