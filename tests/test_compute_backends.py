"""
Tests for the Compute Backend Inventory (modular compute sources).

Covers: config loading, fallback defaults, backend lookup, model-family
resolution (local-first), client-fn generation, and save/load round-trip.
"""

import os
import sys
import json

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "backend"))

from generation import compute_backends as cb


@pytest.fixture(autouse=True)
def reset_cache():
    """Reset the module-level cache before/after each test."""
    cb._loaded = False
    cb._backends = []
    yield
    cb._loaded = False
    cb._backends = []


def test_loads_inventory_from_json():
    backends = cb.load_backends(force=True)
    ids = [b.id for b in backends]
    assert "ai-kvm2-comfyui" in ids
    assert "windows-pc-comfyui" in ids
    assert "runpod-cloud" in ids


def test_default_fallback_when_json_missing(monkeypatch):
    monkeypatch.setattr(cb, "_json_path", "/nonexistent/path/backends.json")
    cb._loaded = False
    backends = cb.load_backends(force=True)
    assert len(backends) == 3


def test_get_backend():
    cb.load_backends(force=True)
    b = cb.get_backend("windows-pc-comfyui")
    assert b is not None
    assert b.type == "comfyui"
    assert b.base_url


def test_resolve_local_first_for_shared_family():
    # wan2.2 runs on ai-kvm2 (comfyui, local-first) even though runpod can too
    cb.load_backends(force=True)
    b = cb.resolve_backend_for_model("wan2.2")
    assert b is not None
    assert b.type == "comfyui"
    assert b.id == "ai-kvm2-comfyui"


def test_resolve_minimax_h3_windows():
    cb.load_backends(force=True)
    b = cb.resolve_backend_for_model("minimax_h3")
    assert b is not None
    assert b.id == "windows-pc-comfyui"


def test_resolve_runpod_only_family():
    cb.load_backends(force=True)
    b = cb.resolve_backend_for_model("ltx")
    assert b is not None
    assert b.type == "runpod"


def test_resolve_unknown_returns_none():
    cb.load_backends(force=True)
    assert cb.resolve_backend_for_model("does-not-exist") is None


def test_enabled_filter_excludes_disabled(tmp_path):
    # Point the module at a throwaway file so we don't touch the real config.
    fp = tmp_path / "backends.json"
    fp.write_text(
        json.dumps({
            "backends": [
                {"id": "windows-pc-comfyui", "name": "Windows", "type": "comfyui",
                 "base_url": "http://192.168.1.245:8188", "enabled": True,
                 "model_families": ["minimax_h3"]},
                {"id": "runpod-cloud", "name": "RunPod", "type": "runpod",
                 "base_url": "", "enabled": True, "model_families": ["minimax_h3"]},
            ]
        })
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cb, "_json_path", fp)
    cb._loaded = False
    cb.load_backends(force=True)
    # Disable windows backend -> minimax_h3 resolves to runpod instead
    backends = cb.list_backends()
    for b in backends:
        if b.id == "windows-pc-comfyui":
            b.enabled = False
    cb.save_backends(backends)
    b = cb.resolve_backend_for_model("minimax_h3")
    assert b is not None
    assert b.id == "runpod-cloud"
    monkeypatch.undo()


def test_save_load_roundtrip(tmp_path):
    # Round-trip on a throwaway file so the real config stays untouched.
    fp = tmp_path / "backends.json"
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(cb, "_json_path", fp)
    cb._loaded = False
    backends = cb.load_backends(force=True).copy()
    cb.save_backends(backends)
    reloaded = cb.load_backends(force=True)
    assert [b.model_dump(mode="json") for b in reloaded] == [
        b.model_dump(mode="json") for b in backends
    ]
    monkeypatch.undo()


def test_client_fn_for_model_exposes_backend_id(monkeypatch):
    cb.load_backends(force=True)
    fn = cb.client_fn_for_model("minimax_h3")
    # .backend_id is refreshed on each call (matching the router, which calls
    # the fn before reading backend_id) so it never drifts after admin edits.
    assert fn.backend_id is None
    client = fn()
    assert fn.backend_id == "windows-pc-comfyui"
    assert client is not None
    assert client.base_url == "http://192.168.1.245:8188"


def test_client_fn_returns_none_for_runpod(monkeypatch):
    cb.load_backends(force=True)
    # ltx is runpod-only -> client fn resolves nothing
    fn = cb.client_fn_for_model("ltx")
    with monkeypatch.context() as m:
        # Ensure no comfyui backend can serve ltx without network calls
        client = fn()
    assert client is None
