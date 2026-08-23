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
    from pathlib import Path
    monkeypatch.setattr(cb, "_json_path", Path("/nonexistent/path/backends.json"))
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


def test_enabled_filter_excludes_disabled(tmp_path, monkeypatch):
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


def test_save_load_roundtrip(tmp_path, monkeypatch):
    # Round-trip on a throwaway file so the real config stays untouched.
    fp = tmp_path / "backends.json"
    monkeypatch.setattr(cb, "_json_path", fp)
    cb._loaded = False
    backends = cb.load_backends(force=True).copy()
    cb.save_backends(backends)
    reloaded = cb.load_backends(force=True)
    assert [b.model_dump(mode="json") for b in reloaded] == [
        b.model_dump(mode="json") for b in backends
    ]


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


def test_client_fn_raises_for_runpod(monkeypatch):
    cb.load_backends(force=True)
    # ltx is runpod-only -> the local client fn raises instead of returning None
    # (a local adapter that reached this point is misconfigured; returning None
    # would later crash with a cryptic AttributeError).
    fn = cb.client_fn_for_model("ltx")
    with pytest.raises(RuntimeError, match="runpod"):
        fn()


def test_client_fn_for_utility_exposes_model_family():
    cb.load_backends(force=True)
    fn = cb.client_fn_for_utility()
    assert fn.model_family == "utility"
    # .backend_id is refreshed on each call (matching client_fn_for_model), so
    # utility jobs record the ComfyUI server that actually ran them.
    assert fn.backend_id is None
    client = fn()
    assert fn.backend_id == "ai-kvm2-comfyui"
    assert client is not None


def test_skips_invalid_backend_type_keeps_others(tmp_path, monkeypatch):
    # A malformed entry (unknown 'type') is skipped with the rest preserved,
    # instead of resetting the whole inventory to built-in defaults.
    fp = tmp_path / "backends.json"
    fp.write_text(
        json.dumps({
            "backends": [
                {"id": "good", "name": "Good", "type": "comfyui",
                 "base_url": "http://localhost:8188", "enabled": True,
                 "model_families": ["sdxl"]},
                {"id": "bad", "name": "Bad", "type": "bogus",
                 "base_url": "http://localhost:9999", "enabled": True,
                 "model_families": ["sdxl"]},
            ]
        })
    )
    monkeypatch.setattr(cb, "_json_path", fp)
    cb._loaded = False
    backends = cb.load_backends(force=True)
    ids = [b.id for b in backends]
    assert "good" in ids
    assert "bad" not in ids


def test_rejects_non_http_scheme_for_comfyui_backend():
    # ComfyUIClient builds http://{host}:{port}; a non-http scheme would be
    # silently ignored, so the inventory model must reject it explicitly.
    with pytest.raises(Exception):
        cb.ComputeBackend(
            id="https-bad", name="HTTPS", type="comfyui",
            base_url="https://192.0.2.1:8188", enabled=True, model_families=[],
        )
    with pytest.raises(Exception):
        cb.ComputeBackend(
            id="ftp-bad", name="FTP", type="comfyui",
            base_url="ftp://192.0.2.1:21", enabled=True, model_families=[],
        )
    # Plain http (and scheme-less host:port) remain valid.
    cb.ComputeBackend(
        id="http-ok", name="HTTP", type="comfyui",
        base_url="http://192.0.2.1:8188", enabled=True, model_families=[],
    )


def test_get_comfyui_client_for_backend_refreshes_on_base_url_change(monkeypatch):
    # Fix: client cache is keyed by (backend_id, base_url), so editing a
    # backend's base_url yields a fresh client on the next dispatch instead of
    # returning the stale one.
    from comfyui_client import get_comfyui_client_for_backend

    b = cb.ComputeBackend(
        id="test-backend", name="Test", type="comfyui",
        base_url="http://192.0.2.1:8188", enabled=True, model_families=[],
    )
    c1 = get_comfyui_client_for_backend(b)
    b.base_url = "http://192.0.2.2:8188"
    c2 = get_comfyui_client_for_backend(b)
    assert c1.base_url == "http://192.0.2.1:8188"
    assert c2.base_url == "http://192.0.2.2:8188"
    assert c1 is not c2
