"""
Compute Backend Inventory — configurable sources of compute for OELALA generation.

Every generation runs on a *compute backend*. A backend is one of:

- ``comfyui`` — any ComfyUI server (headless or desktop) reachable over HTTP by
  ``base_url``. Both the ai-kvm2 default (``localhost:8188``) and the user's
  Windows-PC (``192.168.1.245:8188``) are ComfyUI backends.
- ``runpod`` — a serverless RunPod container, i.e. an ephemeral ComfyUI server
  submitted to via the RunPod client (submit_to_runpod_fn). RunPod is "just a
  container with a temporary ComfyUI server"; endpoint IDs stay in ``.env``.

Each backend declares which ``model_family`` capabilities it can run. The adapter
registry + router use this inventory to pick an enabled backend per request, so
adding a new ComfyUI server becomes a configuration change (Admin UI + JSON)
instead of a code change.

Configuration lives in ``compute_backends.json`` (same directory) and is editable
through the Admin panel → "Compute" API. If the file is missing/unreadable the
module falls back to a built-in default equal to the three known backends
(ai-kvm2, Windows-PC, RunPod), so generation never breaks.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Used by adapters that don't carry a model-specific family (interpolate,
# upscale, lipsync, faceswap, caption, voice-clone, mmaudio) — they run on the
# default local ComfyUI.
UTILITY_FAMILY = "utility"


class ComputeBackend(BaseModel):
    """A configurable source of compute for generation."""

    id: str
    name: str
    type: str = Field(description="'comfyui' or 'runpod'")
    base_url: str = ""
    enabled: bool = True
    model_families: List[str] = Field(default_factory=list)
    notes: str = ""

    # ── RunPod-specific (used only when type == 'runpod') ───────────
    # Endpoint/template IDs are kept in .env (secrets), referenced by name here.
    endpoint_env_var: Optional[str] = None


# Path to the inventory JSON (next to this module).
_BACKENDS_FILE = Path(__file__).with_name("compute_backends.json")

# In-memory source of truth, loaded lazily.
_json_path: Path = Path(os.getenv("COMPUTE_BACKENDS_JSON", str(_BACKENDS_FILE)))
_backends: List[ComputeBackend] = []
_loaded = False


def _default_backends() -> List[ComputeBackend]:
    """Built-in fallback inventory (ai-kvm2, Windows-PC, RunPod)."""
    return [
        ComputeBackend(
            id="ai-kvm2-comfyui",
            name="ai-kvm2 ComfyUI (local)",
            type="comfyui",
            base_url="http://localhost:8188",
            enabled=True,
            model_families=["wan2.2", "sdxl", "flux", "flux2", "krea2", UTILITY_FAMILY],
        ),
        ComputeBackend(
            id="windows-pc-comfyui",
            name="Windows-PC ComfyUI",
            type="comfyui",
            base_url=f"http://{os.getenv('COMFYUI_WINDOWS_HOST', '192.168.1.245')}:{os.getenv('COMFYUI_WINDOWS_PORT', '8188')}",
            enabled=bool(os.getenv("COMFYUI_WINDOWS_HOST", "").strip()),
            model_families=["minimax_h3"],
        ),
        ComputeBackend(
            id="runpod-cloud",
            name="RunPod serverless cloud",
            type="runpod",
            base_url="",
            enabled=True,
            model_families=[
                "wan2.2",
                "ltx",
                "minimax_h3",
                "qwen_image_edit",
                "i2i_edit_model",
            ],
        ),
    ]


def load_backends(force: bool = False) -> List[ComputeBackend]:
    """Load the backend inventory from disk (or built-in defaults).

    Cached after first load; call with force=True to re-read (e.g. after the
    Admin API writes a change).
    """
    global _backends, _loaded
    if _loaded and not force:
        return _backends

    try:
        with open(_json_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        raw = data.get("backends", [])
        _backends = [ComputeBackend(**b) for b in raw]
        logger.info(f"🗄️ Loaded {len(_backends)} compute backends from {_json_path}")
    except Exception as exc:
        logger.warning(
            f"⚠️ Could not load compute backends from {_json_path} ({exc}); "
            "using built-in defaults"
        )
        _backends = _default_backends()
    _loaded = True
    return _backends


def save_backends(backends: List[ComputeBackend]) -> None:
    """Persist the backend inventory to the JSON file and refresh cache."""
    data = {
        "$comment": "Compute Backend Inventory — managed via Admin panel → Compute.",
        "backends": [b.model_dump(mode="json") for b in backends],
    }
    tmp = _json_path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    tmp.replace(_json_path)
    load_backends(force=True)


def list_backends() -> List[ComputeBackend]:
    """Return all backends (enabled and disabled)."""
    return list(load_backends())


def get_backend(backend_id: str) -> Optional[ComputeBackend]:
    """Look up a backend by id."""
    for b in load_backends():
        if b.id == backend_id:
            return b
    return None


def enabled_backends() -> List[ComputeBackend]:
    """Return only enabled backends."""
    return [b for b in load_backends() if b.enabled]


def resolve_backend_for_model(model_family: str) -> Optional[ComputeBackend]:
    """Find the first enabled backend able to run ``model_family``.

    Returns None when nothing enabled can run it. ComfyUI backends are preferred
    over runpod when both can run a family (local-first), matching the router's
    existing local-over-cloud preference.
    """
    candidates = [b for b in enabled_backends() if model_family in b.model_families]
    if not candidates:
        return None
    # Local-first: comfyui ahead of runpod. Stable sort (no secondary key)
    # preserves the inventory's original order within each type.
    candidates.sort(key=lambda b: b.type == "runpod")
    return candidates[0]


def client_fn_for_model(model_family: str):
    """Return a callable returning the ComfyUIClient for ``model_family``.

    The callable lazily resolves against the current inventory each call, so
    admin edits to backends (URLs, enable/disable) take effect on the next
    dispatch without a backend restart. Returns None for families that resolve
    to a runpod (non-comfyui) backend or to no enabled backend.

    The returned callable exposes ``.backend_id`` (the id of the backend it
    currently resolves to, refreshed on every call so it never drifts from the
    server actually used after admin edits) and ``.model_family`` so
    adapters/routers can tag jobs with the backend that ran them.
    """
    from comfyui_client import get_comfyui_client_for_backend

    def _fn():
        backend = resolve_backend_for_model(model_family)
        # Keep .backend_id in sync with what this call would actually use, so
        # job metadata always matches the resolved dispatch backend.
        _fn.backend_id = (
            backend.id if backend is not None and backend.type == "comfyui" else None
        )
        if backend is None or backend.type != "comfyui":
            return None
        return get_comfyui_client_for_backend(backend)

    _fn.backend_id = None
    _fn.model_family = model_family
    return _fn


def client_fn_for_utility():
    """Client fn for utility adapters (no model family) — always the default
    local ComfyUI backend.

    Exposes ``.backend_id`` (refreshed on every call, like
    ``client_fn_for_model``) so utility jobs can be tagged with the ComfyUI
    server that actually ran them.
    """
    from comfyui_client import get_comfyui_client_for_backend

    def _fn():
        backend = resolve_backend_for_model(UTILITY_FAMILY)
        # Keep .backend_id in sync with what this call would actually use, so
        # job metadata always matches the resolved dispatch backend.
        _fn.backend_id = (
            backend.id if backend is not None and backend.type == "comfyui" else None
        )
        if backend is None or backend.type != "comfyui":
            return None
        return get_comfyui_client_for_backend(backend)

    _fn.backend_id = None
    _fn.model_family = UTILITY_FAMILY
    return _fn
