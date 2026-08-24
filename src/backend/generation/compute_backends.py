"""
Compute Backend Inventory — configurable sources of compute for OELALA generation.

Every generation runs on a *compute backend*. A backend is one of:

- ``comfyui`` — any ComfyUI server (headless or desktop) reachable over HTTP by
  ``base_url``. Both the ai-kvm2 default (``localhost:8188``) and the user's
  Windows-PC (configured via ``COMFYUI_WINDOWS_HOST``) are ComfyUI backends.
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
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# Used by adapters that don't carry a model-specific family (interpolate,
# upscale, lipsync, faceswap, caption, voice-clone, mmaudio) — they run on the
# default local ComfyUI.
UTILITY_FAMILY = "utility"


class ComputeBackend(BaseModel):
    """A configurable source of compute for generation."""

    # Slug-constrained so the id stays URL-safe ({backend_id} path parameters).
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str
    type: Literal["comfyui", "runpod"] = "comfyui"
    base_url: str = ""
    enabled: bool = True
    model_families: List[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _validate_inventory_entry(self):
        # Enforce the same type-based base_url rules on the inventory model itself,
        # so a manually edited compute_backends.json can't load a broken entry
        # (an entry load_backends() would otherwise skip anyway).
        if self.type == "comfyui":
            # A comfyui backend must name a reachable HTTP server.
            if not self.base_url:
                raise ValueError("base_url is required for a comfyui backend")
            # ComfyUIClient always builds an http://{host}:{port} base_url, so a
            # non-http scheme (https/ftp/...) would be silently ignored — reject
            # it explicitly, matching the Admin API constraints.
            if "://" in self.base_url and not self.base_url.startswith("http://"):
                raise ValueError("base_url must use http:// for a comfyui backend")
            authority = (
                self.base_url.split("://", 1)[1].split("/")[0]
                if "://" in self.base_url
                else self.base_url.split("/")[0]
            )
            host = authority.split(":")[0]
            if not host:
                raise ValueError("base_url must include a host for a comfyui backend")
            # An explicit port must be numeric and in range. _parse_base_url()
            # falls back to 8188 on parse failure, so without this check a typo
            # like 'http://host:abc' would silently target the wrong server.
            if ":" in authority:
                port = authority.split(":", 1)[1]
                if not port.isdigit() or not (1 <= int(port) <= 65535):
                    raise ValueError(
                        "base_url port must be a number between 1 and 65535"
                    )
        elif self.type == "runpod" and self.base_url:
            # A runpod backend has no base_url; a URL here is meaningless noise.
            raise ValueError("base_url must be empty for a runpod backend")
        return self


# Path to the inventory JSON (next to this module).
_BACKENDS_FILE = Path(__file__).with_name("compute_backends.json")

# In-memory source of truth, loaded lazily.
_json_path: Path = Path(os.getenv("COMPUTE_BACKENDS_JSON", str(_BACKENDS_FILE)))
_backends: List[ComputeBackend] = []
_loaded = False


def _default_backends() -> List[ComputeBackend]:
    """Built-in fallback inventory (ai-kvm2, Windows-PC, RunPod).

    The Windows-PC address comes solely from env config — it is never
    hardcoded here. When ``COMFYUI_WINDOWS_HOST`` is unset the backend is
    omitted entirely rather than created with an invalid empty base_url.
    """
    backends = [
        ComputeBackend(
            id="ai-kvm2-comfyui",
            name="ai-kvm2 ComfyUI (local)",
            type="comfyui",
            base_url="http://localhost:8188",
            enabled=True,
            model_families=["wan2.2", "sdxl", "flux", "flux2", "krea2", UTILITY_FAMILY],
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
    windows_host = os.getenv("COMFYUI_WINDOWS_HOST", "").strip()
    if windows_host:
        windows_port = os.getenv("COMFYUI_WINDOWS_PORT", "8188").strip() or "8188"
        backends.insert(
            1,
            ComputeBackend(
                id="windows-pc-comfyui",
                name="Windows-PC ComfyUI",
                type="comfyui",
                base_url=f"http://{windows_host}:{windows_port}",
                enabled=True,
                model_families=["minimax_h3"],
            ),
        )
    return backends


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
        parsed = []
        for b in raw:
            try:
                parsed.append(ComputeBackend(**b))
            except Exception as exc:
                # Skip a single bad entry instead of dropping the whole
                # inventory (an invalid type/base_url must not reset everything).
                logger.warning(f"⚠️ Skipping invalid compute backend entry: {exc}")
        if not parsed:
            logger.warning(
                "⚠️ No valid compute backends in inventory; using built-in defaults"
            )
            parsed = _default_backends()
        else:
            logger.info(f"🗄️ Loaded {len(parsed)} compute backends from {_json_path}")
        # Assign only once so concurrent readers never observe a partially
        # populated module-level _backends during a reload (atomic swap).
        _backends = parsed
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
    # Ensure the target directory exists (e.g. a fresh COMPUTE_BACKENDS_JSON
    # path in a containerized deployment) before writing the temp file.
    _json_path.parent.mkdir(parents=True, exist_ok=True)
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
    dispatch without a backend restart. It raises a clear ``RuntimeError`` when
    it cannot return a local ComfyUI client — i.e. when no enabled backend
    supports the family, or when the family resolves to a **runpod** backend.

    The returned callable exposes ``.backend_id`` (the id of the backend it
    currently resolves to, refreshed on every call so it never drifts from the
    server actually used after admin edits) and ``.model_family`` so
    adapters/routers can tag jobs with the backend that ran them.
    """
    # Import via the canonical src.backend path to avoid loading comfyui_client
    # twice (two module instances = two separate client caches), falling back
    # to the bare name only when running with src/backend on sys.path (tests).
    try:
        from src.backend.comfyui_client import get_comfyui_client_for_backend
    except ImportError:
        from comfyui_client import get_comfyui_client_for_backend

    def _fn():
        backend = resolve_backend_for_model(model_family)
        # Keep .backend_id in sync with what this call would actually use, so
        # job metadata always matches the resolved dispatch backend.
        _fn.backend_id = (
            backend.id if backend is not None and backend.type == "comfyui" else None
        )
        if backend is None:
            # Nothing usable is configured — surface a clear error instead of
            # letting local adapters crash later with a cryptic AttributeError.
            raise RuntimeError(
                f"No enabled backend supports model family '{model_family}'"
            )
        if backend.type != "comfyui":
            # Resolved to a runpod (cloud) backend, but this fn provides a local
            # ComfyUI client. A local adapter that reaches this point is
            # misconfigured — fail explicitly rather than return None (which
            # would later crash with a cryptic AttributeError).
            raise RuntimeError(
                f"Model family '{model_family}' resolved to runpod backend "
                f"'{backend.id}'; no local ComfyUI client available"
            )
        return get_comfyui_client_for_backend(backend)

    _fn.backend_id = None
    _fn.model_family = model_family
    return _fn


def client_fn_for_utility():
    """Client fn for utility adapters (no model family) — always the default
    local ComfyUI backend.

    Exposes ``.backend_id`` (refreshed on every call, like
    ``client_fn_for_model``) so utility jobs can be tagged with the ComfyUI
    server that actually ran them. Raises a clear ``RuntimeError`` when no
    enabled comfyui backend is available.
    """
    try:
        from src.backend.comfyui_client import get_comfyui_client_for_backend
    except ImportError:
        from comfyui_client import get_comfyui_client_for_backend

    def _fn():
        backend = resolve_backend_for_model(UTILITY_FAMILY)
        # Keep .backend_id in sync with what this call would actually use, so
        # job metadata always matches the resolved dispatch backend.
        _fn.backend_id = (
            backend.id if backend is not None and backend.type == "comfyui" else None
        )
        if backend is None or backend.type != "comfyui":
            raise RuntimeError("No enabled comfyui backend for utility adapters")
        return get_comfyui_client_for_backend(backend)

    _fn.backend_id = None
    _fn.model_family = UTILITY_FAMILY
    return _fn
