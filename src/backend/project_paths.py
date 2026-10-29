"""Paths inside this checkout — the single source of truth.

Every operational path the backend uses is derived from the checkout this module lives in,
so a second clone or a git worktree reads and writes its own files instead of the first
deployment's (AGENTS.md rule 6: no environment-specific paths in committed code).

These replaced absolute ``/home/<user>/oelala/...`` literals spread over ``app.py``,
``admin_api.py`` and ``websocket_handler.py``. On the production checkout the resolved
values are identical to the old literals; anywhere else they point at that checkout, which
is the point — previously a worktree silently shared the production data directory.

Not covered here yet, because they are not checkout-relative and need configuration:
third-party binaries inside a virtualenv (``yt-dlp``), the MinIO data directory, and the
filesystem-browser roots in ``admin_api``.
"""

from pathlib import Path

#: Root of the checkout that contains this module (``<repo>/src/backend/project_paths.py``).
CHECKOUT_ROOT = Path(__file__).resolve().parents[2]


def checkout_root() -> Path:
    """Root of the checkout this backend runs from."""
    return CHECKOUT_ROOT


def data_dir() -> Path:
    """Persistent JSON state written at runtime (jobs, stats, settings)."""
    return CHECKOUT_ROOT / "data"


def logs_dir() -> Path:
    """Runtime logs written by the backend."""
    return CHECKOUT_ROOT / "logs"


def workflows_dir() -> Path:
    """ComfyUI workflow registry and templates."""
    return CHECKOUT_ROOT / "workflows"


def frontend_dir() -> Path:
    """Built frontend assets served by the backend."""
    return CHECKOUT_ROOT / "src" / "frontend"


def comfyui_dir() -> Path:
    """Local ComfyUI installation."""
    return CHECKOUT_ROOT / "ComfyUI"


def comfyui_output_dir() -> Path:
    """ComfyUI's output directory (canonical media location)."""
    return comfyui_dir() / "output"


def comfyui_loras_dir() -> Path:
    """LoRA models available to local workflows."""
    return comfyui_dir() / "models" / "loras"


def media_generated_dir() -> Path:
    """Generated media on local storage."""
    return CHECKOUT_ROOT / "media" / "generated"
