"""Paths inside this checkout — the single source of truth.

Every operational path the backend uses is derived from the checkout this module lives in,
so a second clone or a git worktree reads and writes its own files instead of the first
deployment's (AGENTS.md rule 6: no environment-specific paths in committed code).

These replaced absolute ``/home/<user>/oelala/...`` literals spread over ``app.py``,
``admin_api.py`` and ``websocket_handler.py``. On the production checkout the resolved
values are identical to the old literals; anywhere else they point at that checkout, which
is the point — previously a worktree silently shared the production data directory.

Not covered here yet, because they are not checkout-relative and need configuration:
third-party binaries inside a virtualenv (``yt-dlp``) and the MinIO data directory.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

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


#: Env var listing the mounts the admin panel reports usage for, as ``name=path`` pairs.
DISK_ROOTS_ENV = "ADMIN_DISK_ROOTS"


def disk_usage_roots() -> list[tuple[str, str]]:
    """Mounts to report disk usage for, as ``(name, path)`` pairs.

    Deployments differ in where their data lives — the production box keeps models on a
    separate mount, a container has none of them. ``ADMIN_DISK_ROOTS`` takes a comma
    separated ``name=path`` list (``root=/,data=/mnt/data``); the default reports the root
    filesystem and the running user's home, which exist everywhere and need no literal.

    Malformed entries are skipped with a warning instead of failing the panel: a typo in a
    monitoring setting must not take the health endpoint down.
    """
    raw = os.getenv(DISK_ROOTS_ENV, "").strip()
    if not raw:
        return [("root", "/"), ("home", str(Path.home()))]

    pairs: list[tuple[str, str]] = []
    for entry in raw.split(","):
        name, _, path = entry.strip().partition("=")
        name, path = name.strip(), path.strip()
        if not name or not path:
            logger.warning(
                "⚠️ %s: skipping malformed entry %r (expected name=path)",
                DISK_ROOTS_ENV,
                entry,
            )
            continue
        pairs.append((name, path))
    return pairs or [("root", "/"), ("home", str(Path.home()))]


#: Env var pointing at an extra LoRA root outside the checkout.
LORA_SSD_DIR_ENV = "LORA_SSD_DIR"


def lora_ssd_dir() -> Path | None:
    """The extra LoRA mount, or ``None`` when nothing is configured.

    LoRA libraries are large and deployments often keep them on a separate mount, so
    the location is configuration. Unset means the checkout's own
    ``ComfyUI/models/loras`` is the only source, which is what a fresh clone and a
    worktree should get anyway.
    """
    raw = os.getenv(LORA_SSD_DIR_ENV, "").strip()
    return Path(raw) if raw else None


def lora_roots() -> list[Path]:
    """Every directory LoRAs are read from: the checkout first, then the mount."""
    roots = [comfyui_loras_dir()]
    extra = lora_ssd_dir()
    if extra is not None:
        roots.append(extra)
    return roots
