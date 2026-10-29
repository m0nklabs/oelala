"""Rule 6 ratchet: no absolute host paths creeping back into the backend.

`src/backend/` used to hardcode ``/home/<user>/oelala/...`` in 28 places — the env file, the
data files, the logs, the ComfyUI output and LoRA directories. Those are **write** paths, so
a backend started from another checkout silently rewrote the first deployment's operational
data. They are being converted to `project_paths` in slices; this test is the ratchet that
stops new ones appearing while the rest are worked through.

The allowlist is deliberately an upper bound, not an exact count: converting a path in one
PR must not break another PR that is in flight. When you remove a path, lower its number.
"""

import re
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).parent.parent / "src" / "backend"

#: Remaining absolute host paths per module. Shrink this; never grow it.
ALLOWED = {
    "app.py": 3,  # 2x venv yt-dlp binary + the env file (fixed in #225, pending merge)
    "admin_api.py": 1,  # filesystem-browser root listing
    "comfyui_client.py": 1,  # MinIO data dir, already env-overridable
    "face_service.py": 1,
    "lora_scanner.py": 1,  # default LoRA path, checkout-relative → next slice
    "test_h264_encode.py": 1,
}

_HOST_PATH = re.compile(r"""/home/[A-Za-z0-9_.-]+""")


def _counts() -> dict[str, int]:
    counts = {}
    for path in sorted(BACKEND_DIR.glob("*.py")):
        if path.name == "project_paths.py":
            # Its docstring names the pattern it documents; that is not a path literal.
            continue
        found = len(_HOST_PATH.findall(path.read_text(errors="ignore")))
        if found:
            counts[path.name] = found
    return counts


def test_no_new_absolute_host_paths():
    counts = _counts()
    grew = {name: (n, ALLOWED.get(name, 0)) for name, n in counts.items() if n > ALLOWED.get(name, 0)}
    assert not grew, (
        "new absolute host path in the backend — resolve it from project_paths instead "
        f"(found, allowed): {grew}\ncurrent counts: {counts}"
    )


def test_allowlist_has_no_stale_entries():
    counts = _counts()
    stale = [name for name in ALLOWED if name not in counts]
    assert not stale, (
        f"these modules no longer contain absolute host paths — drop them from ALLOWED: {stale}"
    )


def test_project_paths_resolve_inside_this_checkout():
    """The production guarantee: on this checkout every helper equals the old literal."""
    import project_paths as paths

    root = Path(paths.__file__).resolve().parents[2]
    assert paths.checkout_root() == root
    assert paths.data_dir() == root / "data"
    assert paths.logs_dir() == root / "logs"
    assert paths.workflows_dir() == root / "workflows"
    assert paths.frontend_dir() == root / "src" / "frontend"
    assert paths.comfyui_dir() == root / "ComfyUI"
    assert paths.comfyui_output_dir() == root / "ComfyUI" / "output"
    assert paths.comfyui_loras_dir() == root / "ComfyUI" / "models" / "loras"
    assert paths.media_generated_dir() == root / "media" / "generated"


@pytest.mark.parametrize(
    "module_name,attribute,expected",
    [
        ("app", "FRONTEND_DIR", ("src", "frontend")),
        ("app", "COMFYUI_OUTPUT_DIR", ("ComfyUI", "output")),
        ("app", "CLOUD_JOBS_FILE", ("data", "cloud_jobs.json")),
        ("app", "GENERATION_STATS_FILE", ("data", "generation_stats.json")),
        ("app", "AI_SETTINGS_FILE", ("data", "ai_settings.json")),
        ("app", "CLOUD_LOGS_DIR", ("logs", "cloud")),
        ("admin_api", "MEDIA_GENERATED_DIR", ("media", "generated")),
        ("admin_api", "COMFYUI_OUTPUT_DIR", ("ComfyUI", "output")),
        ("admin_api", "AI_SETTINGS_FILE", ("data", "ai_settings.json")),
    ],
)
def test_converted_constants_point_into_the_checkout(module_name, attribute, expected):
    import importlib

    import project_paths as paths

    module = importlib.import_module(module_name)
    value = Path(getattr(module, attribute))
    assert value == paths.checkout_root().joinpath(*expected)


def test_generation_times_file_is_checkout_relative():
    import project_paths as paths
    from websocket_handler import _GENERATION_TIMES_FILE

    assert Path(_GENERATION_TIMES_FILE) == paths.data_dir() / "generation_times.json"
