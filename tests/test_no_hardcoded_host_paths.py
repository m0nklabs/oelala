"""Rule 6 ratchet: no absolute host paths creeping back into the backend.

`src/backend/` used to hardcode ``/home/<user>/oelala/...`` in 28 places — the env file, the
data files, the logs, the ComfyUI output and LoRA directories. Those are **write** paths, so
a backend started from another checkout silently rewrote the first deployment's operational
data. They are being converted to `project_paths` in slices; this test is the ratchet that
stops new ones appearing while the rest are worked through.

The scan is recursive and keyed by path relative to ``src/backend/``, so a literal in a
subpackage counts, and two modules with the same basename cannot hide each other. No module is
exempt — the central path helper is included, so a literal cannot take shelter in it.

The allowlist is deliberately an upper bound, not an exact count: converting a path in one
PR must not break another PR that is in flight. When you remove a path, lower its number.
"""

import re
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).parent.parent / "src" / "backend"

#: Remaining absolute host paths per module. Shrink this; never grow it.
ALLOWED = {
    # The /home/ generation is gone since #239 and #252. What is left is the LoRA
    # library location, which slice 2c (#256) turns into LORA_SSD_DIR configuration:
    # when that PR lands, this allowlist becomes empty. Until then these four are
    # frozen - a new absolute path anywhere else in the backend fails this test.
    "app.py": 2,
    "lora_scanner.py": 1,
    "generation/lora_utils.py": 1,
}

# Two arms, deliberately:
#   1. quoted absolute paths under any of the rule-6 roots. Quoted only, so a prose
#      mention (a docstring saying "files are streamed from the mount") is not a
#      violation - the rule targets code that *uses* a host path. The old /home/-only
#      regex let /mnt/ssd/loras hide in five places for weeks; this arm covers the
#      whole class.
#   2. the old /home/ behaviour verbatim, unquoted included, because a command string
#      like "mkdir -p /home/<user>/oelala" or an unquoted value in a template must
#      still fail exactly as it did before the widening. The negative lookbehind keeps
#      a quoted /home/... from counting twice (arm 1 already has it).
_HOST_PATH = re.compile(
    r"""(?:["'])(?:/(?:home|mnt|srv|opt|data|var|usr/local)/[A-Za-z0-9_.-]+)"""
    r"""|(?<!["'])/home/[A-Za-z0-9_.-]+"""
)


def _counts() -> dict[str, int]:
    counts = {}
    for path in sorted(BACKEND_DIR.rglob("*.py")):
        found = len(_HOST_PATH.findall(path.read_text(errors="ignore")))
        if found:
            counts[path.relative_to(BACKEND_DIR).as_posix()] = found
    return counts


def test_no_new_absolute_host_paths():
    counts = _counts()
    grew = {
        name: (n, ALLOWED.get(name, 0))
        for name, n in counts.items()
        if n > ALLOWED.get(name, 0)
    }
    assert not grew, (
        "new absolute host path in the backend — resolve it from project_paths instead "
        f"(found, allowed): {grew}\ncurrent counts: {counts}"
    )


#: Committed configuration templates are documentation, and rule 6 covers those too: a real
#: host path in an example file is copied into someone's deployment.
TEMPLATES = [Path(__file__).parent.parent / ".env.example"]


def test_no_absolute_host_paths_in_committed_templates():
    offenders = {}
    for template in TEMPLATES:
        if not template.is_file():
            continue
        found = _HOST_PATH.findall(template.read_text(errors="ignore"))
        if found:
            offenders[template.name] = found
    assert not offenders, (
        "absolute host path in a committed template — use a placeholder such as "
        f"/path/to/... : {offenders}"
    )


def _matches(text: str) -> list[str]:
    """The matched paths, quotes stripped for readability in assertions."""
    return [m.group(0).strip("'\"") for m in _HOST_PATH.finditer(text)]


def test_matcher_regressions():
    """The widening must not lose a single case the old /home/-only regex caught.

    Both arms stop at the first slash after the root, exactly as the original
    /home/-only regex did - a deeper path still trips the match at its first segment.
    """
    # Unquoted in a template value - the old regex caught this, arm 2 must too.
    assert _matches("OELALA_ROOT=/home/flip/oelala") == ["/home/flip"]
    # Unquoted inside a command string.
    assert _matches("run: mkdir -p /home/flip/oelala/data") == ["/home/flip"]
    assert _matches("exec /home/flip/oelala/bin/x --flag") == ["/home/flip"]
    # Quoted in code - arm 1, the new class (both arms stop at the first slash after
    # the root; catching the root segment is what trips the test).
    assert _matches('ROOT = Path("/mnt/ssd/loras")') == ["/mnt/ssd"]
    assert _matches("PATH='/srv/data/cache'") == ["/srv/data"]
    # A quoted /home counts once, not twice (both arms could match it).
    assert _matches('BASE = "/home/flip/oelala"') == ["/home/flip"]
    # Prose is not a violation for the new roots, and a placeholder is not a hit.
    assert _matches("Files are streamed from /mnt/ssd/loras/.") == []
    assert _matches("# MINIO_DATA_DIR=/path/to/minio-data") == []


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
