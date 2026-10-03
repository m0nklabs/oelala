"""Policy for LoRA files that must never be served from a public source.

`scripts/lora_public_exclusions.txt` lists files that carry face-swap capability
or a real person's likeness. They stay on the private mirror and in the local
store, and cloud workers fetch them through the signed self-hosted download.

Single source of truth for both the upload side (`scripts/sync_loras_hf.py`,
which must not push them to a public repo) and the download side
(`build_lora_download_list()`, which must not route them to a public URL).
"""

from __future__ import annotations

import os
from fnmatch import fnmatch
from pathlib import Path

# `<repo>/scripts/lora_public_exclusions.txt`
DEFAULT_EXCLUSIONS_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "lora_public_exclusions.txt"
)


def exclusions_path() -> Path:
    """`LORA_PUBLIC_EXCLUSIONS` if set, else the shipped list."""
    configured = os.getenv("LORA_PUBLIC_EXCLUSIONS", "")
    return Path(configured) if configured else DEFAULT_EXCLUSIONS_PATH


def load_exclusions(path: str | Path | None = None) -> list[str]:
    """Read exclusion globs, ignoring blank lines and `#` comments."""
    source = Path(path) if path else exclusions_path()
    if not source.exists():
        return []
    patterns: list[str] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        entry = line.split("#", 1)[0].strip()
        if entry:
            patterns.append(entry)
    return patterns


def is_excluded(rel_path: str, patterns: list[str] | None = None) -> bool:
    """True when `rel_path` matches any glob, by full relative path or basename."""
    globs = load_exclusions() if patterns is None else patterns
    rel = str(rel_path).replace("\\", "/").lstrip("./")
    name = rel.rsplit("/", 1)[-1]
    return any(fnmatch(rel, pattern) or fnmatch(name, pattern) for pattern in globs)


def partition(files: dict, patterns: list[str]) -> tuple[dict, dict]:
    """Split `{relpath: path}` into `(allowed, excluded)`, preserving order."""
    allowed: dict = {}
    excluded: dict = {}
    for rel, path in files.items():
        target = excluded if is_excluded(rel, patterns) else allowed
        target[rel] = path
    return allowed, excluded
