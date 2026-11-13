"""LoRA roots come from configuration, so no mount is baked into the code."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "backend"))

import project_paths  # noqa: E402


def test_unset_means_the_checkout_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(project_paths.LORA_SSD_DIR_ENV, raising=False)
    assert project_paths.lora_ssd_dir() is None
    assert project_paths.lora_roots() == [project_paths.comfyui_loras_dir()]


def test_configured_mount_is_added_after_the_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(project_paths.LORA_SSD_DIR_ENV, "/srv/lora-library")
    assert project_paths.lora_ssd_dir() == Path("/srv/lora-library")
    assert project_paths.lora_roots() == [
        project_paths.comfyui_loras_dir(),
        Path("/srv/lora-library"),
    ]


def test_blank_value_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(project_paths.LORA_SSD_DIR_ENV, "   ")
    assert project_paths.lora_ssd_dir() is None
    assert len(project_paths.lora_roots()) == 1


def test_the_checkout_root_always_comes_first(monkeypatch: pytest.MonkeyPatch) -> None:
    # A mount may be read-only or absent; the checkout's own directory must stay the
    # primary source, otherwise a machine without the mount sees no LoRAs at all.
    monkeypatch.setenv(project_paths.LORA_SSD_DIR_ENV, "/srv/lora-library")
    roots = project_paths.lora_roots()
    assert roots[0] == project_paths.comfyui_loras_dir()
    assert len(roots) == 2


def test_the_scanner_follows_the_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """lora_scanner builds its roots at import time, so reload it with the env set."""
    monkeypatch.setenv("LORA_SSD_DIR", "/srv/lora-library")
    import lora_scanner

    reloaded = importlib.reload(lora_scanner)
    try:
        assert reloaded.LORA_DIRS == [
            project_paths.comfyui_loras_dir(),
            Path("/srv/lora-library"),
        ]
    finally:
        monkeypatch.delenv("LORA_SSD_DIR", raising=False)
        importlib.reload(lora_scanner)


def test_lora_utils_treats_the_mount_as_optional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scan loop iterates `(LORA_DIR, LORA_SSD_DIR)`, so None must be filtered."""
    import generation.lora_utils as lora_utils

    monkeypatch.delenv("LORA_SSD_DIR", raising=False)
    reloaded = importlib.reload(lora_utils)
    try:
        assert reloaded.LORA_SSD_DIR is None
        # The default root is the checkout's own directory, not a host path.
        assert reloaded.LORA_DIR == project_paths.comfyui_loras_dir()
        # And the resolve loop must not crash on the absent entry: it filters None,
        # so a name that exists nowhere simply resolves to (None, None).
        assert reloaded.resolve_lora_path("does-not-exist.safetensors") == (None, None)
    finally:
        importlib.reload(lora_utils)
