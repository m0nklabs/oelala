"""Tests for the third-party LoRA source index builder.

A dump copy is only a safe substitute when its byte size matches a local file of
the same name: measured dumps carry same-named files with different byte sizes
(different revisions), and serving one of those would silently change output.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from build_lora_source_index import (  # noqa: E402
    configured_dumps,
    local_sizes,
    verified_files,
)


def test_only_size_matching_files_are_accepted():
    remote = {
        "a.safetensors": [("a.safetensors", 100)],
        "b.safetensors": [("b.safetensors", 200)],
        "c.safetensors": [("c.safetensors", 0)],
    }
    local = {"a.safetensors": {100}, "b.safetensors": {999}}
    accepted, rejected = verified_files(remote, local)
    assert accepted == {"a.safetensors": "a.safetensors"}
    assert rejected == ["b.safetensors", "c.safetensors"]


def test_unknown_basename_is_rejected():
    accepted, rejected = verified_files({"stranger.safetensors": [("stranger.safetensors", 42)]}, {})
    assert accepted == {}
    assert rejected == ["stranger.safetensors"]


def test_subdirectory_path_is_recorded():
    """Dumps that keep files in subdirs must resolve, so the path is kept."""
    remote = {"nsfw.safetensors": [("high/nsfw.safetensors", 500)]}
    accepted, _ = verified_files(remote, {"nsfw.safetensors": {500}})
    assert accepted == {"nsfw.safetensors": "high/nsfw.safetensors"}


def test_ambiguous_basename_matches_any_local_size():
    """`high_noise_model.safetensors` exists twice locally; either size counts."""
    remote = {"high_noise_model.safetensors": [("v1/high_noise_model.safetensors", 200)]}
    local = {"high_noise_model.safetensors": {100, 200}}
    accepted, _ = verified_files(remote, local)
    assert accepted == {"high_noise_model.safetensors": "v1/high_noise_model.safetensors"}


def test_local_sizes_collects_nested_files(tmp_path):
    (tmp_path / "deepthroat").mkdir()
    (tmp_path / "deepthroat" / "a.safetensors").write_bytes(b"x" * 11)
    (tmp_path / "b.safetensors").write_bytes(b"y" * 7)
    (tmp_path / "notes.txt").write_text("ignored", encoding="utf-8")
    sizes = local_sizes([tmp_path])
    assert sizes == {"a.safetensors": {11}, "b.safetensors": {7}}


def test_local_sizes_ignores_missing_root(tmp_path):
    assert local_sizes([tmp_path / "nope"]) == {}


def test_configured_dumps_prefers_explicit_list(monkeypatch):
    monkeypatch.setenv("LORA_THIRD_PARTY_DUMPS", "a/b, c/d")
    monkeypatch.setenv("LORA_HF_FLAT_MIRROR_REPO", "ignored/repo")
    assert configured_dumps({}) == ["a/b", "c/d"]


def test_configured_dumps_falls_back_to_flat_mirror(monkeypatch):
    monkeypatch.delenv("LORA_THIRD_PARTY_DUMPS", raising=False)
    monkeypatch.delenv("LORA_HF_FLAT_MIRROR_REPO", raising=False)
    assert configured_dumps({"LORA_HF_FLAT_MIRROR_REPO": "Serenak/chilloutmix"}) == [
        "Serenak/chilloutmix"
    ]
