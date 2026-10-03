"""Tests for the public-source exclusion policy (generation/lora_public_policy.py).

Policy: LoRAs that carry face-swap capability or a real person's likeness must
never be uploaded to a PUBLIC HuggingFace mirror. They stay on the private
mirror and in the local store, reachable by cloud workers through the signed
self-hosted download fallback.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "backend"))

from generation.lora_public_policy import (  # noqa: E402
    is_excluded,
    load_exclusions,
    partition,
)

# Local files the shipped exclusion list must hold back.
MUST_EXCLUDE = [
    "bfs_head_swap_v4.safetensors",
    "ltx/head_swap_v3_rank_adaptive_fro_098.safetensors",
    "face_loras/ohwx_lydia.safetensors",
    "igbaddie-PN.safetensors",
    "sounding/JAV_GTJ130_IL_V1.safetensors",
    "sounding/JAV_GTJ130_Pony_V1.safetensors",
]

# Local files that must stay uploadable.
MUST_ALLOW = [
    "ltx/ltxdeepthroat_v01.safetensors",
    "sounding/The_Ultimate_Urethral_Sounding_LORA_for_Pony.safetensors",
    "tattoogirls-PN.safetensors",
    "wan 2.2/NSFW-22-H-e8.safetensors",
    "minimax-h3/Vagina_minimax-h3_epoch20.safetensors",
]


@pytest.fixture()
def patterns():
    loaded = load_exclusions()
    assert loaded, "shipped exclusion list must not be empty"
    return loaded


@pytest.mark.parametrize("rel", MUST_EXCLUDE)
def test_shipped_list_excludes_sensitive_files(rel, patterns):
    assert is_excluded(rel, patterns), f"{rel} must be held back from public repos"


@pytest.mark.parametrize("rel", MUST_ALLOW)
def test_shipped_list_allows_ordinary_files(rel, patterns):
    assert not is_excluded(rel, patterns), f"{rel} must stay uploadable"


def test_partition_splits_without_losing_files(patterns):
    files = {rel: Path("/tmp/loras") / rel for rel in MUST_EXCLUDE + MUST_ALLOW}
    allowed, excluded = partition(files, patterns)
    assert len(allowed) + len(excluded) == len(files)
    assert set(excluded) == set(MUST_EXCLUDE)
    assert set(allowed) == set(MUST_ALLOW)


def test_basename_and_path_patterns_both_match(patterns):
    # Pattern `*head_swap*` matches the basename at any depth.
    assert is_excluded("some/deep/dir/my_head_swap_v9.safetensors", patterns)
    # Pattern `face_loras/*` matches by relative path.
    assert is_excluded("face_loras/anything.safetensors", patterns)


def test_leading_dot_slash_is_normalised(patterns):
    assert is_excluded("./bfs_head_swap_v4.safetensors", patterns)


def test_missing_list_yields_no_patterns(tmp_path):
    assert load_exclusions(tmp_path / "does-not-exist.txt") == []
    assert not is_excluded("bfs_head_swap_v4.safetensors", [])


def test_comments_and_blank_lines_ignored(tmp_path):
    path = tmp_path / "list.txt"
    path.write_text("# comment only\n\n*head_swap*  # inline note\n", encoding="utf-8")
    assert load_exclusions(path) == ["*head_swap*"]
