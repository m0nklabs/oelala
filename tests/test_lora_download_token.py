"""
Tests for signed LoRA download URLs (cloud worker fetch path).

Cloud workers fetch user LoRAs from the backend's /loras/download endpoint,
which requires an HMAC-signed ?token= query parameter. build_lora_download_list
must sign local-LoRA URLs with the exact same scheme the endpoint verifies
(RUNPOD_API_KEY key, SHA-256 HMAC over the resolved filename, first 32 hex
chars). Regression: URLs were built without the token, so every cloud LoRA
download failed with HTTP 422.
"""

import hashlib
import hmac
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "backend"))

# Other test modules import app.py, which loads .env and therefore populates the
# mirror/base-url environment. Pin the values this module asserts on so the
# results do not depend on test execution order.
TEST_BACKEND_URL = "https://backend.test.invalid"
TEST_RUNPOD_KEY = "test-runpod-key"


@pytest.fixture(autouse=True)
def _hermetic_lora_env(monkeypatch, tmp_path):
    for name in (
        "LORA_HF_FLAT_MIRROR_REPO",
        "LORA_HF_MIRROR_REPO",
        "LORA_HF_MIRROR_TOKEN",
        "HF_LORA_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    # Never read the real generated index: tests pin their own file or none.
    monkeypatch.setenv("LORA_SOURCE_INDEX", str(tmp_path / "absent-index.json"))
    monkeypatch.setenv("BACKEND_PUBLIC_URL", TEST_BACKEND_URL)
    monkeypatch.setenv("RUNPOD_API_KEY", TEST_RUNPOD_KEY)


def _write_index(tmp_path, repos, name="index.json"):
    """Write a source index; `repos` is [(repo, [basenames])] stored flat."""
    path = tmp_path / name
    path.write_text(
        json.dumps({
            "repos": [{"repo": r, "files": {n: n for n in names}} for r, names in repos]
        }),
        encoding="utf-8",
    )
    return path

from generation.lora_utils import build_lora_download_list, lora_download_token  # noqa: E402


def _endpoint_style_token(filename: str) -> str:
    """Independent reimplementation of app.py's endpoint verification scheme."""
    key = os.getenv("RUNPOD_API_KEY", "fallback-lora-key").encode()
    return hmac.new(key, filename.encode(), hashlib.sha256).hexdigest()[:32]


def test_token_matches_endpoint_scheme():
    name = "minimax-h3/example.safetensors"
    assert lora_download_token(name) == _endpoint_style_token(name)
    assert len(lora_download_token(name)) == 32
    assert all(c in "0123456789abcdef" for c in lora_download_token(name))


def test_local_lora_url_is_signed():
    resolved = "minimax-h3/example.safetensors"
    with patch(
        "generation.lora_utils.resolve_lora_path",
        return_value=(Path("/tmp/loras") / resolved, resolved),
    ):
        downloads = build_lora_download_list([{"name": "example", "strength": 1.0}])
    assert len(downloads) == 1
    entry = downloads[0]
    assert entry["filename"] == resolved
    expected_url = (
        TEST_BACKEND_URL +
        f"/loras/download/{resolved}?token={_endpoint_style_token(resolved)}"
    )
    assert entry["url"] == expected_url
    assert "hf_token" not in entry


def test_hf_source_url_unchanged():
    resolved = "hf_lora.safetensors"
    with patch(
        "generation.lora_utils.resolve_lora_path",
        return_value=(Path("/tmp/loras") / resolved, resolved),
    ):
        downloads = build_lora_download_list(
            [{"name": "hf_lora"}],
            hf_sources={resolved: {"repo": "org/repo", "path": f"loras/{resolved}"}},
            hf_token="hf_test_token",
        )
    assert downloads == [
        {
            "filename": resolved,
            "url": f"https://huggingface.co/org/repo/resolve/main/loras/{resolved}",
            "fallback_url": (
                TEST_BACKEND_URL +
                f"/loras/download/{resolved}?token={_endpoint_style_token(resolved)}"
            ),
            "hf_token": "hf_test_token",
        }
    ]


def test_missing_lora_is_skipped():
    with patch(
        "generation.lora_utils.resolve_lora_path",
        return_value=(None, None),
    ):
        downloads = build_lora_download_list([{"name": "ghost_lora"}])
    assert downloads == []


def test_mirror_repo_used_with_fallback():
    """Mirror repo env: HF URL primary, signed self-hosted URL as fallback."""
    resolved = "minimax-h3/example.safetensors"
    env = {
        "LORA_HF_MIRROR_REPO": "m0nk111/oelala-loras",
        "HF_LORA_TOKEN": "hf_test_token",
    }
    with patch(
        "generation.lora_utils.resolve_lora_path",
        return_value=(Path("/tmp/loras") / resolved, resolved),
    ), patch.dict(os.environ, env):
        downloads = build_lora_download_list([{"name": "example"}])
    assert downloads == [
        {
            "filename": resolved,
            "url": f"https://huggingface.co/m0nk111/oelala-loras/resolve/main/{resolved}",
            "fallback_url": (
                TEST_BACKEND_URL +
                f"/loras/download/{resolved}?token={_endpoint_style_token(resolved)}"
            ),
            "hf_token": "hf_test_token",
        }
    ]


def test_explicit_hf_source_wins_over_mirror():
    """Curated hf_sources entries take precedence over the mirror repo."""
    resolved = "hf_lora.safetensors"
    env = {"LORA_HF_MIRROR_REPO": "m0nk111/oelala-loras"}
    with patch(
        "generation.lora_utils.resolve_lora_path",
        return_value=(Path("/tmp/loras") / resolved, resolved),
    ), patch.dict(os.environ, env):
        downloads = build_lora_download_list(
            [{"name": "hf_lora"}],
            hf_sources={resolved: {"repo": "org/other", "path": f"loras/{resolved}"}},
            hf_token="hf_curated_token",
        )
    entry = downloads[0]
    assert entry["url"] == f"https://huggingface.co/org/other/resolve/main/loras/{resolved}"
    assert entry["hf_token"] == "hf_curated_token"
    # fallback to the signed self-hosted URL is still attached
    assert entry["fallback_url"] == (
        TEST_BACKEND_URL +
        f"/loras/download/{resolved}?token={_endpoint_style_token(resolved)}"
    )


def test_flat_mirror_uses_basename_and_fallback():
    """Flat mirror: public repo, files by basename, no token, signed fallback."""
    resolved = "minimax-h3/Vagina_minimax-h3_epoch20.safetensors"
    env = {"LORA_HF_FLAT_MIRROR_REPO": "Serenak/chilloutmix"}
    with patch(
        "generation.lora_utils.resolve_lora_path",
        return_value=(Path("/tmp/loras") / resolved, resolved),
    ), patch.dict(os.environ, env):
        downloads = build_lora_download_list([{"name": "whatever"}])
    assert downloads == [
        {
            "filename": resolved,
            "url": (
                "https://huggingface.co/Serenak/chilloutmix/resolve/main/"
                "Vagina_minimax-h3_epoch20.safetensors"
            ),
            "fallback_url": (
                TEST_BACKEND_URL +
                f"/loras/download/{resolved}?token={_endpoint_style_token(resolved)}"
            ),
        }
    ]


def test_flat_mirror_encodes_special_filenames():
    """Basenames with spaces are percent-encoded for the flat mirror URL."""
    resolved = "ltx/LTX-2.3 - Ahegao Face v1.safetensors"
    env = {"LORA_HF_FLAT_MIRROR_REPO": "Serenak/chilloutmix"}
    with patch(
        "generation.lora_utils.resolve_lora_path",
        return_value=(Path("/tmp/loras") / resolved, resolved),
    ), patch.dict(os.environ, env):
        downloads = build_lora_download_list([{"name": "whatever"}])
    url = downloads[0]["url"]
    assert url == (
        "https://huggingface.co/Serenak/chilloutmix/resolve/main/"
        "LTX-2.3%20-%20Ahegao%20Face%20v1.safetensors"
    )


def test_flat_mirror_wins_over_subdir_mirror():
    """Priority: curated hf_sources > third-party flat mirror > our subdir mirror.

    The flat mirror is a third-party public dump and is preferred on purpose:
    the download traffic stays on that account instead of ours. Our own mirror
    only serves what the dump lacks.
    """
    resolved = "a/file.safetensors"
    env = {
        "LORA_HF_FLAT_MIRROR_REPO": "flat/repo",
        "LORA_HF_MIRROR_REPO": "bomehika/oelala-loras",
        "HF_LORA_TOKEN": "hf_test_token",
    }
    with patch(
        "generation.lora_utils.resolve_lora_path",
        return_value=(Path("/tmp/loras") / resolved, resolved),
    ), patch.dict(os.environ, env):
        downloads = build_lora_download_list([{"name": "file"}])
    assert downloads[0]["url"] == "https://huggingface.co/flat/repo/resolve/main/file.safetensors"
    assert "hf_token" not in downloads[0]  # flat mirror is public
    # our own mirror still rides along as the signed self-hosted fallback
    assert downloads[0]["fallback_url"].startswith(TEST_BACKEND_URL)


def test_subdir_mirror_encodes_spaces_in_path():
    """Layout-preserving mirror paths may contain spaces — percent-encode them."""
    resolved = "wan 2.2/NSFW-22-H-e8.safetensors"
    env = {"LORA_HF_MIRROR_REPO": "bomehika/oelala-loras"}
    with patch(
        "generation.lora_utils.resolve_lora_path",
        return_value=(Path("/tmp/loras") / resolved, resolved),
    ), patch.dict(os.environ, env):
        downloads = build_lora_download_list([{"name": "NSFW-22-H-e8"}])
    assert downloads[0]["url"] == (
        "https://huggingface.co/bomehika/oelala-loras/resolve/main/"
        "wan%202.2/NSFW-22-H-e8.safetensors"
    )


def test_public_mirror_sends_no_token_when_override_is_empty():
    """LORA_HF_MIRROR_TOKEN='' keeps a public mirror anonymous (no token sent)."""
    resolved = "a/file.safetensors"
    env = {
        "LORA_HF_MIRROR_REPO": "bomehika/oelala-loras",
        "LORA_HF_MIRROR_TOKEN": "",
        "HF_LORA_TOKEN": "hf_test_token",
    }
    with patch(
        "generation.lora_utils.resolve_lora_path",
        return_value=(Path("/tmp/loras") / resolved, resolved),
    ), patch.dict(os.environ, env):
        downloads = build_lora_download_list([{"name": "file"}])
    assert "hf_token" not in downloads[0]


def test_no_mirror_single_signed_url_without_fallback():
    """Without mirror env: single signed URL, no fallback (previous behavior)."""
    resolved = "minimax-h3/example.safetensors"
    env = {"LORA_HF_MIRROR_REPO": "", "HF_LORA_TOKEN": ""}
    with patch(
        "generation.lora_utils.resolve_lora_path",
        return_value=(Path("/tmp/loras") / resolved, resolved),
    ), patch.dict(os.environ, env):
        downloads = build_lora_download_list([{"name": "example"}])
    assert downloads == [
        {
            "filename": resolved,
            "url": (
                TEST_BACKEND_URL +
                f"/loras/download/{resolved}?token={_endpoint_style_token(resolved)}"
            ),
        }
    ]


def test_source_index_routes_listed_file_to_third_party_dump(tmp_path, monkeypatch):
    """A file the dump is known to hold goes to the dump, not to our mirror."""
    monkeypatch.setenv(
        "LORA_SOURCE_INDEX",
        str(_write_index(tmp_path, [("third/party", ["listed.safetensors"])])),
    )
    monkeypatch.setenv("LORA_HF_MIRROR_REPO", "bomehika/oelala-loras")
    monkeypatch.setenv("LORA_HF_FLAT_MIRROR_REPO", "Serenak/chilloutmix")
    resolved = "some/dir/listed.safetensors"
    with patch(
        "generation.lora_utils.resolve_lora_path",
        return_value=(Path("/tmp/loras") / resolved, resolved),
    ):
        downloads = build_lora_download_list([{"name": "listed"}])
    assert downloads[0]["url"] == (
        "https://huggingface.co/third/party/resolve/main/listed.safetensors"
    )
    assert downloads[0]["fallback_url"].startswith(TEST_BACKEND_URL)


def test_source_index_sends_unlisted_file_to_our_mirror(tmp_path, monkeypatch):
    """A file the dump does not hold skips it entirely and uses our own mirror."""
    monkeypatch.setenv(
        "LORA_SOURCE_INDEX",
        str(_write_index(tmp_path, [("third/party", ["listed.safetensors"])])),
    )
    monkeypatch.setenv("LORA_HF_MIRROR_REPO", "bomehika/oelala-loras")
    monkeypatch.setenv("LORA_HF_FLAT_MIRROR_REPO", "Serenak/chilloutmix")
    resolved = "some/dir/other.safetensors"
    with patch(
        "generation.lora_utils.resolve_lora_path",
        return_value=(Path("/tmp/loras") / resolved, resolved),
    ):
        downloads = build_lora_download_list([{"name": "other"}])
    assert downloads[0]["url"] == (
        "https://huggingface.co/bomehika/oelala-loras/resolve/main/some/dir/other.safetensors"
    )
    assert "hf_token" not in downloads[0]  # public mirror, anonymous


def test_source_index_first_matching_dump_wins(tmp_path, monkeypatch):
    """Dump order in the index is the priority order."""
    monkeypatch.setenv(
        "LORA_SOURCE_INDEX",
        str(_write_index(tmp_path, [
            ("first/dump", ["shared.safetensors"]),
            ("second/dump", ["shared.safetensors"]),
        ])),
    )
    resolved = "shared.safetensors"
    with patch(
        "generation.lora_utils.resolve_lora_path",
        return_value=(Path("/tmp/loras") / resolved, resolved),
    ):
        downloads = build_lora_download_list([{"name": "shared"}])
    assert downloads[0]["url"] == (
        "https://huggingface.co/first/dump/resolve/main/shared.safetensors"
    )


def test_sensitive_lora_never_gets_a_public_url(tmp_path, monkeypatch):
    """Excluded files (face-swap, real-person likeness) go straight to the
    signed self-hosted URL, even when a dump and our mirror both exist."""
    monkeypatch.setenv(
        "LORA_SOURCE_INDEX",
        str(_write_index(tmp_path, [("third/party", ["bfs_head_swap_v4.safetensors"])])),
    )
    monkeypatch.setenv("LORA_HF_MIRROR_REPO", "bomehika/oelala-loras")
    monkeypatch.setenv("LORA_HF_FLAT_MIRROR_REPO", "Serenak/chilloutmix")
    resolved = "bfs_head_swap_v4.safetensors"
    with patch(
        "generation.lora_utils.resolve_lora_path",
        return_value=(Path("/tmp/loras") / resolved, resolved),
    ):
        downloads = build_lora_download_list([{"name": "bfs_head_swap_v4"}])
    assert downloads == [
        {
            "filename": resolved,
            "url": (
                TEST_BACKEND_URL +
                f"/loras/download/{resolved}?token={_endpoint_style_token(resolved)}"
            ),
        }
    ]


def test_source_index_uses_recorded_subdir_path(tmp_path, monkeypatch):
    """A dump that keeps files in a subdirectory must still resolve."""
    path = tmp_path / "subdir-index.json"
    path.write_text(json.dumps({"repos": [
        {"repo": "jaysowen/wan2.2-nsfw-loras",
         "files": {"NSFW-22-H-e8.safetensors": "high/NSFW-22-H-e8.safetensors"}}
    ]}), encoding="utf-8")
    monkeypatch.setenv("LORA_SOURCE_INDEX", str(path))
    resolved = "wan 2.2/NSFW-22-H-e8.safetensors"
    with patch(
        "generation.lora_utils.resolve_lora_path",
        return_value=(Path("/tmp/loras") / resolved, resolved),
    ):
        downloads = build_lora_download_list([{"name": "NSFW-22-H-e8"}])
    assert downloads[0]["url"] == (
        "https://huggingface.co/jaysowen/wan2.2-nsfw-loras/resolve/main/"
        "high/NSFW-22-H-e8.safetensors"
    )


def test_source_index_legacy_basenames_format(tmp_path, monkeypatch):
    """An index written in the older basenames-only format keeps working."""
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"repos": [
        {"repo": "third/party", "basenames": ["legacy.safetensors"]}
    ]}), encoding="utf-8")
    monkeypatch.setenv("LORA_SOURCE_INDEX", str(path))
    resolved = "legacy.safetensors"
    with patch(
        "generation.lora_utils.resolve_lora_path",
        return_value=(Path("/tmp/loras") / resolved, resolved),
    ):
        downloads = build_lora_download_list([{"name": "legacy"}])
    assert downloads[0]["url"] == (
        "https://huggingface.co/third/party/resolve/main/legacy.safetensors"
    )


def test_source_index_absent_keeps_flat_mirror_first(monkeypatch):
    """Without an index the legacy chain still applies: flat mirror first."""
    monkeypatch.setenv("LORA_HF_FLAT_MIRROR_REPO", "Serenak/chilloutmix")
    monkeypatch.setenv("LORA_HF_MIRROR_REPO", "bomehika/oelala-loras")
    resolved = "unlisted.safetensors"
    with patch(
        "generation.lora_utils.resolve_lora_path",
        return_value=(Path("/tmp/loras") / resolved, resolved),
    ):
        downloads = build_lora_download_list([{"name": "unlisted"}])
    assert downloads[0]["url"] == (
        "https://huggingface.co/Serenak/chilloutmix/resolve/main/unlisted.safetensors"
    )
