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
def _hermetic_lora_env(monkeypatch):
    for name in (
        "LORA_HF_FLAT_MIRROR_REPO",
        "LORA_HF_MIRROR_REPO",
        "HF_LORA_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BACKEND_PUBLIC_URL", TEST_BACKEND_URL)
    monkeypatch.setenv("RUNPOD_API_KEY", TEST_RUNPOD_KEY)

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
    """Priority: curated hf_sources > flat mirror > subdir mirror."""
    resolved = "a/file.safetensors"
    env = {
        "LORA_HF_FLAT_MIRROR_REPO": "flat/repo",
        "LORA_HF_MIRROR_REPO": "m0nk111/oelala-loras",
        "HF_LORA_TOKEN": "hf_test_token",
    }
    with patch(
        "generation.lora_utils.resolve_lora_path",
        return_value=(Path("/tmp/loras") / resolved, resolved),
    ), patch.dict(os.environ, env):
        downloads = build_lora_download_list([{"name": "file"}])
    assert downloads[0]["url"] == "https://huggingface.co/flat/repo/resolve/main/file.safetensors"
    assert "hf_token" not in downloads[0]  # flat mirror is public


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
