"""Tests for FastAPI app security helper functions."""

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException


backend_dir = Path(__file__).parent.parent / "src" / "backend"
sys.path.insert(0, str(backend_dir))

from app import (  # noqa: E402
    _find_existing_media_path,
    _normalize_storage_key,
    _safe_child_path,
    _safe_external_id,
    _safe_filename,
    _safe_user_media_type,
    _validate_public_image_url,
    _validate_youtube_url,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("image.png", "image.png"),
        ("generated/video 01.mp4", "generated/video 01.mp4"),
        ("loras/user/model-v1.safetensors", "loras/user/model-v1.safetensors"),
    ],
)
def test_normalize_storage_key_safe_values(value, expected):
    """Safe object keys are preserved."""
    assert _normalize_storage_key(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "",
        "../secret.txt",
        "generated/../../secret.txt",
        "/absolute/path.png",
        "folder\\file.png",
        "bad<script>.png",
    ],
)
def test_normalize_storage_key_rejects_unsafe_values(value):
    """Unsafe object keys are rejected before storage access."""
    with pytest.raises(HTTPException):
        _normalize_storage_key(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("folder/image.png", "image.png"),
        ("folder\\image.png", "image.png"),
        (None, "file"),
    ],
)
def test_safe_filename_uses_valid_basename(value, expected):
    """Filenames are reduced to a safe basename."""
    assert _safe_filename(value) == expected


def test_safe_child_path_stays_inside_base(tmp_path):
    """Child path resolution never escapes the configured base directory."""
    assert _safe_child_path(tmp_path, "image.png") == tmp_path / "image.png"
    with pytest.raises(HTTPException):
        _safe_child_path(tmp_path, "bad<script>.png")


@pytest.mark.parametrize("media_type", ["images", "videos", "audio", "uploads"])
def test_safe_user_media_type_accepts_known_types(media_type):
    """Known user media types are accepted."""
    assert _safe_user_media_type(media_type) == media_type


@pytest.mark.parametrize("media_type", ["../images", "private", "images/extra"])
def test_safe_user_media_type_rejects_unknown_types(media_type):
    """Unknown user media types are rejected."""
    with pytest.raises(HTTPException):
        _safe_user_media_type(media_type)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/image.png",
        "http://127.0.0.1/image.png",
        "http://169.254.169.254/latest/meta-data/",
        "http://user:pass@example.com/image.png",
    ],
)
def test_validate_public_image_url_rejects_ssrf_targets(url):
    """Server-side image fetches reject local and private targets."""
    with pytest.raises(HTTPException):
        _validate_public_image_url(url)


def test_find_existing_media_path_rejects_absolute_escape():
    """Existing-media references cannot point outside known media roots."""
    with pytest.raises(HTTPException):
        _find_existing_media_path("/etc/passwd")


@pytest.mark.parametrize(
    "value",
    [
        "2f4f4e8a-9c6a-4af6-8b92-64a102e6a7b0",
        "cloud_job-123",
        "train:abc_123",
    ],
)
def test_safe_external_id_accepts_path_segments(value):
    """External service IDs are accepted only as compact path segments."""
    assert _safe_external_id(value, "test ID") == value


@pytest.mark.parametrize("value", ["", "../secret", "bad/id", "bad?query"])
def test_safe_external_id_rejects_unsafe_segments(value):
    """External service IDs reject traversal and nested path syntax."""
    with pytest.raises(HTTPException):
        _safe_external_id(value, "test ID")


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ",
        "https://music.youtube.com/watch?v=dQw4w9WgXcQ",
    ],
)
def test_validate_youtube_url_accepts_known_hosts(url):
    """YouTube imports accept only known YouTube hostnames."""
    assert _validate_youtube_url(url) == url


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/watch?v=x",
        "https://youtube.com.evil.example/watch?v=x",
        "https://user:pass@youtube.com/watch?v=x",
    ],
)
def test_validate_youtube_url_rejects_unsafe_hosts(url):
    """YouTube imports reject non-YouTube and credentialed URLs."""
    with pytest.raises(HTTPException):
        _validate_youtube_url(url)
