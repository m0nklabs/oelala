"""Configuration-driven paths: yt-dlp and the MinIO data directory.

Both used to name a specific host layout (a venv path, a home directory). They now resolve from
configuration with a sane default, which is what AGENTS.md rule 6 asks for — and these tests pin
the resolution order so a future edit cannot quietly reorder it.
"""

import sys
from pathlib import Path


def _app_module():
    import app

    return app


def test_yt_dlp_override_wins(monkeypatch):
    monkeypatch.setenv("YT_DLP_PATH", "/custom/bin/yt-dlp")
    assert _app_module().resolve_yt_dlp() == "/custom/bin/yt-dlp"


def test_yt_dlp_falls_back_to_path(monkeypatch):
    monkeypatch.delenv("YT_DLP_PATH", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/yt-dlp")
    assert _app_module().resolve_yt_dlp() == "/usr/local/bin/yt-dlp"


def test_yt_dlp_falls_back_to_the_running_interpreter(monkeypatch):
    monkeypatch.delenv("YT_DLP_PATH", raising=False)
    monkeypatch.setattr("shutil.which", lambda name: None)
    expected = str(Path(sys.executable).parent / "yt-dlp")
    assert _app_module().resolve_yt_dlp() == expected


def test_minio_data_dir_override(monkeypatch, tmp_path):
    import comfyui_client

    monkeypatch.setenv("MINIO_DATA_DIR", str(tmp_path / "minio"))
    assert comfyui_client.minio_data_dir() == tmp_path / "minio"


def test_minio_data_dir_defaults_to_home(monkeypatch):
    import comfyui_client

    monkeypatch.delenv("MINIO_DATA_DIR", raising=False)
    assert comfyui_client.minio_data_dir() == Path.home() / "minio-data"
