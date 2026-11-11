"""The admin panel's disk-usage mounts come from configuration, not from literals."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "backend"))

import project_paths  # noqa: E402


def test_default_reports_root_and_home(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(project_paths.DISK_ROOTS_ENV, raising=False)
    roots = project_paths.disk_usage_roots()
    assert roots[0] == ("root", "/")
    assert roots[1][0] == "home"
    assert roots[1][1] == str(Path.home())
    # No environment-specific mount is assumed: the production box's extra mount must be
    # configuration, or a second deployment silently reports the first one's filesystem.
    assert all(path == "/" or path == str(Path.home()) for _, path in roots)


def test_configured_roots_win(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        project_paths.DISK_ROOTS_ENV, "root=/,data=/srv/data,scratch=/mnt/scratch"
    )
    assert project_paths.disk_usage_roots() == [
        ("root", "/"),
        ("data", "/srv/data"),
        ("scratch", "/mnt/scratch"),
    ]


def test_whitespace_is_tolerated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(project_paths.DISK_ROOTS_ENV, " root = / , data = /srv/data ")
    assert project_paths.disk_usage_roots() == [("root", "/"), ("data", "/srv/data")]


def test_malformed_entries_are_skipped_not_fatal(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(
        project_paths.DISK_ROOTS_ENV, "root=/,bogus,=nopath,data=/srv/data"
    )
    roots = project_paths.disk_usage_roots()
    assert roots == [("root", "/"), ("data", "/srv/data")]
    # A broken monitoring setting warns; it must not take the health endpoint down.
    assert any(
        "skipping malformed entry" in record.message for record in caplog.records
    )


def test_empty_value_falls_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(project_paths.DISK_ROOTS_ENV, "   ")
    assert project_paths.disk_usage_roots() == [
        ("root", "/"),
        ("home", str(Path.home())),
    ]


def test_all_malformed_falls_back_to_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(project_paths.DISK_ROOTS_ENV, "bogus,,=nopath")
    assert project_paths.disk_usage_roots() == [
        ("root", "/"),
        ("home", str(Path.home())),
    ]
