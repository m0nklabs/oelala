"""A missing user-media object is a client-facing 404, not a server fault.

``GET /user/media/{type}/{filename}`` used to log every storage NoSuchKey at
ERROR. The client only knows a filename (profiles and history do not persist
the media type), so it legitimately probes a segment the file does not live
under - e.g. a generated ``*_000.png`` requested from ``uploads/``. That is a
normal miss and must not flood the log. A real storage failure keeps ERROR so
it stays visible.
"""

import importlib
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from minio.error import S3Error

backend_dir = Path(__file__).parent.parent / "src" / "backend"
sys.path.insert(0, str(backend_dir))


def _import_app_without_startup_recovery():
    """Import the backend app without letting startup recovery touch live state."""
    try:
        import face_train_service
    except ImportError:
        return importlib.import_module("app")

    with patch.object(face_train_service, "recover_stuck_jobs", return_value=0):
        return importlib.import_module("app")


backend = _import_app_without_startup_recovery()

USER_ID = "11111111-2222-3333-4444-555555555555"
MISSING = "8c67168f-0ab5-40a5-82a8-f66192442dba_cloud_minimaxh3_000.png"


def _no_such_key():
    # minio's S3Error signature is (response, code, message, resource, request_id,
    # host_id, bucket_name, object_name) - the response comes first.
    return S3Error(
        None,
        "NoSuchKey",
        "The specified key does not exist.",
        f"/oelala-users/users/{USER_ID}/uploads/{MISSING}",
        "req-1",
        "host-1",
        "oelala-users",
        f"users/{USER_ID}/uploads/{MISSING}",
    )


@pytest.fixture
def media_client(monkeypatch):
    storage = MagicMock()
    monkeypatch.setattr(backend, "get_storage_client", lambda: storage)
    app = backend.app
    app.dependency_overrides[backend.get_current_user] = lambda: SimpleNamespace(
        id=USER_ID
    )
    try:
        yield SimpleNamespace(
            client=TestClient(app, raise_server_exceptions=False), storage=storage
        )
    finally:
        app.dependency_overrides.clear()


def test_missing_object_warns_and_returns_404(media_client, caplog):
    media_client.storage.stream_with_metadata.side_effect = _no_such_key()
    media_client.storage.get_user_media.side_effect = _no_such_key()

    with caplog.at_level(logging.DEBUG):
        response = media_client.client.get(f"/user/media/uploads/{MISSING}")

    assert response.status_code == 404
    media_records = [
        record
        for record in caplog.records
        if "user media" in record.getMessage().lower()
    ]
    assert media_records, "the route must log the miss"
    assert all(record.levelno == logging.WARNING for record in media_records)
    assert not [
        record
        for record in caplog.records
        if record.levelno >= logging.ERROR and "media" in record.getMessage().lower()
    ]


def test_real_storage_failure_still_logs_error(media_client, caplog):
    media_client.storage.stream_with_metadata.side_effect = RuntimeError("minio down")
    media_client.storage.get_user_media.side_effect = RuntimeError("minio down")

    with caplog.at_level(logging.DEBUG):
        response = media_client.client.get("/user/media/uploads/whatever.png")

    assert response.status_code == 404
    assert any(
        record.levelno == logging.ERROR and "user media" in record.getMessage().lower()
        for record in caplog.records
    ), "a real storage failure must stay at ERROR"
