"""Regression tests: cloud (RunPod) outputs must reach the user's media library.

``GET /user/media`` only lists ``storage.list_user_media(user.id)``, so a cloud
output that never reaches user-scoped storage is invisible in My Media even
though the job reports "completed". The 2026-10 fix therefore carried the
authenticated ``user_id`` from the adapter into the RunPod submission, and the
status handler registers that id before uploading the finished output through
``comfyui.on_job_complete_async`` (which calls ``storage.put_user_media``).

These tests pin that contract at the status-handler level, without network,
MinIO, Supabase or a GPU:

* a completed job uploads to the *authenticated user's* storage and reports
  ``storage_path`` plus a signed URL;
* a job whose user-scoped upload fails stays ``running`` so the poller retries —
  "completed" is never reported for media the user cannot see;
* a job without an owner never reaches "completed", i.e. there is no silent
  placeholder ("adapter") owner that would strand the file outside My Media.
"""

import base64
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


backend_dir = Path(__file__).parent.parent / "src" / "backend"
sys.path.insert(0, str(backend_dir))


def _import_app_without_startup_recovery():
    """Import the backend app without letting startup recovery touch live state.

    ``app`` calls ``face_train_service.recover_stuck_jobs()`` while it is being
    imported. That function marks every ``running`` training job as ``failed`` and
    rewrites the job index, so merely collecting this module in an operational
    checkout would corrupt someone's active training job. Stub the call out first.

    ``app`` already tolerates a missing ``face_train_service``, so if the module is
    not importable here we simply import ``app`` (and there is nothing to stub).
    """
    try:
        import face_train_service
    except ImportError:
        return importlib.import_module("app")

    with patch.object(face_train_service, "recover_stuck_jobs", return_value=0):
        return importlib.import_module("app")


backend = _import_app_without_startup_recovery()


USER_ID = "11111111-2222-3333-4444-555555555555"
VIDEO_BYTES = b"\x00\x00\x00\x18ftypmp42" + b"x" * 64
USER_STORAGE_PATH = f"users/{USER_ID}/videos/cloud_out.mp4"


def _completed_runpod_job():
    """Minimal RunPod job object shaped like the real COMPLETED response."""
    return SimpleNamespace(
        status="COMPLETED",
        output={
            "files": [
                {
                    "data": base64.b64encode(VIDEO_BYTES).decode(),
                    "filename": "cloud_out.mp4",
                    "type": "video/mp4",
                }
            ],
            "logs": "worker finished",
        },
        error=None,
        execution_time_s=12.5,
    )


@pytest.fixture
def cloud_env(monkeypatch):
    """Isolate the handler from storage, ComfyUI, persistence and websockets."""
    prompt_id = "test-cloud-prompt-1"
    job_info = {
        "runpod_job_id": "rp-abc123",
        "user_id": USER_ID,
        "model": "minimax_h3",
        "prompt": "a cat surfing",
    }

    storage = MagicMock()
    comfyui = MagicMock()
    comfyui.register_job = MagicMock()
    comfyui.on_job_complete_async = AsyncMock(return_value=USER_STORAGE_PATH)

    runpod = MagicMock()
    runpod.get_job_status = AsyncMock(return_value=_completed_runpod_job())

    monkeypatch.setattr(backend, "get_storage_client", lambda: storage)
    monkeypatch.setattr(backend, "get_comfyui_client", lambda: comfyui)
    monkeypatch.setattr(
        backend,
        "get_signed_media_url",
        lambda path, expires_in=0: f"https://signed.test/{path}",
    )
    monkeypatch.setattr(backend, "_persist_cloud_jobs", lambda: None)
    monkeypatch.setattr(backend, "record_generation_complete", lambda *a, **k: None)
    monkeypatch.setattr(backend, "_save_cloud_logs", lambda *a, **k: None)
    monkeypatch.setattr(backend, "ws_manager", None)
    monkeypatch.setattr(backend, "_cloud_completed_cache", {})
    monkeypatch.setattr(backend, "active_jobs", {prompt_id: {"status": "running"}})
    monkeypatch.setattr(backend, "_runpod", runpod)

    return SimpleNamespace(
        prompt_id=prompt_id,
        job_info=job_info,
        storage=storage,
        comfyui=comfyui,
        runpod=runpod,
    )


async def test_completed_cloud_job_uploads_to_authenticated_user(cloud_env):
    """The finished output is uploaded for the job's user, not a placeholder."""
    result = await backend._handle_cloud_job_status(
        cloud_env.prompt_id, cloud_env.job_info
    )

    assert result["status"] == "completed"
    assert result["storage_path"] == USER_STORAGE_PATH
    assert result["signed_url"] == f"https://signed.test/{USER_STORAGE_PATH}"

    # The authenticated user id is what the upload is scoped to.
    register_kwargs = cloud_env.comfyui.register_job.call_args.kwargs
    assert register_kwargs["user_id"] == USER_ID

    upload_args = cloud_env.comfyui.on_job_complete_async.call_args.args
    assert upload_args[0] == cloud_env.prompt_id
    assert upload_args[1].startswith("generated/cloud-minimaxh3/")
    assert upload_args[2] == "video"

    # Staging copy is written first, then removed once user storage has it.
    assert cloud_env.storage.put.call_args.args[0] == "generated"
    assert cloud_env.storage.delete.call_count == 1


async def test_cloud_job_stays_running_when_user_upload_fails(cloud_env):
    """No completion for media the user cannot see — the poller must retry."""
    cloud_env.comfyui.on_job_complete_async = AsyncMock(return_value=None)

    result = await backend._handle_cloud_job_status(
        cloud_env.prompt_id, cloud_env.job_info
    )

    assert result["status"] == "running"
    assert result.get("storage_path") is None
    # Staging copy stays so the retry can still upload it.
    cloud_env.storage.delete.assert_not_called()


async def test_cloud_job_without_owner_never_completes(cloud_env):
    """A job without a user id has no owner to scope storage to, so no completion."""
    cloud_env.job_info.pop("user_id")
    cloud_env.comfyui.on_job_complete_async = AsyncMock(return_value=None)

    result = await backend._handle_cloud_job_status(
        cloud_env.prompt_id, cloud_env.job_info
    )

    assert result["status"] == "running"
    # No placeholder ("adapter") owner is invented for an unowned job.
    cloud_env.comfyui.register_job.assert_not_called()
