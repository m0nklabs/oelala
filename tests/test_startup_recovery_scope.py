"""Regression: importing the app must not rewrite training-job state.

``face_train_service.recover_stuck_jobs()`` marks every job still marked ``running`` as
``failed`` — correct on a real restart, because daemon training threads do not survive one.
It used to run at *import* time, so merely importing ``app`` (a test, a script, CI, a REPL)
failed live jobs and then reported them to the user as "backend restarted during training".

The recovery now runs in the startup hook. The first test proves the import path is inert
in a genuinely fresh interpreter; the second proves the hook still does the work.
"""

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

BACKEND_DIR = Path(__file__).parent.parent / "src" / "backend"


def _seed_running_job(jobs_dir: Path) -> Path:
    jobs_dir.mkdir(parents=True, exist_ok=True)
    index = jobs_dir / "index.json"
    index.write_text(
        json.dumps(
            {
                "job-1": {
                    "status": "running",
                    "steps_total": 10,
                    "steps_done": 2,
                }
            }
        )
    )
    return index


def test_importing_app_leaves_running_jobs_alone(tmp_path):
    """A fresh interpreter imports ``app`` and the seeded job must still be running."""
    jobs_dir = tmp_path / "jobs"
    index = _seed_running_job(jobs_dir)

    script = f"""
import json, sys
sys.path.insert(0, {str(BACKEND_DIR)!r})
import face_train_service as fts
fts.JOBS_DIR = __import__("pathlib").Path({str(jobs_dir)!r})
fts.JOBS_INDEX = fts.JOBS_DIR / "index.json"
import app  # noqa: F401  -- the import under test
print(json.loads(open({str(index)!r}).read())["job-1"]["status"])
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "running", (
        "importing app rewrote live training-job state; recovery belongs in the "
        f"startup hook (stdout: {result.stdout[-500:]})"
    )


def test_startup_hook_still_recovers_orphaned_jobs(tmp_path):
    """The hook is where the recovery lives now, so it has to do it."""
    import face_train_service
    import app as app_module

    jobs_dir = tmp_path / "jobs"
    index = _seed_running_job(jobs_dir)

    # The hook also awaits the ComfyUI history backfill, which makes its own HTTP request
    # to the configured ComfyUI and writes data/generation_times.json when it finds new
    # outputs. Left alone it makes this regression test depend on a live ComfyUI and lets
    # it rewrite operational state — the very thing the module under test is guarding.
    backfill = AsyncMock(return_value=0)

    with (
        patch.object(face_train_service, "JOBS_DIR", jobs_dir),
        patch.object(face_train_service, "JOBS_INDEX", index),
        patch(
            "websocket_handler.backfill_generation_times_from_comfyui",
            backfill,
        ),
        patch.object(
            app_module,
            "get_comfyui_client",
            MagicMock(return_value=MagicMock(is_available=lambda: False)),
        ),
        patch.object(app_module, "ws_manager", None),
        patch.object(app_module, "job_queue_manager", None),
        patch.object(app_module, "progress_monitor", None),
        patch.object(app_module, "llm_queue_manager", None),
        patch.object(
            app_module, "webhook_service", MagicMock(start_retry_worker=AsyncMock())
        ),
        patch.object(app_module, "_cloud_job_poller", AsyncMock()),
        patch.object(app_module, "_local_job_poller", AsyncMock()),
    ):
        asyncio.run(app_module.startup_event())

    recovered = json.loads(index.read_text())["job-1"]
    assert recovered["status"] == "failed"
    assert "restarted" in recovered["error"].lower()
    assert backfill.await_count == 1, (
        "the startup hook no longer awaits the history backfill; if that is intentional, "
        "drop this patch (and the isolation reason for it)"
    )


@pytest.mark.parametrize("attribute", ["JOBS_DIR", "JOBS_INDEX"])
def test_conftest_redirects_job_paths_away_from_the_checkout(attribute, tmp_path):
    """Guards the isolation itself: a test run must not touch the checkout's index."""
    import face_train_service

    value = Path(getattr(face_train_service, attribute))
    assert "oelala-test-face-jobs-" in str(value), (
        f"conftest did not redirect {attribute} ({value}); tests would read and rewrite "
        "the operational job index"
    )
