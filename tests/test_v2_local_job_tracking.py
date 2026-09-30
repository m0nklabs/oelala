"""Regression tests for V2 local job tracking.

``track_v2_local_job`` must preserve ``backend_id`` in the active_jobs entry:
``_resolve_local_job_result`` uses it to resolve the backend-scoped ComfyUI
client that holds the ``register_job`` metadata. Dropping the field made the
completion poller fall back to the default client instance (which has no
metadata), skip the auto-upload, and leave the output unreachable via
/comfyui/output/ (HTTP 404 in the frontend).
"""

import sys
from pathlib import Path

backend_dir = Path(__file__).parent.parent / "src" / "backend"
sys.path.insert(0, str(backend_dir))

from app import active_jobs, track_v2_local_job  # noqa: E402


def test_track_v2_local_job_preserves_backend_id():
    prompt_id = "test-prompt-backend-id"
    try:
        track_v2_local_job(
            prompt_id,
            {
                "user_id": "user-1",
                "prompt": "a test prompt",
                "adapter_name": "krea2-local-t2i",
                "backend_id": "test-backend-comfyui",
                "model_type": "krea2",
                "source": "v2",
            },
        )
        entry = active_jobs[prompt_id]
        assert entry["backend_id"] == "test-backend-comfyui"
        assert entry["adapter_name"] == "krea2-local-t2i"
        assert entry["compute_target"] == "local"
        assert entry["user_id"] == "user-1"
    finally:
        active_jobs.pop(prompt_id, None)


def test_track_v2_local_job_backend_id_defaults_to_none():
    """Jobs without a backend_id still track cleanly (legacy/v1 compat)."""
    prompt_id = "test-prompt-no-backend-id"
    try:
        track_v2_local_job(prompt_id, {"user_id": "user-2", "prompt": "x"})
        assert active_jobs[prompt_id]["backend_id"] is None
    finally:
        active_jobs.pop(prompt_id, None)
