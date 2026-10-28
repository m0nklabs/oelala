"""Shared test setup.

``face_train_service`` keeps its job index under ``src/data/face_train_jobs``. Tests that
import ``app`` — and any that exercise its startup hook — must not read or rewrite the
index of the checkout they run in: it is operational state, and a ``running`` job that a
test touches is a job a user loses.

Redirecting the module paths once, here, runs before any test module imports ``app``. The
real ``recover_stuck_jobs()`` keeps working (tests may seed a job and exercise it); it just
operates on a throwaway directory.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

try:
    import face_train_service
except ImportError:  # pragma: no cover - app tolerates a missing face_train_service
    pass
else:
    _JOBS_DIR = Path(tempfile.mkdtemp(prefix="oelala-test-face-jobs-"))
    face_train_service.JOBS_DIR = _JOBS_DIR
    face_train_service.JOBS_INDEX = _JOBS_DIR / "index.json"
    if hasattr(face_train_service, "JOBS_ROOT"):
        face_train_service.JOBS_ROOT = _JOBS_DIR
