"""Shared test setup.

``app`` calls ``face_train_service.recover_stuck_jobs()`` while it is being imported.
That function marks every ``running`` training job as ``failed`` and rewrites the job
index, so importing the app in a test — even during collection — would mutate live job
state in an operational checkout.

Stubbing it here runs before any test module imports ``app``, which makes the whole suite
side-effect free instead of every app-importing module needing its own guard. The central
fix (startup work behind a real startup hook) is tracked in #220.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "backend"))

try:
    import face_train_service
except ImportError:  # pragma: no cover - app tolerates a missing face_train_service
    pass
else:
    face_train_service.recover_stuck_jobs = lambda *args, **kwargs: 0
