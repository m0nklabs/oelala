"""Regression: the unified media endpoint must actually reach the public gallery.

``GET /api/media/unified?source=public`` reads ``published_media`` through a Supabase
service-role client. It used to import that client from ``supabase_utils``, a module that
does not exist in the repository, and the surrounding ``except Exception`` swallowed the
``ImportError`` into a ``logger.debug`` — so the public gallery silently returned nothing
while the endpoint happily answered 200.

The assertion here is therefore "the published_media query ran", not "the status is 200":
a swallowed ImportError leaves the status untouched.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


backend_dir = Path(__file__).parent.parent / "src" / "backend"
sys.path.insert(0, str(backend_dir))

from app import app as backend_app  # noqa: E402  (recovery is stubbed in conftest)
from auth import User, get_current_user  # noqa: E402


class _Query:
    """Minimal Supabase table/query chain: ``select().eq().execute()``."""

    def __init__(self, table):
        self.table = table

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, *_args, **_kwargs):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, *_args, **_kwargs):
        return self

    def execute(self):
        return MagicMock(data=[])


class _Supabase:
    def __init__(self):
        self.tables = []

    def table(self, name):
        self.tables.append(name)
        return _Query(name)


@pytest.fixture
def client_with_user():
    backend_app.dependency_overrides[get_current_user] = lambda: User(
        id="00000000-0000-4000-8000-000000000000", email="test@example.invalid"
    )
    try:
        yield TestClient(backend_app)
    finally:
        backend_app.dependency_overrides.pop(get_current_user, None)


def test_public_source_queries_published_media(client_with_user):
    fake = _Supabase()
    with patch("gallery_api.get_supabase_client", return_value=fake):
        response = client_with_user.get("/api/media/unified", params={"source": "public"})

    assert response.status_code == 200
    assert "published_media" in fake.tables, (
        "the public gallery query never ran — the client import failed and the "
        f"endpoint swallowed it (tables queried: {fake.tables})"
    )


def test_no_module_imports_a_nonexistent_supabase_utils():
    """Guards the module name itself: the defect was an import of something absent."""
    offenders = [
        str(path.relative_to(backend_dir))
        for path in backend_dir.rglob("*.py")
        if "supabase_utils" in path.read_text(errors="ignore")
    ]
    assert not offenders, (
        "supabase_utils does not exist; the published-media paths must use "
        f"gallery_api.get_supabase_client (offenders: {offenders})"
    )
