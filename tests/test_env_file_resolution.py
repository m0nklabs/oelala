"""Regression: the backend must load *its own* `.env`, not a hardcoded host path.

`app.py` used to call ``load_dotenv(dotenv_path="/home/<user>/oelala/.env")``, so every
process started from another clone or worktree silently loaded the first deployment's
production configuration, and a checkout anywhere else had no configuration at all.

The path now comes from ``OELALA_ENV_FILE`` or is derived from this module's location.
"""

import os
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).parent.parent / "src" / "backend"


def _app_module():
    import app

    return app


def test_default_env_file_is_derived_from_the_module_location(monkeypatch):
    """No override: the file sits at the root of *this* checkout, computed from __file__."""
    monkeypatch.delenv("OELALA_ENV_FILE", raising=False)
    module = _app_module()

    expected = Path(module.__file__).resolve().parents[2] / ".env"
    assert module.resolve_env_file() == expected


def test_env_file_override_is_honoured(monkeypatch, tmp_path):
    module = _app_module()
    target = tmp_path / "custom.env"
    target.write_text("SOMETHING=1\n")

    monkeypatch.setenv("OELALA_ENV_FILE", str(target))
    assert module.resolve_env_file() == target


def test_env_file_override_expands_a_home_relative_path(monkeypatch):
    module = _app_module()
    monkeypatch.setenv("OELALA_ENV_FILE", "~/somewhere.env")
    assert module.resolve_env_file() == Path("~/somewhere.env").expanduser()


def test_importing_app_loads_the_configured_env_file(tmp_path):
    """A fresh interpreter, run from an unrelated cwd, reads the file it was pointed at."""
    env_file = tmp_path / "elsewhere.env"
    env_file.write_text("OELALA_TEST_MARKER=from-override\n")

    script = f"""
import os, sys
sys.path.insert(0, {str(BACKEND_DIR)!r})
import app  # noqa: F401  -- the import under test
print(os.getenv("OELALA_TEST_MARKER", ""))
"""
    env = {k: v for k, v in os.environ.items() if k != "OELALA_TEST_MARKER"}
    env["OELALA_ENV_FILE"] = str(env_file)

    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=str(tmp_path),
        env=env,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip().splitlines()[-1] == "from-override"


def test_env_loading_has_no_baked_in_host_path():
    """The defect itself: an absolute path in the dotenv call cannot come back."""
    source = (BACKEND_DIR / "app.py").read_text()
    offenders = [
        line.strip()
        for line in source.splitlines()
        if "dotenv_path=" in line and "/home/" in line
    ]
    assert not offenders, (
        "the env file path must come from OELALA_ENV_FILE or the module location: "
        f"{offenders}"
    )
