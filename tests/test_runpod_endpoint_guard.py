"""Regression tests for the RunPod endpoint guard behind ``POST /v2/generate``.

Every cloud video submission (text-to-video and image-to-video alike) answered
HTTP 503 even though valid RunPod endpoints were configured. ``app._submit_to_runpod``
opened with a guard that ignored the ``endpoint_id`` argument the function itself
accepts:

    if not _runpod or not _runpod.has_endpoint():
        raise HTTPException(503, ...)

Every cloud adapter resolves its *own* family endpoint (``RUNPOD_MINIMAX_H3_ENDPOINT_ID``,
``RUNPOD_LTX23_ENDPOINT_ID``, ``RUNPOD_I2I_ENDPOINT_ID``) and passes it in, so with the
generic ``RUNPOD_ENDPOINT_ID`` unset ``has_endpoint()`` was False and the guard fired
before the adapter's endpoint could ever be used.

This module locks in the fix (``endpoint_id or _runpod.has_endpoint()``) and the endpoint
resolution helpers it leans on:

* ``runpod_defaults.configured_endpoint_ids()`` — family env vars that are really set.
* ``runpod_defaults.default_endpoint_id()`` — ``RUNPOD_ENDPOINT_ID`` or the first
  configured family in ``DEFAULT_ENDPOINT_PREFERENCE`` order (MiniMax-H3 first, NOT the
  registry's insertion order, which starts at ltx23).
* ``RunPodClient`` default resolution from the family variables, and
  ``has_endpoint()`` honesty (hardcoded registry fallbacks never satisfy it).

Hermetic by construction: fake endpoint ids only, no network, no storage, no real
``.env`` values. ``app`` calls ``load_dotenv()`` at import time, so the ambient
environment can carry real endpoint ids — every test that reads resolution state clears
the whole endpoint env surface first (see ``_clear_endpoint_env``).
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import app as app_module  # noqa: E402
import runpod_defaults  # noqa: E402
from runpod_client import RunPodClient  # noqa: E402

# Obviously fake ids — never a real endpoint id (repo rule: no environment-specific
# values in committed files).
H3_ENDPOINT = "ep-h3-test"
LTX_ENDPOINT = "ep-ltx-test"
I2I_ENDPOINT = "ep-i2i-test"
GENERIC_ENDPOINT = "ep-generic-test"

# Generic (non-family) endpoint env vars the client also reads.
GENERIC_ENDPOINT_ENV_VARS = ("RUNPOD_ENDPOINT_ID", "RUNPOD_ENDPOINT_IDS")


def _family_env_vars():
    """Every family endpoint env var declared by the registry."""
    return [
        env_var
        for defaults in runpod_defaults.RUNPOD_ENDPOINT_DEFAULTS.values()
        for env_var in defaults.endpoint_env_vars
    ]


def _clear_endpoint_env(monkeypatch):
    """Remove every endpoint env var so ambient .env values cannot leak in."""
    for name in (*GENERIC_ENDPOINT_ENV_VARS, *_family_env_vars()):
        monkeypatch.delenv(name, raising=False)


def _set_family_endpoint(monkeypatch, profile, endpoint_id):
    """Point one family profile at an endpoint through its own env var."""
    defaults = runpod_defaults.RUNPOD_ENDPOINT_DEFAULTS[profile]
    monkeypatch.setenv(defaults.endpoint_env_vars[0], endpoint_id)


def test_configured_endpoint_ids_collects_only_the_vars_that_are_set(monkeypatch):
    """Unset families contribute nothing — and no hardcoded fallback leaks in."""
    _clear_endpoint_env(monkeypatch)
    assert runpod_defaults.configured_endpoint_ids() == []

    _set_family_endpoint(monkeypatch, "minimax_h3", H3_ENDPOINT)
    assert runpod_defaults.configured_endpoint_ids() == [H3_ENDPOINT]

    _set_family_endpoint(monkeypatch, "ltx23", LTX_ENDPOINT)
    assert set(runpod_defaults.configured_endpoint_ids()) == {
        H3_ENDPOINT,
        LTX_ENDPOINT,
    }
    # The i2i profile is still unset, so its fallback id must not appear.
    assert I2I_ENDPOINT not in runpod_defaults.configured_endpoint_ids()


def test_configured_endpoint_ids_skips_blank_values(monkeypatch):
    """An env var set to whitespace is not a configured endpoint."""
    _clear_endpoint_env(monkeypatch)
    h3_defaults = runpod_defaults.RUNPOD_ENDPOINT_DEFAULTS["minimax_h3"]
    monkeypatch.setenv(h3_defaults.endpoint_env_vars[0], "   ")
    _set_family_endpoint(monkeypatch, "ltx23", LTX_ENDPOINT)

    assert runpod_defaults.configured_endpoint_ids() == [LTX_ENDPOINT]


def test_configured_endpoint_ids_dedupes_repeated_ids(monkeypatch):
    """The same id configured for two families is reported once."""
    _clear_endpoint_env(monkeypatch)
    _set_family_endpoint(monkeypatch, "minimax_h3", H3_ENDPOINT)
    _set_family_endpoint(monkeypatch, "ltx23", H3_ENDPOINT)

    assert runpod_defaults.configured_endpoint_ids() == [H3_ENDPOINT]


def test_default_endpoint_id_prefers_the_generic_variable(monkeypatch):
    """RUNPOD_ENDPOINT_ID wins over every family endpoint."""
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", GENERIC_ENDPOINT)
    _set_family_endpoint(monkeypatch, "minimax_h3", H3_ENDPOINT)
    _set_family_endpoint(monkeypatch, "ltx23", LTX_ENDPOINT)
    _set_family_endpoint(monkeypatch, "i2i", I2I_ENDPOINT)

    assert runpod_defaults.default_endpoint_id() == GENERIC_ENDPOINT


def test_default_endpoint_id_follows_the_preference_order(monkeypatch):
    """MiniMax-H3 leads, even though the registry lists ltx23 first.

    This is the ordering regression that matters: ``configured_endpoint_ids()`` walks
    the registry (ltx23, i2i, minimax_h3), so "first configured" would have picked the
    LTX-2.3 endpoint. ``DEFAULT_ENDPOINT_PREFERENCE`` must decide instead.
    """
    _clear_endpoint_env(monkeypatch)
    _set_family_endpoint(monkeypatch, "minimax_h3", H3_ENDPOINT)
    _set_family_endpoint(monkeypatch, "ltx23", LTX_ENDPOINT)
    _set_family_endpoint(monkeypatch, "i2i", I2I_ENDPOINT)

    configured = runpod_defaults.configured_endpoint_ids()
    assert configured[0] == LTX_ENDPOINT  # registry order, not preference order

    assert runpod_defaults.default_endpoint_id() == H3_ENDPOINT
    assert runpod_defaults.default_endpoint_id() != LTX_ENDPOINT


def test_default_endpoint_id_falls_through_to_the_next_preference(monkeypatch):
    """With MiniMax-H3 unset, LTX-2.3 is the next in preference order."""
    _clear_endpoint_env(monkeypatch)
    _set_family_endpoint(monkeypatch, "ltx23", LTX_ENDPOINT)
    _set_family_endpoint(monkeypatch, "i2i", I2I_ENDPOINT)

    assert runpod_defaults.default_endpoint_id() == LTX_ENDPOINT


def test_default_endpoint_id_is_none_when_nothing_is_configured(monkeypatch):
    """No generic default and no family endpoints means no default at all."""
    _clear_endpoint_env(monkeypatch)

    assert runpod_defaults.default_endpoint_id() is None


def test_client_with_only_family_endpoints_reports_an_endpoint(monkeypatch):
    """Family variables alone are enough for has_endpoint() to be True."""
    _clear_endpoint_env(monkeypatch)
    _set_family_endpoint(monkeypatch, "minimax_h3", H3_ENDPOINT)
    _set_family_endpoint(monkeypatch, "ltx23", LTX_ENDPOINT)
    _set_family_endpoint(monkeypatch, "i2i", I2I_ENDPOINT)

    client = RunPodClient(api_key="test-key")

    assert client.has_endpoint() is True
    assert client.default_endpoint_id is not None
    # Preference order, not the registry order the family vars are read in.
    assert client.default_endpoint_id == H3_ENDPOINT
    # The failover set stays empty: family endpoints must never enter endpoint_ids,
    # because _candidate_endpoint_ids() builds the health-based failover list from
    # default_endpoint_id + endpoint_ids + _endpoints. Seeding them here would let a
    # MiniMax-H3 job fail over to the LTX-2.3 worker (other model, other container).
    # has_endpoint() is True purely because default_endpoint_id resolved above.
    assert client.endpoint_ids == []


def test_family_endpoints_never_join_the_failover_candidates(monkeypatch):
    """An H3 job must never fail over to the LTX-2.3 or i2i worker.

    ``_candidate_endpoint_ids()`` feeds the health-based failover in
    ``select_submit_endpoint()`` from ``default_endpoint_id + endpoint_ids + _endpoints``.
    With only family variables configured, the MiniMax-H3 endpoint must therefore be the
    single candidate: the other two families run a different model in a different
    container, so failing over to them would silently generate the wrong thing.
    """
    _clear_endpoint_env(monkeypatch)
    _set_family_endpoint(monkeypatch, "minimax_h3", H3_ENDPOINT)
    _set_family_endpoint(monkeypatch, "ltx23", LTX_ENDPOINT)
    _set_family_endpoint(monkeypatch, "i2i", I2I_ENDPOINT)

    client = RunPodClient(api_key="test-key")

    assert client._candidate_endpoint_ids() == [H3_ENDPOINT]
    assert LTX_ENDPOINT not in client._candidate_endpoint_ids()
    assert I2I_ENDPOINT not in client._candidate_endpoint_ids()


def test_explicit_failover_list_is_kept_verbatim(monkeypatch):
    """RUNPOD_ENDPOINT_IDS is still the deliberate failover set, order preserved."""
    _clear_endpoint_env(monkeypatch)
    monkeypatch.setenv("RUNPOD_ENDPOINT_IDS", "ep-eu-test, ep-us-test, ep-eu-test")
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", GENERIC_ENDPOINT)
    _set_family_endpoint(monkeypatch, "minimax_h3", H3_ENDPOINT)

    client = RunPodClient(api_key="test-key")

    # Explicit default first, deduped and in order; no family endpoint sneaks in.
    assert client.endpoint_ids == [GENERIC_ENDPOINT, "ep-eu-test", "ep-us-test"]
    assert client.default_endpoint_id == GENERIC_ENDPOINT
    assert H3_ENDPOINT not in client._candidate_endpoint_ids()


def test_client_without_endpoint_env_reports_no_endpoint(monkeypatch):
    """The guard must stay honest: registry fallbacks are not configured endpoints."""
    _clear_endpoint_env(monkeypatch)

    client = RunPodClient(api_key="test-key")

    assert client.has_endpoint() is False
    assert client.default_endpoint_id is None
    assert client.endpoint_ids == []


@pytest.fixture
def fake_runpod(monkeypatch):
    """Fake RunPod client wired into app, with the 503 guard left as the only variable.

    ``has_endpoint()`` reports False (the broken state that produced the 503s) while
    ``submit_workflow`` returns a fake job, so the code after the guard runs for real.
    Disk and storage side effects are patched out; nothing reaches the network.
    """
    fake_job = SimpleNamespace(id="rp-job-test", endpoint_id=H3_ENDPOINT)
    client = MagicMock()
    client.has_endpoint.return_value = False
    client.submit_workflow = AsyncMock(return_value=fake_job)

    monkeypatch.setattr(app_module, "_runpod", client)
    monkeypatch.setattr(app_module, "active_jobs", {})
    monkeypatch.setattr(app_module, "_persist_cloud_jobs", MagicMock())
    monkeypatch.setattr(app_module, "save_gen_start_artifacts", MagicMock())
    return client


def _workflow():
    """Minimal ComfyUI workflow (only the shapes the helper inspects)."""
    return {"1": {"class_type": "KSamplerAdvanced", "inputs": {"steps": 8, "cfg": 1.0}}}


class TestSubmitToRunpodGuard:
    """The bug: a per-family submission was rejected for want of a generic default."""

    async def test_explicit_endpoint_id_passes_the_guard(self, fake_runpod):
        """endpoint_id alone is sufficient, even with has_endpoint() False."""
        result = await app_module._submit_to_runpod(
            workflow=_workflow(),
            user_id="test-user-endpoint-guard",
            prompt_id="prompt-h3-guard-test",
            job_info={"prompt": "a cat", "job_type": "t2v"},
            endpoint_id=H3_ENDPOINT,
        )

        # The guard no longer consults has_endpoint() when an id is supplied.
        assert fake_runpod.has_endpoint() is False
        assert result["success"] is True
        assert result["runpod_job_id"] == "rp-job-test"
        assert result["compute_target"] == "cloud"
        fake_runpod.submit_workflow.assert_awaited_once()
        _, kwargs = fake_runpod.submit_workflow.call_args
        assert kwargs["endpoint_id"] == H3_ENDPOINT
        assert (
            app_module.active_jobs["prompt-h3-guard-test"]["runpod_endpoint_id"]
            == H3_ENDPOINT
        )

    async def test_no_endpoint_and_no_client_endpoint_raises_503(self, fake_runpod):
        """Without an explicit id and without a client endpoint, the guard still fires."""
        with pytest.raises(HTTPException) as excinfo:
            await app_module._submit_to_runpod(
                workflow=_workflow(),
                user_id="test-user-endpoint-guard",
                prompt_id="prompt-no-endpoint-test",
                job_info={"prompt": "a cat", "job_type": "t2v"},
                endpoint_id=None,
            )

        assert excinfo.value.status_code == 503
        fake_runpod.submit_workflow.assert_not_awaited()

    async def test_missing_client_raises_503_even_with_an_endpoint_id(
        self, fake_runpod, monkeypatch
    ):
        """No RunPod client at all is still a 503, whatever endpoint was resolved."""
        monkeypatch.setattr(app_module, "_runpod", None)

        with pytest.raises(HTTPException) as excinfo:
            await app_module._submit_to_runpod(
                workflow=_workflow(),
                user_id="test-user-endpoint-guard",
                prompt_id="prompt-no-client-test",
                job_info={"prompt": "a cat", "job_type": "t2v"},
                endpoint_id=H3_ENDPOINT,
            )

        assert excinfo.value.status_code == 503
        fake_runpod.submit_workflow.assert_not_awaited()

    async def test_client_endpoint_without_an_explicit_id_still_passes(
        self, fake_runpod
    ):
        """The pre-existing behaviour (client has an endpoint) must survive the fix."""
        fake_runpod.has_endpoint.return_value = True

        result = await app_module._submit_to_runpod(
            workflow=_workflow(),
            user_id="test-user-endpoint-guard",
            prompt_id="prompt-client-endpoint-test",
            job_info={"prompt": "a cat", "job_type": "i2v"},
            endpoint_id=None,
        )

        assert result["success"] is True
        fake_runpod.submit_workflow.assert_awaited_once()
        _, kwargs = fake_runpod.submit_workflow.call_args
        assert kwargs["endpoint_id"] is None
