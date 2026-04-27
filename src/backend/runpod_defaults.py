"""RunPod endpoint defaults and request policies for Oelala.

The endpoint scaling defaults are intentionally mirrored by the deploy scripts.
The request policies are applied to every submitted RunPod job so long-running
video generations do not inherit RunPod's 10 minute execution timeout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


MILLISECONDS_PER_SECOND = 1000
MILLISECONDS_PER_MINUTE = 60 * MILLISECONDS_PER_SECOND


@dataclass(frozen=True)
class RunPodEndpointDefaults:
    """Best-known production defaults for one RunPod endpoint."""

    profile: str
    name: str
    endpoint_env_vars: tuple[str, ...]
    template_env_var: str | None
    fallback_endpoint_id: str
    fallback_template_id: str
    gpu_ids: str
    workers_min: int
    workers_max: int
    idle_timeout: int
    scaler_type: str
    scaler_value: int
    execution_timeout_ms: int
    ttl_ms: int

    def endpoint_id(self) -> str:
        """Resolve the configured endpoint ID from environment or fallback."""
        for env_var in self.endpoint_env_vars:
            value = os.getenv(env_var, "").strip()
            if value:
                return value
        return self.fallback_endpoint_id

    def template_id(self) -> str:
        """Resolve the configured template ID from environment or fallback."""
        if self.template_env_var:
            value = os.getenv(self.template_env_var, "").strip()
            if value:
                return value
        return self.fallback_template_id

    def request_policy(self) -> dict[str, Any]:
        """Return RunPod's per-job policy payload."""
        return {
            "executionTimeout": self.execution_timeout_ms,
            "ttl": self.ttl_ms,
            "lowPriority": False,
        }

    def endpoint_payload(self) -> dict[str, Any]:
        """Return a GraphQL saveEndpoint-compatible payload."""
        return {
            "id": self.endpoint_id(),
            "name": self.name,
            "templateId": self.template_id(),
            "gpuIds": self.gpu_ids,
            "workersMin": self.workers_min,
            "workersMax": self.workers_max,
            "idleTimeout": self.idle_timeout,
            "scalerType": self.scaler_type,
            "scalerValue": self.scaler_value,
        }


RUNPOD_ENDPOINT_DEFAULTS: dict[str, RunPodEndpointDefaults] = {
    "wan22": RunPodEndpointDefaults(
        profile="wan22",
        name="oelala-wan22",
        endpoint_env_vars=("RUNPOD_WAN22_ENDPOINT_ID", "RUNPOD_ENDPOINT_ID"),
        template_env_var="RUNPOD_WAN22_TEMPLATE_ID",
        fallback_endpoint_id="x2x496ymkidl3m",
        fallback_template_id="tkpy0pi8gt",
        gpu_ids="AMPERE_48,ADA_48_PRO,AMPERE_80,ADA_80_PRO,BLACKWELL_96,HOPPER_141,BLACKWELL_180",
        workers_min=0,
        workers_max=2,
        idle_timeout=120,
        scaler_type="QUEUE_DELAY",
        scaler_value=4,
        execution_timeout_ms=60 * MILLISECONDS_PER_MINUTE,
        ttl_ms=2 * 60 * MILLISECONDS_PER_MINUTE,
    ),
    "ltx23": RunPodEndpointDefaults(
        profile="ltx23",
        name="oelala-ltx23",
        endpoint_env_vars=("RUNPOD_LTX23_ENDPOINT_ID",),
        template_env_var="RUNPOD_LTX23_TEMPLATE_ID",
        fallback_endpoint_id="ctpoa610dva4ww",
        fallback_template_id="c1fz26l07d",
        gpu_ids="AMPERE_80,ADA_80_PRO,HOPPER_141,BLACKWELL_96,BLACKWELL_180",
        workers_min=0,
        workers_max=2,
        idle_timeout=120,
        scaler_type="QUEUE_DELAY",
        scaler_value=1,
        execution_timeout_ms=45 * MILLISECONDS_PER_MINUTE,
        ttl_ms=2 * 60 * MILLISECONDS_PER_MINUTE,
    ),
    "i2i": RunPodEndpointDefaults(
        profile="i2i",
        name="oelala-i2i",
        endpoint_env_vars=("RUNPOD_I2I_ENDPOINT_ID",),
        template_env_var="RUNPOD_I2I_TEMPLATE_ID",
        fallback_endpoint_id="8djiexluyybooj",
        fallback_template_id="ed2614hd8k",
        gpu_ids="AMPERE_48,ADA_48_PRO,AMPERE_80,ADA_80_PRO,BLACKWELL_96,HOPPER_141,BLACKWELL_180",
        workers_min=0,
        workers_max=2,
        idle_timeout=120,
        scaler_type="QUEUE_DELAY",
        scaler_value=4,
        execution_timeout_ms=15 * MILLISECONDS_PER_MINUTE,
        ttl_ms=60 * MILLISECONDS_PER_MINUTE,
    ),
}


def endpoint_profile_for_id(endpoint_id: str | None) -> str | None:
    """Resolve a RunPod endpoint ID to a known Oelala profile name."""
    if not endpoint_id:
        return None
    for profile, defaults in RUNPOD_ENDPOINT_DEFAULTS.items():
        ids = {defaults.fallback_endpoint_id, defaults.endpoint_id()}
        if endpoint_id in ids:
            return profile
    return None


def get_runpod_job_policy(endpoint_id: str | None) -> dict[str, Any] | None:
    """Return the RunPod per-job policy for a known endpoint ID."""
    profile = endpoint_profile_for_id(endpoint_id)
    if profile is None:
        return None
    return RUNPOD_ENDPOINT_DEFAULTS[profile].request_policy()
