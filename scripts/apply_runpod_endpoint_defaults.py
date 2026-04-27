#!/usr/bin/env python3
"""Apply Oelala's researched RunPod endpoint defaults.

This updates endpoint-level scaling/GPU placement settings only. Worker image and
template updates still go through the endpoint-specific deploy scripts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "src" / "backend"
sys.path.insert(0, str(BACKEND))

from runpod_defaults import RUNPOD_ENDPOINT_DEFAULTS  # noqa: E402


GRAPHQL_URL = "https://api.runpod.io/graphql"


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs without overriding existing environment."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def graphql_value(value: object) -> str:
    """Render a Python scalar as a GraphQL input literal."""
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def mutation_for(payload: dict[str, object]) -> str:
    """Build the saveEndpoint mutation for a payload."""
    fields = "\n".join(
        f"                    {key}: {graphql_value(value)}"
        for key, value in payload.items()
    )
    return f"""
        mutation {{
            saveEndpoint(input: {{
{fields}
            }}) {{
                id
                name
                gpuIds
                workersMin
                workersMax
                idleTimeout
                scalerType
                scalerValue
                templateId
            }}
        }}
    """


def post_graphql(api_key: str, query: str) -> dict[str, object]:
    """Post a GraphQL query to RunPod without exposing the API key in errors."""
    try:
        response = httpx.post(
            f"{GRAPHQL_URL}?api_key={api_key}",
            json={"query": query},
            timeout=30,
        )
    except httpx.HTTPError as exc:
        raise RuntimeError(f"RunPod GraphQL request failed: {exc}") from exc
    if response.status_code >= 400:
        raise RuntimeError(f"RunPod GraphQL HTTP {response.status_code}: {response.text}")

    data = response.json()
    if "errors" in data:
        raise RuntimeError(json.dumps(data["errors"], indent=2))
    return data


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        action="append",
        choices=sorted(RUNPOD_ENDPOINT_DEFAULTS),
        help="Endpoint profile to apply. Can be passed multiple times.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the endpoint payloads without applying them.",
    )
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    selected = args.profile or list(RUNPOD_ENDPOINT_DEFAULTS)

    if args.dry_run:
        for profile in selected:
            defaults = RUNPOD_ENDPOINT_DEFAULTS[profile]
            print(json.dumps(defaults.endpoint_payload(), indent=2))
        return 0

    api_key = os.getenv("RUNPOD_API_KEY", "").strip()
    if not api_key:
        print("RUNPOD_API_KEY is not configured", file=sys.stderr)
        return 1

    for profile in selected:
        defaults = RUNPOD_ENDPOINT_DEFAULTS[profile]
        data = post_graphql(api_key, mutation_for(defaults.endpoint_payload()))
        saved = data.get("data", {}).get("saveEndpoint", {})
        print(
            f"{profile}: {saved.get('id')} workers={saved.get('workersMin')}/"
            f"{saved.get('workersMax')} idle={saved.get('idleTimeout')}s "
            f"scaler={saved.get('scalerType')}:{saved.get('scalerValue')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
