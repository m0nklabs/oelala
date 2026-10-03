#!/usr/bin/env python3
"""Build the LoRA source index used for per-file download-source selection.

The backend prefers a third-party public dump over our own mirrors (traffic and
exposure stay on someone else's account), but a flat dump is matched by basename
and cannot be probed cheaply at request time. This script records which
basenames each dump actually holds, so `build_lora_download_list()` can pick the
right primary URL per LoRA instead of firing a guaranteed 404.

Output (default `data/lora_source_index.json`, override with LORA_SOURCE_INDEX):

    {"generated_at": "...", "repos": [{"repo": "...", "files": {"<basename>": "<path in repo>"}}]}

Repo order in the file is the priority order. Re-run after a dump gains files;
the backend falls back to the simple `LORA_HF_FLAT_MIRROR_REPO`-first chain when
the index file is missing or unreadable.

Usage:
    python scripts/build_lora_source_index.py                       # env-configured dumps
    python scripts/build_lora_source_index.py --repo Serenak/chilloutmix
    python scripts/build_lora_source_index.py --repos a/b,c/d --out data/x.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data" / "lora_source_index.json"
WEIGHT_SUFFIXES = (".safetensors", ".ckpt", ".pt", ".pth")


def load_env(path: Path) -> dict:
    env: dict[str, str] = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def configured_dumps(env: dict) -> list[str]:
    """Third-party dump repos, in priority order, from env or .env."""
    raw = os.getenv("LORA_THIRD_PARTY_DUMPS", "") or env.get("LORA_THIRD_PARTY_DUMPS", "")
    repos = [r.strip() for r in raw.split(",") if r.strip()]
    if not repos:
        single = os.getenv("LORA_HF_FLAT_MIRROR_REPO", "") or env.get("LORA_HF_FLAT_MIRROR_REPO", "")
        if single:
            repos = [single]
    return repos


def local_sizes(roots: list[Path] | None = None) -> dict[str, set[int]]:
    """Map basename -> set of local file sizes (a basename can be ambiguous)."""
    if roots is None:
        roots = [
            Path(os.getenv("LORA_SSD_DIR", "/mnt/ssd/loras")),
            Path(os.getenv("LORA_DIR", str(REPO_ROOT / "ComfyUI" / "models" / "loras"))),
        ]
    sizes: dict[str, set[int]] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*.safetensors"):
            try:
                sizes.setdefault(path.name, set()).add(path.stat().st_size)
            except OSError:
                continue
    return sizes


def verified_files(
    remote: dict[str, list[tuple[str, int]]], local: dict[str, set[int]]
) -> tuple[dict[str, str], list[str]]:
    """
    Split remote files into (accepted `{basename: path_in_repo}`, rejected names).

    A dump copy is only a safe substitute when its byte size matches a local file
    of the same name: same basename alone does not prove the same revision. The
    recorded path is used verbatim for the download URL, so dumps that keep files
    in subdirectories (e.g. `high/`) still resolve.
    """
    accepted: dict[str, str] = {}
    rejected: list[str] = []
    for name, candidates in sorted(remote.items()):
        sizes = local.get(name, set())
        match = next((path for path, size in candidates if size and size in sizes), None)
        if match:
            accepted[name] = match
        else:
            rejected.append(name)
    return accepted, rejected


def list_weight_files(repo: str, timeout: int = 60) -> dict[str, list[tuple[str, int]]]:
    """Fetch a public repo's file list as {basename: [(path, size), ...]}."""
    url = f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true"
    with urllib.request.urlopen(url, timeout=timeout) as response:
        entries = json.load(response)
    if not isinstance(entries, list):
        raise RuntimeError(f"unexpected tree response for {repo}")
    files: dict[str, list[tuple[str, int]]] = {}
    for entry in entries:
        if entry.get("type") != "file" or not entry["path"].lower().endswith(WEIGHT_SUFFIXES):
            continue
        path = entry["path"]
        files.setdefault(Path(path).name, []).append((path, int(entry.get("size") or 0)))
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="", help="single dump repo")
    parser.add_argument("--repos", default="", help="comma-separated dump repos, priority order")
    parser.add_argument("--out", default="", help=f"output path (default {DEFAULT_OUT})")
    parser.add_argument(
        "--verbose-rejected",
        action="store_true",
        help="list the dump files that were skipped by the size check",
    )
    args = parser.parse_args()

    env = load_env(REPO_ROOT / ".env")
    if args.repos:
        repos = [r.strip() for r in args.repos.split(",") if r.strip()]
    elif args.repo:
        repos = [args.repo]
    else:
        repos = configured_dumps(env)
    if not repos:
        raise SystemExit(
            "❌ No dump repo: pass --repo/--repos or set LORA_THIRD_PARTY_DUMPS "
            "(or LORA_HF_FLAT_MIRROR_REPO) in .env"
        )

    out = Path(args.out) if args.out else Path(
        os.getenv("LORA_SOURCE_INDEX", "") or DEFAULT_OUT
    )

    entries = []
    local = local_sizes()
    print(f"📂 local store: {len(local)} distinct basename(s)")
    for repo in repos:
        try:
            remote = list_weight_files(repo)
        except Exception as exc:  # keep the previous index usable
            print(f"⚠️  {repo}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        accepted, rejected = verified_files(remote, local)
        print(
            f"📦 {repo}: {len(remote)} weight file(s), "
            f"{len(accepted)} size-verified against the local store, "
            f"{len(rejected)} skipped (not ours / different revision)"
        )
        if args.verbose_rejected:
            for name in rejected:
                print(f"     skipped: {name}")
        entries.append({"repo": repo, "files": accepted})

    if not entries:
        raise SystemExit("❌ No repo could be listed — index not written")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repos": entries,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"✅ Wrote {out} ({len(entries)} repo(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
