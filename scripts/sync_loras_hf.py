#!/usr/bin/env python3
"""Sync the local LoRA store to a HuggingFace mirror repo.

Cloud workers try the HF mirror (fast CDN) first and fall back to the signed
self-hosted download, so anything missing on the mirror simply falls back.

Two targets are supported:
  * public delivery mirror (`bomehika/oelala-loras`, no worker token needed)
  * private mirror (`m0nk111/oelala-loras`) for files the exclusion policy keeps
    out of public view

Usage:
  python scripts/sync_loras_hf.py --dry-run          # list what would upload
  python scripts/sync_loras_hf.py                    # upload everything new
  python scripts/sync_loras_hf.py --include minimax-h3/   # one subdir only
  python scripts/sync_loras_hf.py --repo bomehika/oelala-loras \
      --token-env HF_PUBLIC_TOKEN --allow-public       # public delivery mirror

Config (from .env): the token named by --token-env (write access), plus
LORA_HF_MIRROR_REPO for the default repo. A PUBLIC repo needs --allow-public;
files listed in scripts/lora_public_exclusions.txt are never uploaded to one.
"""

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env(path: Path) -> dict:
    env = {}
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def collect_files() -> dict[str, Path]:
    """Collect *.safetensors from both LoRA roots; LORA_DIR wins on conflicts."""
    lora_dir = Path(os.getenv("LORA_DIR", str(REPO_ROOT / "ComfyUI/models/loras")))
    lora_ssd_dir = Path(os.getenv("LORA_SSD_DIR", "/mnt/ssd/loras"))
    files: dict[str, Path] = {}
    # SSD first, primary dir second (primary overrides on same relpath)
    for root in (lora_ssd_dir, lora_dir):
        if not root.is_dir():
            continue
        for path in root.rglob("*.safetensors"):
            rel = str(path.relative_to(root))
            files[rel] = path
    return files


def detect_repo_type(api, repo_id: str) -> str:
    for repo_type in ("model", "dataset"):
        try:
            api.repo_info(repo_id, repo_type=repo_type)
            return repo_type
        except Exception:
            continue
    raise SystemExit(f"❌ Repo '{repo_id}' not accessible (check token/permissions)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=os.getenv("LORA_HF_MIRROR_REPO", ""))
    parser.add_argument("--include", default="", help="only relpaths starting with this prefix")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--allow-public",
        action="store_true",
        help="permit a PUBLIC mirror repo (exclusion policy still applies)",
    )
    parser.add_argument(
        "--token-env",
        default="HF_LORA_TOKEN",
        help="env var holding the HF token for this repo (bomehika repos use HF_PUBLIC_TOKEN)",
    )
    parser.add_argument(
        "--exclusions",
        default="",
        help="exclusion list path (default: scripts/lora_public_exclusions.txt)",
    )
    parser.add_argument(
        "--only-excluded",
        action="store_true",
        help="upload ONLY the exclusion-list files (for a private mirror)",
    )
    args = parser.parse_args()

    env = load_env(REPO_ROOT / ".env")
    token = env.get(args.token_env, "") or os.getenv(args.token_env, "")
    repo = args.repo or env.get("LORA_HF_MIRROR_REPO", "")
    if not repo:
        raise SystemExit("❌ No mirror repo: pass --repo or set LORA_HF_MIRROR_REPO in .env")
    if not token:
        raise SystemExit(f"❌ No HF token: set {args.token_env} in .env")

    from huggingface_hub import HfApi

    sys.path.insert(0, str(REPO_ROOT / "src" / "backend"))
    from generation.lora_public_policy import load_exclusions, partition

    api = HfApi(token=token)
    repo_type = detect_repo_type(api, repo)
    info = api.repo_info(repo, repo_type=repo_type)
    is_public = not getattr(info, "private", False)
    if is_public and not args.allow_public:
        raise SystemExit(
            "🚫 Refusing to upload: mirror repo is PUBLIC. LoRAs may contain "
            "NSFW content — make the repo private first (or pass --allow-public "
            "to accept the public-mirror policy)."
        )

    files = collect_files()
    if args.include:
        files = {k: v for k, v in files.items() if k.startswith(args.include)}
    print(f"📂 {len(files)} local LoRA file(s) under consideration")

    patterns = load_exclusions(args.exclusions or None)
    allowed, excluded = partition(files, patterns)
    print(f"🔒 Exclusion policy: {len(patterns)} pattern(s) — "
          f"{len(excluded)} local file(s) held back")
    if args.only_excluded:
        if is_public:
            raise SystemExit(
                "🚫 --only-excluded targets a PRIVATE mirror: excluded files "
                "must not go to a public repo."
            )
        files = excluded
        print(f"🔐 Private-mirror mode: uploading only the {len(files)} excluded file(s)")
    elif is_public:
        for rel in sorted(excluded):
            print(f"  ⛔ not for public mirror: {rel}")
        files = allowed
    elif excluded:
        for rel in sorted(excluded):
            print(f"  ⚠️  excluded file allowed on this PRIVATE mirror: {rel}")

    try:
        repo_files = set(api.list_repo_files(repo, repo_type=repo_type))
    except Exception:
        repo_files = set()
    todo = {k: v for k, v in files.items() if k not in repo_files}
    total_gb = sum(v.stat().st_size for v in todo.values()) / 1e9
    already = len(files) - len(todo)
    print(f"☁️  Mirror: {repo} ({repo_type}) — {already} already present, "
          f"{len(todo)} to upload ({total_gb:.1f} GB)")

    if args.dry_run:
        for rel in sorted(todo):
            print(f"  would upload: {rel} ({todo[rel].stat().st_size / 1e6:.0f} MB)")
        return

    for i, (rel, path) in enumerate(sorted(todo.items()), 1):
        size_mb = path.stat().st_size / 1e6
        print(f"[{i}/{len(todo)}] ⬆️  {rel} ({size_mb:.0f} MB)...", flush=True)
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=rel,
            repo_id=repo,
            repo_type=repo_type,
            commit_message=f"sync {rel}",
        )
    print(f"✅ Sync complete: {len(todo)} uploaded, {already} already present")


if __name__ == "__main__":
    sys.exit(main())
