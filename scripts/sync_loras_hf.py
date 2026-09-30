#!/usr/bin/env python3
"""Sync the local LoRA store to the private HuggingFace mirror repo.

Cloud workers try the HF mirror (fast CDN) first and fall back to the signed
self-hosted download, so the mirror only needs to hold the frequently used
files — anything missing simply falls back.

Usage:
  python scripts/sync_loras_hf.py --dry-run          # list what would upload
  python scripts/sync_loras_hf.py                    # upload everything new
  python scripts/sync_loras_hf.py --include minimax-h3/   # one subdir only

Config (from .env): HF_LORA_TOKEN (needs write access), LORA_HF_MIRROR_REPO.
The script refuses to upload unless the target repo is PRIVATE.
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
    args = parser.parse_args()

    env = load_env(REPO_ROOT / ".env")
    token = env.get("HF_LORA_TOKEN", "") or os.getenv("HF_LORA_TOKEN", "")
    repo = args.repo or env.get("LORA_HF_MIRROR_REPO", "")
    if not repo:
        raise SystemExit("❌ No mirror repo: pass --repo or set LORA_HF_MIRROR_REPO in .env")
    if not token:
        raise SystemExit("❌ No HF token: set HF_LORA_TOKEN in .env")

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repo_type = detect_repo_type(api, repo)
    info = api.repo_info(repo, repo_type=repo_type)
    if not getattr(info, "private", False):
        raise SystemExit(
            "🚫 Refusing to upload: mirror repo is PUBLIC. LoRAs may contain "
            "NSFW content — make the repo private first (or pass --allow-public)."
        )

    files = collect_files()
    if args.include:
        files = {k: v for k, v in files.items() if k.startswith(args.include)}
    print(f"📂 {len(files)} local LoRA file(s) under consideration")

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
