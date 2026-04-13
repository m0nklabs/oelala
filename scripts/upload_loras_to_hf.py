#!/usr/bin/env python3
"""Upload LoRA files to a HuggingFace repo for CDN-backed cloud downloads.

Usage:
    python scripts/upload_loras_to_hf.py                     # upload all LTX LoRAs
    python scripts/upload_loras_to_hf.py ltx/specific.safetensors  # upload one file

After uploading, add entries to LORA_HF_SOURCES in src/backend/app.py
so the backend uses HF CDN URLs instead of self-hosted downloads.

Requirements:
    pip install huggingface_hub
"""

import argparse
import sys
from pathlib import Path

try:
    from huggingface_hub import HfApi, create_repo
except ImportError:
    print("❌ Install huggingface_hub: pip install huggingface_hub")
    sys.exit(1)

LORA_DIR = Path("/mnt/ssd/loras")
HF_REPO = "m0nk111/oelala-loras"  # Personal account (org needs separate token permissions)
REPO_TYPE = "model"


def upload_lora(api: HfApi, local_path: Path, repo_path: str) -> None:
    """Upload a single LoRA file to the HF repo."""
    size_mb = local_path.stat().st_size / (1024 * 1024)
    print(f"⬆️  Uploading {repo_path} ({size_mb:.0f} MB)...")
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_path,
        repo_id=HF_REPO,
        repo_type=REPO_TYPE,
    )
    print(f"✅ Uploaded: {repo_path}")
    print(f"   URL: https://huggingface.co/{HF_REPO}/resolve/main/{repo_path}")
    print(f'   Mapping: "{repo_path}": {{"repo": "{HF_REPO}", "path": "{repo_path}"}},')


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload LoRAs to HuggingFace")
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific LoRA files (relative to /mnt/ssd/loras/). If omitted, uploads all ltx/ LoRAs.",
    )
    parser.add_argument(
        "--private", action="store_true", default=True,
        help="Create repo as private (default: True)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded")
    args = parser.parse_args()

    api = HfApi()

    # Ensure repo exists
    if not args.dry_run:
        try:
            create_repo(HF_REPO, repo_type=REPO_TYPE, private=args.private, exist_ok=True)
            print(f"📦 Repo ready: https://huggingface.co/{HF_REPO}")
        except Exception as e:
            print(f"⚠️ Repo creation: {e}")

    # Collect files to upload
    if args.files:
        files = [(LORA_DIR / f, f) for f in args.files]
    else:
        # Default: upload all ltx/ LoRAs
        ltx_dir = LORA_DIR / "ltx"
        if not ltx_dir.exists():
            print(f"❌ {ltx_dir} does not exist")
            sys.exit(1)
        files = [(p, f"ltx/{p.name}") for p in ltx_dir.glob("*.safetensors")]

    if not files:
        print("No LoRA files found to upload.")
        sys.exit(0)

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Files to upload ({len(files)}):")
    for local, repo in sorted(files, key=lambda x: x[1]):
        if not local.exists():
            print(f"  ❌ {repo} — file not found: {local}")
            continue
        size_mb = local.stat().st_size / (1024 * 1024)
        print(f"  📄 {repo} ({size_mb:.0f} MB)")

    if args.dry_run:
        print("\nDry run complete. Add --no-dry-run to upload.")
        return

    print()
    for local, repo in sorted(files, key=lambda x: x[1]):
        if local.exists():
            upload_lora(api, local, repo)

    print("\n✅ Done! Add the mappings above to LORA_HF_SOURCES in src/backend/app.py")


if __name__ == "__main__":
    main()
