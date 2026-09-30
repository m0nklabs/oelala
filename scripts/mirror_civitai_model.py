#!/usr/bin/env python3
"""Mirror a Civitai model file into the private HuggingFace model repo.

The worker downloads checkpoints from HuggingFace (fast, no Civitai auth in the
hot path) and falls back to the Civitai URL. This script keeps the mirror in
sync and always uses the upstream Civitai filename, so manual downloads and
worker downloads agree.

Usage:
    scripts/mirror_civitai_model.py --version 3294059 --file-id 3178732
    scripts/mirror_civitai_model.py --version 3314686 --file-id 3203135 --keep-local

Tokens come from the project .env: CIVITAI_TOKEN (download) and
HF_PUBLIC_TOKEN (upload to the public mirror). Nothing logged contains a token.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import quote

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_STAGING = Path(os.environ.get("H3_STAGING_DIR", "/mnt/ssd/h3_checkpoints"))
DEFAULT_REPO = os.environ.get("HF_MODEL_MIRROR_REPO", "bomehika/oelala-models")
DEFAULT_SUBDIR = "diffusion_models"
CIVITAI_API = "https://civitai.com/api/v1"
CHUNK = 8 * 1024 * 1024


def env_value(name: str) -> str:
    """Read a value from the process environment, then from the project .env."""
    value = os.environ.get(name, "")
    if value:
        return value.strip().strip('"').strip("'")
    if ENV_FILE.exists():
        for line in ENV_FILE.read_text().splitlines():
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def resolve_file(version_id: int, file_id: int) -> dict:
    """Look up one file of a Civitai model version (name, size, download URL)."""
    model_id = requests.get(
        f"{CIVITAI_API}/model-versions/{version_id}", timeout=30
    ).json().get("modelId")
    if not model_id:
        raise SystemExit(f"❌ version {version_id} not found")
    model = requests.get(f"{CIVITAI_API}/models/{model_id}", timeout=30).json()
    for version in model.get("modelVersions", []):
        if version.get("id") != version_id:
            continue
        for entry in version.get("files", []):
            if entry.get("id") == file_id:
                return {
                    "model": model.get("name"),
                    "version": version.get("name"),
                    "filename": entry["name"],
                    "size_bytes": int(entry.get("sizeKB", 0) * 1024),
                    "fp": (entry.get("metadata") or {}).get("fp"),
                    "url": entry.get("downloadUrl"),
                }
    raise SystemExit(f"❌ file {file_id} not part of version {version_id}")


def download(url: str, target: Path, expected_bytes: int, token: str) -> None:
    """Stream a Civitai download to *target* (token passed as query param)."""
    sep = "&" if "?" in url else "?"
    url = f"{url}{sep}token={quote(token)}"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".download")
    started = time.time()
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(tmp, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=CHUNK):
                handle.write(chunk)
    size = tmp.stat().st_size
    if size < 1_000_000:
        tmp.unlink()
        raise SystemExit("❌ download too small — likely an auth page")
    speed = size / max(time.time() - started, 1) / 1e6
    print(f"   downloaded {size / 1e9:.2f} GB at {speed:.1f} MB/s")
    if expected_bytes and abs(size - expected_bytes) > expected_bytes * 0.01:
        print(f"   ⚠️  size differs from Civitai metadata ({expected_bytes / 1e9:.2f} GB)")
    tmp.rename(target)


def header_summary(path: Path) -> str:
    """Summarize a safetensors header (dtypes + quantization formats)."""
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
    meta = header.pop("__metadata__", {})
    dtypes = dict(Counter(v["dtype"] for v in header.values()))
    parts = [f"{len(header)} tensors", f"dtypes {dtypes}"]
    quant = meta.get("_quantization_metadata")
    if quant:
        layers = json.loads(quant).get("layers", {})
        parts.append(f"quant {dict(Counter(v.get('format') for v in layers.values()))}")
    if meta.get("description"):
        parts.append(f"description={meta['description']!r}")
    return " | ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", type=int, required=True, help="Civitai model version id")
    parser.add_argument("--file-id", type=int, required=True, help="Civitai file id")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="target HF model repo")
    parser.add_argument("--subdir", default=DEFAULT_SUBDIR, help="folder inside the repo")
    parser.add_argument("--staging", default=str(DEFAULT_STAGING), help="local download dir")
    parser.add_argument("--keep-local", action="store_true", help="keep the downloaded file")
    parser.add_argument("--dry-run", action="store_true", help="resolve and report only")
    args = parser.parse_args()

    civitai_token = env_value("CIVITAI_TOKEN")
    hf_token = env_value("HF_PUBLIC_TOKEN") or env_value("HF_LORA_TOKEN")
    if not civitai_token:
        print("❌ CIVITAI_TOKEN missing (env or .env)")
        return 1

    info = resolve_file(args.version, args.file_id)
    print(f"📦 {info['model']} — {info['version']}")
    print(f"   file: {info['filename']} ({info['size_bytes'] / 1e9:.2f} GB, fp={info['fp']})")
    if args.dry_run:
        print(f"   would upload to {args.repo}:{args.subdir}/{info['filename']}")
        return 0

    staging = Path(args.staging)
    target = staging / info["filename"]
    if target.exists() and abs(target.stat().st_size - info["size_bytes"]) <= info["size_bytes"] * 0.01:
        print("   already staged locally, skipping download")
    else:
        print("⬇️  downloading from Civitai...")
        download(info["url"], target, info["size_bytes"], civitai_token)

    try:
        print(f"🔍 {header_summary(target)}")
    except Exception as exc:  # noqa: BLE001 - header is informative only
        print(f"   ⚠️  header unreadable: {exc}")

    if not hf_token:
        print("❌ HF_PUBLIC_TOKEN/HF_LORA_TOKEN missing — cannot upload")
        return 1
    from huggingface_hub import HfApi

    api = HfApi(token=hf_token)
    path_in_repo = f"{args.subdir}/{info['filename']}"
    print(f"⬆️  uploading to {args.repo}:{path_in_repo} ...")
    commit = api.upload_file(
        path_or_fileobj=str(target),
        path_in_repo=path_in_repo,
        repo_id=args.repo,
        repo_type="model",
        commit_message=f"mirror {info['filename']} (civitai {info['model']} / {info['version']})",
    )
    print(f"✅ {commit}")

    if not args.keep_local:
        target.unlink()
        print("🧹 local staging file removed (use --keep-local to keep it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
