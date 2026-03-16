#!/usr/bin/env python3
"""Upload selected private assets to the detached RunPod LoRA Network Volume.

This tool is intentionally strict:
- only uploads to the configured LoRA/private asset volume
- only writes under models/loras/ or models/custom/
- blocks known public/general model filenames used by Cloud Max

It is meant for curated, on-demand uploads of local rare assets, not for broad
model library syncs or cold-start cache prewarming.
"""

from __future__ import annotations

import argparse
import mimetypes
import os
from pathlib import Path
from typing import Iterable


DEFAULT_VOLUME_ID = os.getenv("RUNPOD_LORA_VOLUME_ID", "ochebt0xbq")
DEFAULT_DATACENTER = os.getenv("RUNPOD_LORA_VOLUME_DATACENTER", "EU-CZ-1")
DEFAULT_ENDPOINT = os.getenv("RUNPOD_LORA_S3_ENDPOINT", "https://s3api-eu-cz-1.runpod.io")
DEFAULT_ALLOWED_PREFIXES = ("models/loras", "models/custom")
DEFAULT_LORA_ROOTS = (
    Path("/mnt/ssd/loras"),
    Path("/home/flip/oelala/ComfyUI/models/loras"),
)
BLOCKED_PUBLIC_FILENAMES = {
    "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
    "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
    "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
    "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
    "umt5_xxl_fp16.safetensors",
    "wan_2.1_vae.safetensors",
    "clip_vision_h.safetensors",
}
BLOCKED_PATH_PARTS = {
    "huggingface-cache",
    "diffusion_models",
    "unet",
    "clip",
    "clip_vision",
    "text_encoders",
    "vae",
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", help="Local file or directory paths to upload")
    parser.add_argument(
        "--remote-prefix",
        default="models/loras",
        help="Remote prefix on the RunPod volume. Only models/loras or models/custom are allowed.",
    )
    parser.add_argument("--volume-id", default=DEFAULT_VOLUME_ID, help="RunPod Network Volume ID")
    parser.add_argument("--datacenter", default=DEFAULT_DATACENTER, help="RunPod datacenter ID")
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT, help="RunPod S3 endpoint URL")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without writing")
    return parser


def _normalize_remote_prefix(remote_prefix: str) -> str:
    normalized = remote_prefix.strip().strip("/")
    if normalized not in DEFAULT_ALLOWED_PREFIXES:
        allowed = ", ".join(DEFAULT_ALLOWED_PREFIXES)
        raise ValueError(f"Remote prefix must be one of: {allowed}")
    return normalized


def _iter_upload_candidates(raw_paths: Iterable[str]) -> list[Path]:
    candidates: list[Path] = []
    for raw_path in raw_paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file():
                    candidates.append(child)
        else:
            candidates.append(path)
    return candidates


def _default_relative_name(path: Path) -> Path:
    for root in DEFAULT_LORA_ROOTS:
        try:
            return path.relative_to(root)
        except ValueError:
            continue
    return Path(path.name)


def _validate_candidate(path: Path, remote_prefix: str) -> None:
    if path.name in BLOCKED_PUBLIC_FILENAMES:
        raise ValueError(f"Blocked public/general model filename: {path.name}")

    lower_parts = {part.lower() for part in path.parts}
    if lower_parts & BLOCKED_PATH_PARTS and remote_prefix != "models/custom":
        blocked = ", ".join(sorted(lower_parts & BLOCKED_PATH_PARTS))
        raise ValueError(f"Blocked path looks like public/general model content: {path} ({blocked})")


def _build_remote_key(path: Path, remote_prefix: str) -> str:
    relative_name = _default_relative_name(path).as_posix().lstrip("/")
    return f"{remote_prefix}/{relative_name}"


def _build_s3_client(datacenter: str, endpoint: str):
    try:
        import boto3
        from botocore.client import Config
    except ImportError as exc:
        raise RuntimeError("boto3 is required for RunPod S3 uploads") from exc

    access_key = os.getenv("RUNPOD_S3_ACCESS_KEY_ID") or os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("RUNPOD_S3_SECRET_ACCESS_KEY") or os.getenv("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        raise RuntimeError(
            "Missing RunPod S3 credentials. Set RUNPOD_S3_ACCESS_KEY_ID and RUNPOD_S3_SECRET_ACCESS_KEY."
        )

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=datacenter,
        config=Config(signature_version="s3v4"),
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    remote_prefix = _normalize_remote_prefix(args.remote_prefix)
    candidates = _iter_upload_candidates(args.paths)

    uploads: list[tuple[Path, str]] = []
    for path in candidates:
        _validate_candidate(path, remote_prefix)
        uploads.append((path, _build_remote_key(path, remote_prefix)))

    if not uploads:
        print("No files to upload.")
        return 0

    for path, remote_key in uploads:
        print(f"PLAN {path} -> s3://{args.volume_id}/{remote_key}")

    if args.dry_run:
        return 0

    client = _build_s3_client(args.datacenter, args.endpoint)
    for path, remote_key in uploads:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        extra_args = {"ContentType": content_type}
        client.upload_file(str(path), args.volume_id, remote_key, ExtraArgs=extra_args)
        print(f"UPLOADED {path} -> s3://{args.volume_id}/{remote_key}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
