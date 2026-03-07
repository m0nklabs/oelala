#!/usr/bin/env python3
"""
Recover a completed RunPod cloud job whose video was never saved to the user's bucket.

Usage:
    python scripts/recover_cloud_job.py \
        --runpod-job-id 2195de54-27c6-4fdd-9593-2d35d078974c-e1 \
        --user-id 68d7289d-427a-46a0-86d0-dffff1e43116 \
        --prompt "pope test cloud gen" \
        --gen-type t2v
"""

import argparse
import asyncio
import base64
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# Add backend to path so we can import backend modules
BACKEND_DIR = Path(__file__).parent.parent / "src" / "backend"
sys.path.insert(0, str(BACKEND_DIR))

# Load .env
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import httpx

RUNPOD_ENDPOINT_ID = "x2x496ymkidl3m"
RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
STORAGE_API_KEY = os.environ["STORAGE_API_KEY"]
STORAGE_BASE_URL = "http://localhost:7990"


async def fetch_runpod_result(runpod_job_id: str) -> dict:
    """Fetch the completed job output from RunPod."""
    url = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/status/{runpod_job_id}"
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    print(f"✅ RunPod status: {data.get('status')} | executionTime: {data.get('executionTime')}ms")
    return data


def decode_video_from_result(job_data: dict) -> tuple[bytes, str]:
    """Extract and base64-decode the video file. Returns (bytes, original_filename)."""
    output = job_data.get("output", {})
    if isinstance(output, dict):
        files = output.get("files", [])
    elif isinstance(output, list):
        files = output
    else:
        raise ValueError(f"Unexpected output format: {type(output)}")

    if not files:
        raise ValueError("No files in RunPod output")

    first_file = files[0]
    b64_data = first_file.get("data") or first_file.get("content")
    if not b64_data:
        raise ValueError(f"No base64 data in file entry: {list(first_file.keys())}")

    filename = first_file.get("filename", "output.mp4")
    print(f"📦 Decoding {len(b64_data)} chars of base64 → {filename}")
    return base64.b64decode(b64_data), filename


async def upload_to_storage(file_bytes: bytes, filename: str) -> str:
    """Upload the video bytes to oelala-storage. Returns the storage path."""
    bucket = "generated"
    object_path = f"cloud-max/{filename}"
    url = f"{STORAGE_BASE_URL}/{bucket}/{object_path}"
    headers = {
        "Authorization": f"Bearer {STORAGE_API_KEY}",
        "Content-Type": "video/mp4",
    }
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.put(url, content=file_bytes, headers=headers)
        resp.raise_for_status()
    storage_path = f"{bucket}/{object_path}"
    print(f"✅ Uploaded to storage: {storage_path} ({len(file_bytes)} bytes)")
    return storage_path


async def register_in_supabase(user_id: str, storage_path: str, filename: str, gen_type: str, prompt: str) -> dict:
    """
    Insert a user_media record in Supabase so the video appears in pope's gallery.
    Uses service key (bypasses RLS).
    """
    url = f"{SUPABASE_URL}/rest/v1/user_media"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    payload = {
        "user_id": user_id,
        "storage_path": storage_path,
        "filename": filename,
        "generation_type": gen_type,
        "prompt": prompt,
        "media_type": "video",
        "size_bytes": None,  # Will be backfilled if needed
        "metadata": {
            "recovered": True,
            "recovery_reason": "RunPod timeout bug — local state cleared before retrieval",
            "recovery_date": datetime.utcnow().isoformat(),
        },
    }
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(url, json=payload, headers=headers)
        if resp.status_code not in (200, 201):
            print(f"⚠️  Supabase response {resp.status_code}: {resp.text}")
            resp.raise_for_status()
        result = resp.json()
    record = result[0] if isinstance(result, list) else result
    print(f"✅ Supabase media record created: id={record.get('id')} | path={storage_path}")
    return record


async def main():
    parser = argparse.ArgumentParser(description="Recover a lost RunPod cloud gen result")
    parser.add_argument("--runpod-job-id", required=True)
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--prompt", default="cloud generation (recovered)")
    parser.add_argument("--gen-type", default="t2v")
    parser.add_argument("--dry-run", action="store_true", help="Fetch + decode but do NOT upload/register")
    args = parser.parse_args()

    print(f"\n🔍 Fetching RunPod job: {args.runpod_job_id}")
    job_data = await fetch_runpod_result(args.runpod_job_id)

    if job_data.get("status") != "COMPLETED":
        print(f"❌ Job is not COMPLETED (status={job_data.get('status')}). Aborting.")
        sys.exit(1)

    print("\n📥 Decoding video...")
    video_bytes, orig_filename = decode_video_from_result(job_data)
    print(f"   Size: {len(video_bytes):,} bytes ({len(video_bytes)/1024:.1f} KB)")

    # Build a canonical filename
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    ext = Path(orig_filename).suffix or ".mp4"
    save_filename = f"cloud_max_recovered_{timestamp}_000{ext}"
    print(f"   Save as: {save_filename}")

    if args.dry_run:
        print("\n🔔 Dry-run mode — skipping upload and Supabase registration.")
        # Save locally for inspection
        out = Path("/tmp") / save_filename
        out.write_bytes(video_bytes)
        print(f"   Saved locally: {out}")
        return

    print("\n📤 Uploading to oelala-storage...")
    storage_path = await upload_to_storage(video_bytes, save_filename)

    print("\n📝 Registering in Supabase...")
    record = await register_in_supabase(
        user_id=args.user_id,
        storage_path=storage_path,
        filename=save_filename,
        gen_type=args.gen_type,
        prompt=args.prompt,
    )

    print(f"\n🎉 Recovery complete!")
    print(f"   Storage path : {storage_path}")
    print(f"   Supabase ID  : {record.get('id')}")
    print(f"   User         : {args.user_id}")
    print(f"   Access via   : https://api.oelala.xyz/media/{storage_path}")


if __name__ == "__main__":
    asyncio.run(main())
