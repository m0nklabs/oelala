"""
Generation artifact storage — saves workflow, settings, input images, and logs
per generation to the user's storage bucket.

Storage layout per generation:
  generated/users/{user_id}/generations/{prompt_id}/
    ├── manifest.json      # Full prompt, all settings, timestamps, job type
    ├── workflow.json       # Complete ComfyUI API workflow
    ├── input_image.{ext}   # Start image (I2V only)
    └── execution.log       # Full execution logs (ComfyUI / RunPod handler)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy-initialised storage client (avoids circular imports)
_storage_client = None
BUCKET = "generated"


def _get_storage():
    global _storage_client
    if _storage_client is None:
        from storage_client import get_client as get_storage_client
        _storage_client = get_storage_client()
    return _storage_client


def _gen_key(user_id: str, prompt_id: str, filename: str) -> str:
    """Build storage key for a generation artifact."""
    return f"users/{user_id}/generations/{prompt_id}/{filename}"


# ── Save at job start ───────────────────────────────────────────────────────

def save_gen_start_artifacts(
    user_id: str,
    prompt_id: str,
    workflow: dict,
    prompt: str,
    job_info: dict,
    input_image_path: Optional[str] = None,
) -> None:
    """Save workflow, manifest, and optional input image at generation start.

    Fire-and-forget: failures are logged but never block generation.
    """
    try:
        storage = _get_storage()

        # 1. Full workflow JSON
        wf_key = _gen_key(user_id, prompt_id, "workflow.json")
        storage.put(BUCKET, wf_key, json.dumps(workflow, indent=2).encode("utf-8"))

        # 2. Manifest with full prompt + all settings
        private_keys = {
            "user_id", "_start_time", "_job_type", "_cloud_completed",
            "_cloud_status", "_cloud_error", "_prompt_full",
            "runpod_endpoint_id",
        }
        manifest = {
            "prompt_id": prompt_id,
            "user_id": user_id,
            "prompt": prompt,
            "created_at": datetime.now().isoformat(),
            "compute_target": job_info.get("compute_target", "local"),
            "job_type": job_info.get("job_type", "unknown"),
            "settings": {
                k: v for k, v in job_info.items()
                if k not in private_keys and not k.startswith("_")
            },
        }
        credits = job_info.get("credits_required")
        if credits is not None:
            manifest["credits_required"] = credits
        rpjob = job_info.get("runpod_job_id")
        if rpjob:
            manifest["runpod_job_id"] = rpjob

        mf_key = _gen_key(user_id, prompt_id, "manifest.json")
        storage.put(BUCKET, mf_key, json.dumps(manifest, indent=2).encode("utf-8"))

        # 3. Input image (I2V) — read from disk if available
        if input_image_path:
            img_path = Path(input_image_path)
            if img_path.exists():
                ext = img_path.suffix or ".png"
                img_key = _gen_key(user_id, prompt_id, f"input_image{ext}")
                storage.put(BUCKET, img_key, img_path.read_bytes())
                logger.debug(
                    f"📎 Saved input image for {prompt_id}: "
                    f"{img_path.stat().st_size} bytes"
                )

        logger.info(f"📁 Gen artifacts saved: {prompt_id} (user={user_id[:8]}…)")

    except Exception as e:
        logger.warning(f"⚠️ Failed to save gen artifacts for {prompt_id}: {e}")


# ── Save at job completion ──────────────────────────────────────────────────

def save_gen_logs(
    user_id: str,
    prompt_id: str,
    log_text: str,
    status: str = "completed",
    duration_seconds: Optional[float] = None,
) -> None:
    """Save execution logs for a completed generation.

    Called from record_generation_complete() for both local and cloud jobs.
    """
    try:
        storage = _get_storage()

        header = [
            f"# Generation Log: {prompt_id}",
            f"# Status: {status}",
            f"# Timestamp: {datetime.now().isoformat()}",
        ]
        if duration_seconds is not None:
            header.append(f"# Duration: {duration_seconds:.1f}s")
        header.append("")

        full_log = "\n".join(header) + log_text

        log_key = _gen_key(user_id, prompt_id, "execution.log")
        storage.put(BUCKET, log_key, full_log.encode("utf-8"))
        logger.info(f"📋 Gen log saved: {prompt_id} ({len(log_text)} chars)")

    except Exception as e:
        logger.warning(f"⚠️ Failed to save gen log for {prompt_id}: {e}")


# ── Extract execution log from ComfyUI history ─────────────────────────────

def format_comfyui_history_log(prompt_id: str, job_data: dict) -> str:
    """Format ComfyUI history data as a human-readable execution log.

    Used for local jobs where we don't have handler stdout,
    but we do have the ComfyUI history with execution trace.
    """
    lines = []

    # Status messages (execution trace)
    status = job_data.get("status", {})
    status_str = status.get("status_str", "unknown")
    lines.append(f"ComfyUI execution status: {status_str}")
    lines.append("")

    messages = status.get("messages", [])
    for msg_type, msg_data in messages:
        if msg_type == "execution_start":
            lines.append(f"[execution_start] prompt_id={msg_data.get('prompt_id', '?')}")
        elif msg_type == "execution_cached":
            cached = msg_data.get("nodes", [])
            if cached:
                lines.append(f"[execution_cached] {len(cached)} nodes cached: {cached}")
        elif msg_type == "executing":
            node = msg_data.get("node", "?")
            lines.append(f"[executing] node={node}")
        elif msg_type == "executed":
            node = msg_data.get("node", "?")
            lines.append(f"[executed] node={node}")
        elif msg_type == "execution_error":
            lines.append(f"[ERROR] {msg_data}")
        else:
            lines.append(f"[{msg_type}] {msg_data}")

    # Output summary
    outputs = job_data.get("outputs", {})
    if outputs:
        lines.append("")
        lines.append("--- Outputs ---")
        for node_id, node_output in outputs.items():
            if "gifs" in node_output:
                for gif in node_output["gifs"]:
                    lines.append(
                        f"  node {node_id}: video → {gif.get('filename', '?')}"
                    )
            if "images" in node_output:
                for img in node_output["images"]:
                    lines.append(
                        f"  node {node_id}: image → {img.get('filename', '?')}"
                    )
            if "audio" in node_output:
                for aud in node_output["audio"]:
                    lines.append(
                        f"  node {node_id}: audio → {aud.get('filename', '?')}"
                    )

    return "\n".join(lines) + "\n"
