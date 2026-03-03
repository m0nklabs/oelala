#!/usr/bin/env python3
"""
RunPod Serverless Handler for Oelala ComfyUI Worker

Receives ComfyUI workflow JSON via RunPod API, executes it on the
local ComfyUI instance, and returns the output (images/videos).

Input format:
{
    "input": {
        "workflow": { ... ComfyUI API-format workflow ... },
        "images": {            // optional: base64-encoded input images
            "input_image.png": "<base64>"
        }
    }
}

Output format:
{
    "status": "COMPLETED",
    "output": {
        "files": [
            {"filename": "output_00001.mp4", "url": "<presigned_url>", "type": "video/mp4"}
        ],
        "execution_time_s": 123.4
    }
}
"""

import os
import sys
import json
import time
import base64
import subprocess
import threading
import logging
import glob
from pathlib import Path

import requests
import runpod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("oelala-worker")

# ---- Configuration ----
COMFYUI_HOST = "127.0.0.1"
COMFYUI_PORT = 8188
COMFYUI_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"
OUTPUT_DIR = "/comfyui/output"
INPUT_DIR = "/comfyui/input"
MODEL_VOLUME = os.getenv("RUNPOD_VOLUME_PATH", "/runpod-volume")

# ---- Model symlinks ----

def setup_model_links():
    """
    Create symlinks from network volume models to ComfyUI model dirs.
    This allows models to be shared across workers without baking into the image.
    """
    volume_models = Path(MODEL_VOLUME) / "models"
    comfyui_models = Path("/comfyui/models")

    if not volume_models.exists():
        logger.warning(f"⚠️ No network volume at {volume_models}, using built-in models only")
        return

    # Map volume subdirs to ComfyUI model dirs
    model_dirs = [
        "checkpoints", "diffusion_models", "vae", "text_encoders",
        "clip", "loras", "upscale_models", "clip_vision",
    ]

    for d in model_dirs:
        src = volume_models / d
        dst = comfyui_models / d
        if src.exists():
            # Symlink individual files (don't replace entire dirs)
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                target = dst / f.name
                if not target.exists():
                    target.symlink_to(f)
                    logger.info(f"🔗 Linked model: {d}/{f.name}")

    logger.info("✅ Model symlinks configured")


# ---- ComfyUI Process Management ----

_comfyui_process = None

def start_comfyui():
    """Start ComfyUI server in background."""
    global _comfyui_process
    logger.info("🚀 Starting ComfyUI server...")

    _comfyui_process = subprocess.Popen(
        [sys.executable, "main.py", "--listen", COMFYUI_HOST, "--port", str(COMFYUI_PORT),
         "--disable-auto-launch", "--disable-metadata"],
        cwd="/comfyui",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    # Stream ComfyUI logs in background thread
    def log_reader():
        for line in iter(_comfyui_process.stdout.readline, b''):
            logger.info(f"[ComfyUI] {line.decode().strip()}")
    threading.Thread(target=log_reader, daemon=True).start()

    # Wait for ComfyUI to be ready
    max_wait = 120  # seconds
    start = time.time()
    while (time.time() - start) < max_wait:
        try:
            r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=2)
            if r.status_code == 200:
                stats = r.json()
                gpu = stats.get("devices", [{}])[0]
                logger.info(f"✅ ComfyUI ready! GPU: {gpu.get('name', 'unknown')} "
                          f"VRAM: {gpu.get('vram_total', 0) / 1024**3:.1f}GB")
                return True
        except Exception:
            pass
        time.sleep(2)

    logger.error("❌ ComfyUI failed to start within 120s")
    return False


def save_input_images(images: dict):
    """Save base64-encoded input images to ComfyUI input directory."""
    Path(INPUT_DIR).mkdir(parents=True, exist_ok=True)
    saved = []
    for filename, b64_data in images.items():
        filepath = Path(INPUT_DIR) / filename
        data = base64.b64decode(b64_data)
        filepath.write_bytes(data)
        saved.append(str(filepath))
        logger.info(f"📥 Saved input image: {filename} ({len(data)} bytes)")
    return saved


def queue_workflow(workflow: dict) -> str:
    """Queue a workflow in ComfyUI and return the prompt_id."""
    resp = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow})
    resp.raise_for_status()
    data = resp.json()
    prompt_id = data.get("prompt_id")
    logger.info(f"📋 Queued workflow: {prompt_id}")
    return prompt_id


def wait_for_completion(prompt_id: str, timeout: int = 1800) -> dict:
    """
    Poll ComfyUI /history until the job is done.
    Returns the history entry for this prompt_id.
    """
    start = time.time()
    while (time.time() - start) < timeout:
        try:
            resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
            if resp.status_code == 200:
                history = resp.json()
                entry = history.get(prompt_id)
                if entry:
                    status = entry.get("status", {})
                    if status.get("completed", False) or status.get("status_str") == "success":
                        logger.info(f"✅ Job {prompt_id} completed!")
                        return entry
                    if status.get("status_str") == "error":
                        error_msg = status.get("messages", [["", "Unknown error"]])[-1][-1]
                        logger.error(f"❌ Job {prompt_id} failed: {error_msg}")
                        raise RuntimeError(f"ComfyUI job failed: {error_msg}")
        except requests.exceptions.RequestException:
            pass
        time.sleep(3)

    raise TimeoutError(f"Job {prompt_id} timed out after {timeout}s")


def collect_outputs(history_entry: dict) -> list:
    """
    Extract output files from ComfyUI history entry.
    Returns list of {filename, path, type, size} dicts.
    """
    outputs_node = history_entry.get("outputs", {})
    files = []

    for node_id, node_output in outputs_node.items():
        # Check for images
        for img in node_output.get("images", []):
            filename = img.get("filename", "")
            subfolder = img.get("subfolder", "")
            filepath = Path(OUTPUT_DIR) / subfolder / filename if subfolder else Path(OUTPUT_DIR) / filename
            if filepath.exists():
                files.append({
                    "filename": filename,
                    "path": str(filepath),
                    "type": "image/png" if filename.endswith(".png") else "image/jpeg",
                    "size": filepath.stat().st_size,
                })

        # Check for gifs (VHS video output)
        for gif in node_output.get("gifs", []):
            filename = gif.get("filename", "")
            subfolder = gif.get("subfolder", "")
            filepath = Path(OUTPUT_DIR) / subfolder / filename if subfolder else Path(OUTPUT_DIR) / filename
            if filepath.exists():
                mime = "video/mp4" if filename.endswith(".mp4") else "video/webm"
                files.append({
                    "filename": filename,
                    "path": str(filepath),
                    "type": mime,
                    "size": filepath.stat().st_size,
                })

    logger.info(f"📦 Collected {len(files)} output files")
    return files


def encode_outputs(files: list) -> list:
    """Encode output files as base64 for API response."""
    encoded = []
    for f in files:
        path = Path(f["path"])
        if path.exists():
            b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
            encoded.append({
                "filename": f["filename"],
                "data": b64,
                "type": f["type"],
                "size": f["size"],
            })
            logger.info(f"📤 Encoded: {f['filename']} ({f['size']} bytes)")
    return encoded


# ---- RunPod Handler ----

def handler(event: dict) -> dict:
    """
    Main RunPod handler function.

    Receives a workflow, queues it in ComfyUI, waits for completion,
    and returns the output files (base64 encoded).
    """
    start_time = time.time()
    input_data = event.get("input", {})

    workflow = input_data.get("workflow")
    if not workflow:
        return {"error": "No workflow provided in input.workflow"}

    # Save input images if provided
    images = input_data.get("images", {})
    if images:
        save_input_images(images)

    try:
        # Queue the workflow
        prompt_id = queue_workflow(workflow)

        # Wait for completion
        timeout = input_data.get("timeout", 1800)  # 30 min default
        history = wait_for_completion(prompt_id, timeout=timeout)

        # Collect outputs
        files = collect_outputs(history)
        if not files:
            return {"error": "No output files generated", "prompt_id": prompt_id}

        # Encode outputs as base64
        encoded_files = encode_outputs(files)

        elapsed = time.time() - start_time
        logger.info(f"✅ Job complete in {elapsed:.1f}s — {len(encoded_files)} files")

        return {
            "files": encoded_files,
            "prompt_id": prompt_id,
            "execution_time_s": round(elapsed, 1),
        }

    except TimeoutError as e:
        return {"error": str(e)}
    except RuntimeError as e:
        return {"error": str(e)}
    except Exception as e:
        logger.exception(f"❌ Handler error: {e}")
        return {"error": f"Unexpected error: {str(e)}"}


# ---- Startup ----

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🎬 Oelala ComfyUI Worker starting...")
    logger.info("=" * 60)

    # Setup model symlinks from network volume
    setup_model_links()

    # Start ComfyUI
    if not start_comfyui():
        logger.error("Failed to start ComfyUI, exiting")
        sys.exit(1)

    # Start RunPod serverless handler
    logger.info("🎯 RunPod handler ready, waiting for jobs...")
    runpod.serverless.start({"handler": handler})
