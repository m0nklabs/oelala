#!/usr/bin/env python3
"""
RunPod Serverless Handler for Oelala ComfyUI Worker

Receives ComfyUI workflow JSON via RunPod API, executes it on the
local ComfyUI instance, and returns the output (images/videos).

Supports two model loading strategies:
1. Network Volume — models pre-loaded, symlinked at startup (fast)
2. Download-at-startup — models downloaded from HuggingFace on first boot
   (slower cold start but no monthly Network Volume cost)

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

# ---- Cloud Max model definitions ----
# Source: Comfy-Org/Wan_2.1_ComfyUI_repackaged on HuggingFace
HF_REPO = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
CLOUD_MAX_MODELS = [
    {
        "hf_path": "split_files/diffusion_models/wan2.1_i2v_720p_14B_bf16.safetensors",
        "local_dir": "unet",
        "filename": "wan2.1_i2v_720p_14B_bf16.safetensors",
        "size_gb": 32.8,
        "description": "I2V 720p bf16 diffusion model",
    },
    {
        "hf_path": "split_files/diffusion_models/wan2.1_t2v_14B_bf16.safetensors",
        "local_dir": "unet",
        "filename": "wan2.1_t2v_14B_bf16.safetensors",
        "size_gb": 28.6,
        "description": "T2V bf16 diffusion model",
    },
    {
        "hf_path": "split_files/text_encoders/umt5_xxl_fp16.safetensors",
        "local_dir": "clip",
        "filename": "umt5_xxl_fp16.safetensors",
        "size_gb": 11.4,
        "description": "UMT5-XXL fp16 text encoder",
    },
    {
        "hf_path": "split_files/vae/wan_2.1_vae.safetensors",
        "local_dir": "vae",
        "filename": "wan_2.1_vae.safetensors",
        "size_gb": 0.25,
        "description": "Wan 2.1 VAE",
    },
    {
        "hf_path": "split_files/clip_vision/clip_vision_h.safetensors",
        "local_dir": "clip_vision",
        "filename": "clip_vision_h.safetensors",
        "size_gb": 1.26,
        "description": "CLIP Vision H (I2V conditioning)",
    },
]


# ---- Model Setup ----

def setup_model_links():
    """
    Create symlinks from network volume models to ComfyUI model dirs.
    Fallback for deployments using RunPod Network Volume.
    """
    volume_models = Path(MODEL_VOLUME) / "models"
    comfyui_models = Path("/comfyui/models")

    if not volume_models.exists():
        logger.info(f"ℹ️ No network volume at {volume_models}")
        return False

    model_dirs = [
        "checkpoints", "diffusion_models", "unet", "vae", "text_encoders",
        "clip", "loras", "upscale_models", "clip_vision",
    ]

    linked = 0
    for d in model_dirs:
        src = volume_models / d
        dst = comfyui_models / d
        if src.exists():
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                target = dst / f.name
                if not target.exists():
                    target.symlink_to(f)
                    logger.info(f"🔗 Linked: {d}/{f.name}")
                    linked += 1

    logger.info(f"✅ Model symlinks: {linked} files linked")
    return linked > 0


def download_models():
    """
    Download Cloud Max bf16 models from HuggingFace if not already present.
    Uses huggingface_hub for efficient downloading with resume support.

    Downloads to Network Volume if mounted (persistent), otherwise to
    container disk (lost on restart).
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("❌ huggingface_hub not installed, cannot download models")
        return False

    # Determine target: prefer Network Volume (persistent) over container disk
    volume_models = Path(MODEL_VOLUME) / "models"
    comfyui_models = Path("/comfyui/models")

    if Path(MODEL_VOLUME).exists():
        target_base = volume_models
        logger.info(f"📁 Downloading to Network Volume: {target_base}")
    else:
        target_base = comfyui_models
        logger.info(f"📁 Downloading to container disk: {target_base} (NOT persistent!)")

    total_to_download = 0
    models_needed = []

    # Check which models are missing (check both volume and comfyui dirs)
    for model in CLOUD_MAX_MODELS:
        dest_vol = volume_models / model["local_dir"] / model["filename"]
        dest_local = comfyui_models / model["local_dir"] / model["filename"]
        if dest_vol.exists() or dest_local.exists():
            logger.info(f"✅ {model['filename']} ({model['size_gb']}GB) — already present")
        else:
            models_needed.append(model)
            total_to_download += model["size_gb"]

    if not models_needed:
        logger.info("✅ All Cloud Max models already downloaded")
        return True

    logger.info(f"📦 Downloading {len(models_needed)} models ({total_to_download:.1f}GB total)...")
    start = time.time()

    for i, model in enumerate(models_needed, 1):
        dest_dir = target_base / model["local_dir"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / model["filename"]

        logger.info(f"⬇️ [{i}/{len(models_needed)}] {model['description']} "
                    f"({model['size_gb']}GB)...")

        try:
            dl_start = time.time()
            downloaded_path = hf_hub_download(
                repo_id=HF_REPO,
                filename=model["hf_path"],
                local_dir="/tmp/hf_cache",
                local_dir_use_symlinks=False,
            )
            # Move to ComfyUI model dir
            import shutil
            shutil.move(downloaded_path, str(dest))
            elapsed = time.time() - dl_start
            speed = model["size_gb"] / elapsed * 1024 if elapsed > 0 else 0
            logger.info(f"✅ {model['filename']} downloaded in {elapsed:.0f}s "
                       f"({speed:.0f} MB/s)")
        except Exception as e:
            logger.error(f"❌ Failed to download {model['filename']}: {e}")
            return False

    # Clean up HF cache
    import shutil
    shutil.rmtree("/tmp/hf_cache", ignore_errors=True)

    total_elapsed = time.time() - start
    logger.info(f"✅ All models downloaded in {total_elapsed:.0f}s "
               f"({total_to_download:.1f}GB)")
    return True


def ensure_models():
    """
    Ensure all required models are available. Tries strategies in order:
    1. Network Volume symlinks (instant, if volume has models)
    2. Download from HuggingFace to Network Volume (persistent, slow first time)
    3. Download from HuggingFace to container disk (non-persistent fallback)
    """
    comfyui_models = Path("/comfyui/models")
    volume_path = Path(MODEL_VOLUME)
    volume_models = volume_path / "models"

    # Strategy 1: Network Volume already has models
    if volume_path.exists() and volume_models.exists():
        logger.info("📁 Network Volume detected, setting up symlinks...")
        if setup_model_links():
            # Verify at least the diffusion model is available
            i2v = comfyui_models / "unet" / "wan2.1_i2v_720p_14B_bf16.safetensors"
            if not i2v.exists():
                i2v = comfyui_models / "diffusion_models" / "wan2.1_i2v_720p_14B_bf16.safetensors"
            if i2v.exists():
                logger.info("✅ Models loaded from Network Volume")
                return True
            logger.warning("⚠️ Network Volume found but missing Cloud Max models, will download")

    # Strategy 2: Download from HuggingFace (to volume if available, else container)
    logger.info("📥 Downloading models from HuggingFace...")
    if not download_models():
        return False

    # If downloaded to volume, set up symlinks to ComfyUI dirs
    if volume_path.exists() and volume_models.exists():
        setup_model_links()

    return True


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

    # Ensure models are available (Network Volume or download from HF)
    if not ensure_models():
        logger.error("❌ Failed to load models, exiting")
        sys.exit(1)

    # Start ComfyUI
    if not start_comfyui():
        logger.error("❌ Failed to start ComfyUI, exiting")
        sys.exit(1)

    # Start RunPod serverless handler
    logger.info("🎯 RunPod handler ready, waiting for jobs...")
    runpod.serverless.start({"handler": handler})
