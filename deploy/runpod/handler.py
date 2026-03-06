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
import io
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
# Source: Comfy-Org/Wan_2.2_ComfyUI_Repackaged on HuggingFace
# Wan 2.2 uses dedicated high/low noise models for better temporal coherence.
# fp8_scaled precision: near-bf16 quality, fits on 48GB GPUs (28.6GB total vs 57GB bf16).
HF_REPO_22 = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
HF_REPO_21 = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"  # CLIP Vision not in 2.2 repo
CLOUD_MAX_MODELS = [
    {
        "hf_repo": HF_REPO_22,
        "hf_path": "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        "local_dir": "unet",
        "filename": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        "size_gb": 14.3,
        "description": "Wan 2.2 I2V high noise 14B fp8_scaled",
        "required": True,
    },
    {
        "hf_repo": HF_REPO_22,
        "hf_path": "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        "local_dir": "unet",
        "filename": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        "size_gb": 14.3,
        "description": "Wan 2.2 I2V low noise 14B fp8_scaled",
        "required": True,
    },
    {
        "hf_repo": HF_REPO_22,
        "hf_path": "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        "local_dir": "unet",
        "filename": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        "size_gb": 14.3,
        "description": "Wan 2.2 T2V high noise 14B fp8_scaled",
        "required": False,  # Skip at startup — download on first T2V request
    },
    {
        "hf_repo": HF_REPO_22,
        "hf_path": "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        "local_dir": "unet",
        "filename": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        "size_gb": 14.3,
        "description": "Wan 2.2 T2V low noise 14B fp8_scaled",
        "required": False,  # Skip at startup — download on first T2V request
    },
    {
        "hf_repo": HF_REPO_22,
        "hf_path": "split_files/text_encoders/umt5_xxl_fp16.safetensors",
        "local_dir": "clip",
        "filename": "umt5_xxl_fp16.safetensors",
        "size_gb": 11.4,
        "description": "UMT5-XXL fp16 text encoder",
        "required": True,
    },
    {
        "hf_repo": HF_REPO_21,
        "hf_path": "split_files/vae/wan_2.1_vae.safetensors",
        "local_dir": "vae",
        "filename": "wan_2.1_vae.safetensors",
        "size_gb": 0.40,
        "description": "Wan 2.1 VAE — required for 14B models (wan2.2_vae is 5B only)",
        "required": True,
    },
    {
        "hf_repo": HF_REPO_21,
        "hf_path": "split_files/clip_vision/clip_vision_h.safetensors",
        "local_dir": "clip_vision",
        "filename": "clip_vision_h.safetensors",
        "size_gb": 1.26,
        "description": "CLIP Vision H (I2V conditioning)",
        "required": True,
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
        # Skip optional models at startup (e.g. T2V when no volume)
        if not model.get("required", True):
            logger.info(f"⏭️ {model['filename']} ({model['size_gb']}GB) — optional, skipping")
            continue

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
                repo_id=model.get("hf_repo", HF_REPO_22),
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

            # Clean HF cache after each download to free disk space
            shutil.rmtree("/tmp/hf_cache", ignore_errors=True)
        except Exception as e:
            logger.error(f"❌ Failed to download {model['filename']}: {e}")
            return False

    total_elapsed = time.time() - start
    logger.info(f"✅ All models downloaded in {total_elapsed:.0f}s "
               f"({total_to_download:.1f}GB)")
    return True


def _model_destinations(model: dict) -> tuple[Path, Path]:
    """Return volume and local destinations for a model file."""
    volume_models = Path(MODEL_VOLUME) / "models"
    comfyui_models = Path("/comfyui/models")
    return (
        volume_models / model["local_dir"] / model["filename"],
        comfyui_models / model["local_dir"] / model["filename"],
    )


def _is_model_present(model: dict) -> bool:
    """Check whether a model already exists in either persistent or local storage."""
    dest_vol, dest_local = _model_destinations(model)
    return dest_vol.exists() or dest_local.exists()


def download_requested_models(filenames: list[str]) -> int:
    """Download a specific subset of models on demand."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError("huggingface_hub not installed, cannot download models")

    requested = {name for name in filenames if name}
    models = [model for model in CLOUD_MAX_MODELS if model["filename"] in requested]
    if not models:
        return 0

    volume_models = Path(MODEL_VOLUME) / "models"
    comfyui_models = Path("/comfyui/models")
    target_base = volume_models if Path(MODEL_VOLUME).exists() else comfyui_models
    downloaded = 0

    for model in models:
        if _is_model_present(model):
            logger.info(f"✅ {model['filename']} already present")
            continue

        dest_dir = target_base / model["local_dir"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / model["filename"]
        logger.info(f"⬇️ On-demand model download: {model['filename']}")

        downloaded_path = hf_hub_download(
            repo_id=model.get("hf_repo", HF_REPO_22),
            filename=model["hf_path"],
            local_dir="/tmp/hf_cache",
            local_dir_use_symlinks=False,
        )

        import shutil

        shutil.move(downloaded_path, str(dest))
        shutil.rmtree("/tmp/hf_cache", ignore_errors=True)

        # If downloaded to volume, make sure ComfyUI sees it immediately.
        if target_base == volume_models:
            comfyui_target = comfyui_models / model["local_dir"] / model["filename"]
            comfyui_target.parent.mkdir(parents=True, exist_ok=True)
            if not comfyui_target.exists():
                comfyui_target.symlink_to(dest)
                logger.info(f"🔗 Symlinked on-demand model: {model['filename']}")

        downloaded += 1

    return downloaded


def restart_comfyui():
    """Restart ComfyUI so newly downloaded models appear in node input lists."""
    global _comfyui_process

    if _comfyui_process and _comfyui_process.poll() is None:
        logger.info("🔄 Restarting ComfyUI to reload model lists...")
        _comfyui_process.terminate()
        try:
            _comfyui_process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            _comfyui_process.kill()
            _comfyui_process.wait(timeout=10)

    if not start_comfyui():
        raise RuntimeError("ComfyUI failed to restart after downloading models")


def ensure_workflow_models(workflow: dict, job=None) -> int:
    """Ensure workflow-referenced optional models exist before queueing."""
    referenced = set()
    for node in workflow.values():
        inputs = node.get("inputs", {})
        for key in ("unet_name", "clip_name", "vae_name"):
            value = inputs.get(key)
            if isinstance(value, str):
                referenced.add(value)

    requested = [
        model["filename"]
        for model in CLOUD_MAX_MODELS
        if model["filename"] in referenced and not _is_model_present(model)
    ]
    if not requested:
        return 0

    if job:
        _progress(job, f"Downloading {len(requested)} required model(s) for workflow...")

    downloaded = download_requested_models(requested)
    if downloaded > 0:
        restart_comfyui()
    return downloaded


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

    # Clean up old Wan 2.1 models to free space on volume
    _cleanup_old_models(volume_models, comfyui_models)

    # Strategy 1: Network Volume already has models
    if volume_path.exists() and volume_models.exists():
        logger.info("📁 Network Volume detected, setting up symlinks...")
        if setup_model_links():
            # Verify at least the high noise diffusion model is available
            i2v_hi = comfyui_models / "unet" / "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
            if not i2v_hi.exists():
                i2v_hi = comfyui_models / "diffusion_models" / "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
            if i2v_hi.exists():
                logger.info("✅ Models loaded from Network Volume")
                return True
            logger.warning("⚠️ Network Volume found but missing Wan 2.2 models, will download")

    # Strategy 2: Download from HuggingFace (to volume if available, else container)
    logger.info("📥 Downloading models from HuggingFace...")
    if not download_models():
        return False

    # If downloaded to volume, set up symlinks to ComfyUI dirs
    if volume_path.exists() and volume_models.exists():
        setup_model_links()

    return True


def _cleanup_old_models(volume_models: Path, comfyui_models: Path):
    """Remove deprecated Wan 2.1 models to free disk space."""
    deprecated = [
        ("unet", "wan2.1_i2v_720p_14B_bf16.safetensors"),
        ("vae", "wan_2.1_vae.safetensors"),
    ]
    for subdir, filename in deprecated:
        for base in [volume_models, comfyui_models]:
            path = base / subdir / filename
            if path.exists():
                try:
                    if path.is_symlink():
                        path.unlink()
                        logger.info(f"🗑️ Removed symlink: {path}")
                    else:
                        size_gb = path.stat().st_size / (1024**3)
                        path.unlink()
                        logger.info(f"🗑️ Removed old model: {path} ({size_gb:.1f}GB freed)")
                except Exception as e:
                    logger.warning(f"⚠️ Could not remove {path}: {e}")


# ---- ComfyUI Process Management ----

_comfyui_process = None


def wait_for_cuda(max_wait: int = 60) -> bool:
    """
    Wait for CUDA to become available before starting ComfyUI.

    On RunPod serverless cold starts, the GPU may not be immediately
    available when the container starts. This check prevents ComfyUI
    from crashing with 'CUDA-capable device(s) is/are busy or unavailable'.
    """
    logger.info("🔍 Checking CUDA availability...")
    start = time.time()
    attempt = 0

    while (time.time() - start) < max_wait:
        attempt += 1
        try:
            result = subprocess.run(
                [sys.executable, "-c",
                 "import torch; "
                 "assert torch.cuda.is_available(), 'CUDA not available'; "
                 "d = torch.cuda.device_count(); "
                 "name = torch.cuda.get_device_name(0); "
                 "mem = torch.cuda.get_device_properties(0).total_mem / 1024**3; "
                 "print(f'OK|{d}|{name}|{mem:.1f}')"],
                capture_output=True, text=True, timeout=30,
                env={**os.environ, "CUDA_LAUNCH_BLOCKING": "1"},
            )
            if result.returncode == 0 and result.stdout.strip().startswith("OK|"):
                parts = result.stdout.strip().split("|")
                logger.info(f"✅ CUDA ready (attempt {attempt}): "
                          f"{parts[2]}, {parts[3]}GB VRAM, {int(parts[1])} device(s)")
                return True
            else:
                stderr = result.stderr.strip()[-200:] if result.stderr else "no output"
                logger.warning(f"⚠️ CUDA check attempt {attempt} failed: {stderr}")
        except subprocess.TimeoutExpired:
            logger.warning(f"⚠️ CUDA check attempt {attempt} timed out")
        except Exception as e:
            logger.warning(f"⚠️ CUDA check attempt {attempt} error: {e}")

        time.sleep(5)

    logger.error(f"❌ CUDA not available after {max_wait}s ({attempt} attempts)")
    return False


def start_comfyui(max_retries: int = 2):
    """
    Start ComfyUI server in background with retry logic.

    Retries if ComfyUI crashes during startup (e.g. transient CUDA errors).
    """
    global _comfyui_process

    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            logger.info(f"🔄 Retry {attempt}/{max_retries}: restarting ComfyUI...")
            # Kill leftover process if any
            if _comfyui_process and _comfyui_process.poll() is None:
                _comfyui_process.terminate()
                try:
                    _comfyui_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    _comfyui_process.kill()
            time.sleep(5)

        logger.info(f"🚀 Starting ComfyUI server (attempt {attempt}/{max_retries})...")

        env = {**os.environ, "CUDA_LAUNCH_BLOCKING": "1"}
        _comfyui_process = subprocess.Popen(
            [sys.executable, "main.py", "--listen", COMFYUI_HOST, "--port", str(COMFYUI_PORT),
             "--disable-auto-launch", "--disable-metadata"],
            cwd="/comfyui",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )

        # Stream ComfyUI logs in background thread
        def log_reader(proc=_comfyui_process):
            for line in iter(proc.stdout.readline, b''):
                logger.info(f"[ComfyUI] {line.decode().strip()}")
        threading.Thread(target=log_reader, daemon=True).start()

        # Wait for ComfyUI to be ready
        max_wait = 120  # seconds
        start = time.time()
        while (time.time() - start) < max_wait:
            # Check if process crashed
            if _comfyui_process.poll() is not None:
                logger.error(f"❌ ComfyUI process exited with code {_comfyui_process.returncode}")
                break

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

        # If we get here, this attempt failed
        if _comfyui_process.poll() is None:
            logger.error(f"❌ ComfyUI failed to start within {max_wait}s (attempt {attempt})")
        # Continue to next retry

    logger.error(f"❌ ComfyUI failed to start after {max_retries} attempts")
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


def download_loras(lora_downloads: list, job=None):
    """
    Download LoRA files from backend on demand for cloud jobs.
    Downloads to Network Volume (persistent cache) if available,
    otherwise to container's ComfyUI loras dir (ephemeral).

    Skips LoRAs that already exist (cached from previous jobs).
    """
    volume_loras = Path(MODEL_VOLUME) / "models" / "loras"
    comfyui_loras = Path("/comfyui/models/loras")

    # Prefer volume (persistent across jobs) over container disk
    target_dir = volume_loras if Path(MODEL_VOLUME).exists() else comfyui_loras
    target_dir.mkdir(parents=True, exist_ok=True)
    comfyui_loras.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    for lora in lora_downloads:
        filename = lora["filename"]
        url = lora["url"]
        target = target_dir / filename

        # Check if already present (volume cache or comfyui dir)
        comfyui_target = comfyui_loras / filename
        if target.exists() or comfyui_target.exists():
            logger.info(f"✅ LoRA cached: {filename}")
            continue

        logger.info(f"⬇️ Downloading LoRA: {filename}...")
        if job:
            _progress(job, f"Downloading LoRA: {filename}...")

        try:
            resp = requests.get(url, stream=True, timeout=600)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            tmp = target.with_suffix(".download")
            received = 0
            last_pct = -1
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):  # 8MB chunks
                    f.write(chunk)
                    received += len(chunk)
                    if total > 0:
                        pct = int(received / total * 100)
                        # Report progress every 25%
                        if pct >= last_pct + 25:
                            last_pct = pct
                            if job:
                                _progress(job, f"Downloading LoRA {filename}: {pct}%")
            tmp.rename(target)
            size_mb = target.stat().st_size / (1024 * 1024)
            logger.info(f"✅ LoRA downloaded: {filename} ({size_mb:.0f}MB)")

            # If saved to volume, also symlink into comfyui dir so ComfyUI finds it
            if target_dir == volume_loras and not comfyui_target.exists():
                comfyui_target.symlink_to(target)
                logger.info(f"🔗 Symlinked LoRA: {filename}")

            downloaded += 1
        except Exception as e:
            logger.error(f"❌ Failed to download LoRA {filename}: {e}")
            tmp = target.with_suffix(".download")
            if tmp.exists():
                tmp.unlink()
            raise RuntimeError(f"Failed to download LoRA {filename}: {e}")

    if downloaded > 0:
        logger.info(f"✅ Downloaded {downloaded} LoRA(s) on demand")
    return downloaded


def queue_workflow(workflow: dict) -> str:
    """Queue a workflow in ComfyUI and return the prompt_id."""
    resp = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow})
    if resp.status_code != 200:
        # Log full error details from ComfyUI
        try:
            error_body = resp.json()
        except Exception:
            error_body = resp.text[:2000]
        logger.error(f"❌ ComfyUI /prompt returned {resp.status_code}: {json.dumps(error_body, indent=2)[:2000]}")
        raise RuntimeError(f"ComfyUI rejected workflow ({resp.status_code}): {json.dumps(error_body)[:1000]}")
    data = resp.json()
    prompt_id = data.get("prompt_id")
    logger.info(f"📋 Queued workflow: {prompt_id}")
    return prompt_id


def wait_for_completion(prompt_id: str, timeout: int = 1800, job=None) -> dict:
    """
    Poll ComfyUI /history until the job is done.
    Returns the history entry for this prompt_id.
    Sends RunPod progress_update when new logs appear or every 10s.
    """
    global _log_buffer
    start = time.time()
    last_progress = 0
    last_log_len = len(_log_buffer.getvalue()) if _log_buffer else 0
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

        # Send progress when new ComfyUI logs appear (within 3s) or every 60s heartbeat
        elapsed = time.time() - start
        current_log_len = len(_log_buffer.getvalue()) if _log_buffer else 0
        has_new_logs = current_log_len > last_log_len
        if job and (has_new_logs or elapsed - last_progress >= 60):
            _progress(job, f"Generating... {elapsed:.0f}s elapsed", log_locally=has_new_logs)
            last_progress = elapsed
            # Re-read after _progress adds its own log line
            last_log_len = len(_log_buffer.getvalue()) if _log_buffer else 0

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

_log_buffer = None  # Set by handler(), read by _progress() for real-time log streaming


def _progress(job, message: str, log_locally: bool = True):
    """Send a progress update to RunPod with accumulated logs (visible when polling job status)."""
    global _log_buffer
    try:
        if _log_buffer:
            payload = {"message": message, "logs": _log_buffer.getvalue()}
        else:
            payload = message
        runpod.serverless.progress_update(job, payload)
        if log_locally:
            logger.info(f"📡 Progress: {message}")
    except Exception as e:
        logger.warning(f"⚠️ progress_update failed: {e}")


def handler(event: dict) -> dict:
    """
    Main RunPod handler function.

    Receives a workflow, queues it in ComfyUI, waits for completion,
    and returns the output files (base64 encoded).

    Progress updates are sent via RunPod API during execution.
    All logs are captured and returned in the 'logs' field.
    """
    # Capture logs during this job (also used by _progress for real-time log streaming)
    global _log_buffer
    log_buffer = io.StringIO()
    _log_buffer = log_buffer
    log_handler = logging.StreamHandler(log_buffer)
    log_handler.setLevel(logging.DEBUG)
    log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(log_handler)

    start_time = time.time()
    input_data = event.get("input", {})

    _progress(event, "Job received, validating workflow...")

    workflow = input_data.get("workflow")
    if not workflow:
        logger.removeHandler(log_handler)
        return {"error": "No workflow provided in input.workflow", "logs": log_buffer.getvalue()}

    # Save input images if provided
    images = input_data.get("images", {})
    if images:
        _progress(event, f"Saving {len(images)} input image(s)...")
        save_input_images(images)

    # Download LoRAs on demand if provided
    lora_downloads = input_data.get("lora_downloads", [])
    if lora_downloads:
        _progress(event, f"Downloading {len(lora_downloads)} LoRA(s)...")
        download_loras(lora_downloads, job=event)

    # Download optional workflow models on demand (e.g. Cloud Max T2V fp8 models)
    downloaded_models = ensure_workflow_models(workflow, job=event)
    if downloaded_models > 0:
        _progress(event, f"Reloaded ComfyUI after downloading {downloaded_models} model(s)")

    try:
        # Queue the workflow
        _progress(event, "Queuing workflow in ComfyUI...")
        prompt_id = queue_workflow(workflow)
        _progress(event, f"Workflow queued (prompt_id: {prompt_id}), generating...")

        # Wait for completion
        timeout = input_data.get("timeout", 1800)  # 30 min default
        history = wait_for_completion(prompt_id, timeout=timeout, job=event)

        # Collect outputs
        _progress(event, "Generation complete, collecting outputs...")
        files = collect_outputs(history)
        if not files:
            logger.removeHandler(log_handler)
            return {"error": "No output files generated", "prompt_id": prompt_id, "logs": log_buffer.getvalue()}

        # Encode outputs as base64
        total_size = sum(f["size"] for f in files)
        _progress(event, f"Encoding {len(files)} file(s) ({total_size / 1024 / 1024:.1f} MB)...")
        encoded_files = encode_outputs(files)

        elapsed = time.time() - start_time
        logger.info(f"✅ Job complete in {elapsed:.1f}s — {len(encoded_files)} files")
        _progress(event, f"Done! {len(encoded_files)} file(s) in {elapsed:.0f}s")

        logger.removeHandler(log_handler)
        return {
            "files": encoded_files,
            "prompt_id": prompt_id,
            "execution_time_s": round(elapsed, 1),
            "logs": log_buffer.getvalue(),
        }

    except TimeoutError as e:
        _progress(event, f"❌ Timeout: {e}")
        logger.removeHandler(log_handler)
        return {"error": str(e), "logs": log_buffer.getvalue()}
    except RuntimeError as e:
        _progress(event, f"❌ Error: {e}")
        logger.removeHandler(log_handler)
        return {"error": str(e), "logs": log_buffer.getvalue()}
    except Exception as e:
        logger.exception(f"❌ Handler error: {e}")
        _progress(event, f"❌ Unexpected error: {e}")
        logger.removeHandler(log_handler)
        return {"error": f"Unexpected error: {str(e)}", "logs": log_buffer.getvalue()}


# ---- Startup ----

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🎬 Oelala ComfyUI Worker starting...")
    logger.info("=" * 60)

    # Ensure models are available (Network Volume or download from HF)
    if not ensure_models():
        logger.error("❌ Failed to load models, exiting")
        sys.exit(1)

    # Wait for CUDA to be available (RunPod cold start may delay GPU readiness)
    if not wait_for_cuda(max_wait=60):
        logger.error("❌ CUDA not available, exiting")
        sys.exit(1)

    # Start ComfyUI (with retry on CUDA errors)
    if not start_comfyui(max_retries=2):
        logger.error("❌ Failed to start ComfyUI, exiting")
        sys.exit(1)

    # Start RunPod serverless handler
    logger.info("🎯 RunPod handler ready, waiting for jobs...")
    runpod.serverless.start({"handler": handler})
