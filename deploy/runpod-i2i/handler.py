"""
RunPod Serverless Handler — I2I Worker (SDXL/Pony/Qwen)
========================================================
Downloads models from HuggingFace Hub, starts ComfyUI,
and processes instruction-based image editing and transformations.

Target: 48 GB+ VRAM GPUs (A6000, A40, L40S, A100, H100)

Models:
  - qwen_image_edit_2511_fp8mixed.safetensors  (19.1 GB, Comfy-Org/Qwen-Image-Edit_ComfyUI)
    - jib_mix_qwen_v6_fp8.safetensors            (19.0 GB, optional JIB Mix Qwen V6 variant)
  - qwen_2.5_vl_7b_fp8_scaled.safetensors      (8.8 GB, Comfy-Org/Qwen-Image_ComfyUI)
  - qwen_image_vae.safetensors                  (243 MB, Comfy-Org/Qwen-Image_ComfyUI)
  - Lightning LoRA (optional, 850 MB, lightx2v/Qwen-Image-Edit-2511-Lightning)
"""

import base64
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
import runpod

# ---- Logging ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("i2i-handler")

# ---- Constants ----
COMFYUI_PORT = 8188
COMFYUI_URL = f"http://127.0.0.1:{COMFYUI_PORT}"
COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "/comfyui")
MODEL_VOLUME = os.environ.get("MODEL_VOLUME", "/runpod-volume")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Qwen Edit & SDXL generate images, much faster than video — shorter timeouts
COMFYUI_STARTUP_TIMEOUT = 180  # 3 minutes
WORKFLOW_TIMEOUT = 300  # 5 minutes (image gen, not video)

# Minimum free disk to attempt a download (GB)
MIN_FREE_DISK_GB = 5.0


@dataclass
class WorkflowModelPrepResult:
    """Result of model preparation before workflow execution."""
    models_ready: bool = False
    error: Optional[str] = None
    download_time_s: float = 0.0
    models_downloaded: List[str] = field(default_factory=list)
    models_linked: List[str] = field(default_factory=list)


# ---- Model Definitions ----

CLOUD_I2I_MODELS = [
    {
        "filename": "qwen_image_edit_2511_fp8mixed.safetensors",
        "repo": "Comfy-Org/Qwen-Image-Edit_ComfyUI",
        "hf_path": "split_files/diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors",
        "target_dir": "diffusion_models",
        "size_gb": 19.1,
        "description": "Qwen Image Edit 2511 UNET (fp8 mixed precision)",
        "startup_required": True,
    },
    {
        "filename": "qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "repo": "Comfy-Org/Qwen-Image_ComfyUI",
        "hf_path": "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors",
        "target_dir": "text_encoders",
        "size_gb": 8.8,
        "description": "Qwen 2.5 VL 7B CLIP/text encoder (fp8 scaled)",
        "startup_required": True,
    },
    {
        "filename": "qwen_image_vae.safetensors",
        "repo": "Comfy-Org/Qwen-Image_ComfyUI",
        "hf_path": "split_files/vae/qwen_image_vae.safetensors",
        "target_dir": "vae",
        "size_gb": 0.24,
        "description": "Qwen Image VAE",
        "startup_required": True,
    },
    {
        "filename": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "repo": "lightx2v/Qwen-Image-Edit-2511-Lightning",
        "hf_path": "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors",
        "target_dir": "loras",
        "size_gb": 0.85,
        "description": "Lightning LoRA for 4-step fast generation",
        "startup_required": False,
    },

    {
        "filename": "ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
        "url": "https://civitai.red/api/download/models/290640?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "target_dir": "checkpoints",
        "size_gb": 6.8,
        "description": "Pony Diffusion V6 XL",
        "startup_required": False,
    },
    {
        "filename": "juggernautXL_ragnarok.safetensors",
        "url": "https://civitai.red/api/download/models/691060?type=Model&format=SafeTensor&size=pruned&fp=fp16",
        "target_dir": "checkpoints",
        "size_gb": 6.8,
        "description": "Juggernaut XL Ragnarok",
        "startup_required": False,
    }
]

PUBLIC_MODEL_FILENAMES = {model["filename"] for model in CLOUD_I2I_MODELS}
MODEL_CATALOG_BY_FILENAME = {model["filename"]: model for model in CLOUD_I2I_MODELS}

# Directories where ComfyUI looks for models
NODE_MANAGED_MODEL_DIRS = [
    "diffusion_models",
    "loras",
    "text_encoders",
    "vae",
]

# Workflow input keys that might reference model filenames
WORKFLOW_MODEL_INPUT_KEYS = (
    "unet_name",
    "clip_name",
    "vae_name",
    "lora_name",
)


def collect_required_model_filenames(workflow: Dict[str, Any]) -> set[str]:
    """Infer which known model files are required by the submitted workflow."""
    required: set[str] = set()

    for node in workflow.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue

        for key in WORKFLOW_MODEL_INPUT_KEYS:
            value = inputs.get(key)
            if isinstance(value, str) and value in MODEL_CATALOG_BY_FILENAME:
                required.add(value)

    return required


# ---- Model Setup ----

def setup_model_links():
    """
    Create symlinks from network volume assets to ComfyUI model dirs.

    Public models are never linked from the volume — they come from HF Hub.
    Volume is reserved for LoRAs and private/custom assets only.
    """
    volume_models = Path(MODEL_VOLUME) / "models"
    comfyui_models = Path(COMFYUI_PATH) / "models"

    if not volume_models.exists():
        logger.info("📁 No network volume mounted — skipping volume links")
        return

    for model_dir in NODE_MANAGED_MODEL_DIRS:
        volume_dir = volume_models / model_dir
        comfyui_dir = comfyui_models / model_dir

        if not volume_dir.exists():
            continue

        comfyui_dir.mkdir(parents=True, exist_ok=True)

        for model_file in volume_dir.iterdir():
            if not model_file.is_file():
                continue

            # Skip public models — those are downloaded from HF, not volume
            if model_file.name in PUBLIC_MODEL_FILENAMES:
                continue

            target = comfyui_dir / model_file.name
            if not target.exists():
                target.symlink_to(model_file)
                logger.info(f"🔗 Linked volume asset: {model_dir}/{model_file.name}")


def ensure_model_directories():
    """Create all model directories."""
    comfyui_models = Path(COMFYUI_PATH) / "models"
    for d in NODE_MANAGED_MODEL_DIRS:
        (comfyui_models / d).mkdir(parents=True, exist_ok=True)


def _check_download_capacity(size_gb: float) -> bool:
    """Check if there is enough disk space for a download."""
    try:
        usage = shutil.disk_usage("/")
        free_gb = usage.free / (1024 ** 3)
        return free_gb >= (size_gb + MIN_FREE_DISK_GB)
    except Exception:
        return True  # Optimistic fallback


def _find_cached_model(filename: str) -> Optional[Path]:
    """
    Search for a cached copy of the model in RunPod's cache/volume hierarchy.
    RunPod caches models across cold starts in certain locations.
    """
    search_paths = [
        Path(MODEL_VOLUME) / "models",
        Path("/runpod-volume"),
        Path("/workspace"),
        Path("/tmp/comfyui_cache"),
    ]
    for base in search_paths:
        if not base.exists():
            continue
        # Direct file match
        for model_dir in NODE_MANAGED_MODEL_DIRS:
            candidate = base / model_dir / filename
            if candidate.exists() and candidate.stat().st_size > 1_000_000:
                return candidate
        # Recursive search (slow, use sparingly)
        for candidate in base.rglob(filename):
            if candidate.stat().st_size > 1_000_000:
                return candidate
    return None


def download_model(model: Dict[str, Any]) -> bool:
    """Download a single model file from HuggingFace Hub."""
    filename = model["filename"]
    target_dir = model["target_dir"]
    size_gb = model["size_gb"]

    target = Path(COMFYUI_PATH) / "models" / target_dir / filename

    # Already present?
    if target.exists() and target.stat().st_size > 1_000_000:
        logger.info(f"✅ Model already present: {target_dir}/{filename}")
        return True

    # Check for cached copy
    cached = _find_cached_model(filename)
    if cached:
        logger.info(f"📦 Found cached model: {cached}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(cached)
        logger.info(f"🔗 Linked cached: {target_dir}/{filename}")
        return True

    # Check disk space
    if not _check_download_capacity(size_gb):
        logger.error(f"❌ Not enough disk space for {filename} ({size_gb:.1f} GB)")
        return False

    target.parent.mkdir(parents=True, exist_ok=True)

    if model.get("url"):
        logger.info(f"⬇️  Downloading {filename} ({size_gb:.1f} GB) from direct URL...")
        try:
            tmp = target.with_suffix(".download")
            with requests.get(
                model["url"],
                stream=True,
                timeout=600,
                headers={"User-Agent": "Mozilla/5.0"},
                allow_redirects=True,
            ) as resp:
                resp.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
            tmp.rename(target)

            if target.exists() and target.stat().st_size > 1_000_000:
                logger.info(f"✅ Downloaded: {target_dir}/{filename}")
                return True

            logger.error(f"❌ Direct download produced no valid file: {target}")
            return False
        except Exception as e:
            logger.error(f"❌ Failed to download {filename}: {e}")
            return False

    # Download from HuggingFace
    repo = model["repo"]
    hf_path = model["hf_path"]
    logger.info(f"⬇️  Downloading {filename} ({size_gb:.1f} GB) from {repo}...")
    try:
        from huggingface_hub import hf_hub_download

        kwargs = {
            "repo_id": repo,
            "filename": hf_path,
            "local_dir": str(target.parent),
            "local_dir_use_symlinks": False,
        }
        if HF_TOKEN:
            kwargs["token"] = HF_TOKEN

        downloaded_path = hf_hub_download(**kwargs)

        # hf_hub_download may put it in a subdir — move it
        downloaded = Path(downloaded_path)
        if downloaded != target and downloaded.exists():
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(downloaded), str(target))
                logger.info(f"📂 Moved {downloaded} → {target}")

        if target.exists() and target.stat().st_size > 1_000_000:
            logger.info(f"✅ Downloaded: {target_dir}/{filename}")
            return True
        else:
            logger.error(f"❌ Download produced no valid file: {target}")
            return False

    except Exception as e:
        logger.error(f"❌ Failed to download {filename}: {e}")
        return False


def ensure_models(workflow: Optional[Dict[str, Any]] = None) -> WorkflowModelPrepResult:
    """Download startup models plus any optional model referenced by the workflow."""
    result = WorkflowModelPrepResult()
    start = time.time()

    ensure_model_directories()
    setup_model_links()

    required_filenames = {
        model["filename"] for model in CLOUD_I2I_MODELS if model.get("startup_required", False)
    }
    if workflow:
        required_filenames.update(collect_required_model_filenames(workflow))

    for model in CLOUD_I2I_MODELS:
        if model["filename"] not in required_filenames:
            continue
        if download_model(model):
            result.models_downloaded.append(model["filename"])
        elif model.get("startup_required", False):
            result.error = f"Required model missing: {model['filename']}"
            result.download_time_s = time.time() - start
            return result

    result.models_ready = True
    result.download_time_s = time.time() - start
    logger.info(
        f"✅ All models ready in {result.download_time_s:.1f}s "
        f"({len(result.models_downloaded)} files)"
    )
    return result


# ---- ComfyUI Process Management ----

_comfyui_process: Optional[subprocess.Popen] = None


def start_comfyui() -> bool:
    """Start ComfyUI as a subprocess and wait for it to be ready."""
    global _comfyui_process

    if _comfyui_process and _comfyui_process.poll() is None:
        try:
            r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=5)
            if r.status_code == 200:
                logger.info("✅ ComfyUI already running")
                return True
        except Exception:
            logger.warning("⚠️ ComfyUI process exists but not responding, restarting")
            _comfyui_process.terminate()
            _comfyui_process.wait(timeout=10)

    logger.info("🚀 Starting ComfyUI...")
    cmd = [
        sys.executable, "main.py",
        "--listen", "127.0.0.1",
        "--port", str(COMFYUI_PORT),
        "--disable-auto-launch",
        "--disable-metadata",
    ]
    _comfyui_process = subprocess.Popen(
        cmd,
        cwd=COMFYUI_PATH,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for ComfyUI to become responsive
    deadline = time.time() + COMFYUI_STARTUP_TIMEOUT
    while time.time() < deadline:
        if _comfyui_process.poll() is not None:
            output = _comfyui_process.stdout.read() if _comfyui_process.stdout else ""
            logger.error(f"❌ ComfyUI exited during startup:\n{output[-2000:]}")
            return False

        try:
            r = requests.get(f"{COMFYUI_URL}/system_stats", timeout=3)
            if r.status_code == 200:
                logger.info("✅ ComfyUI ready!")
                return True
        except Exception:
            pass

        time.sleep(2)

    logger.error(f"❌ ComfyUI did not start within {COMFYUI_STARTUP_TIMEOUT}s")
    return False


def wait_for_cuda() -> bool:
    """Quick check that CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            logger.info(f"🖥️  CUDA device: {dev} ({mem:.1f} GB)")
            return True
        logger.error("❌ No CUDA device available")
        return False
    except Exception as e:
        logger.error(f"❌ CUDA check failed: {e}")
        return False


# ---- Workflow Processing ----

def save_input_images(images: Dict[str, str]) -> Dict[str, str]:
    """
    Save base64-encoded input images to ComfyUI's input directory.
    Returns mapping of original name → saved filename.
    """
    saved = {}
    input_dir = Path(COMFYUI_PATH) / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    for name, b64data in images.items():
        try:
            if "," in b64data:
                b64data = b64data.split(",", 1)[1]

            img_bytes = base64.b64decode(b64data)
            ext = ".png"
            if img_bytes[:3] == b"\xff\xd8\xff":
                ext = ".jpg"
            elif img_bytes[:4] == b"\x89PNG":
                ext = ".png"
            elif img_bytes[:4] == b"RIFF":
                ext = ".webp"

            safe_name = f"input_{uuid.uuid4().hex[:8]}{ext}"
            save_path = input_dir / safe_name
            save_path.write_bytes(img_bytes)
            saved[name] = safe_name
            logger.info(f"💾 Saved input image: {safe_name} ({len(img_bytes)} bytes)")
        except Exception as e:
            logger.error(f"❌ Failed to save image '{name}': {e}")

    return saved


def download_loras(lora_downloads: List[Dict[str, Any]]) -> bool:
    """Download LoRA files from backend URL for cloud jobs."""
    if not lora_downloads:
        return True

    lora_dir = Path(COMFYUI_PATH) / "models" / "loras"
    lora_dir.mkdir(parents=True, exist_ok=True)

    for lora in lora_downloads:
        filename = lora.get("filename", "")
        url = lora.get("url", "")

        if not filename or not url:
            logger.warning(f"⚠️ Skipping LoRA entry with missing filename/url: {lora}")
            continue

        target = lora_dir / filename
        if target.exists():
            logger.info(f"✅ LoRA already present: {filename}")
            continue

        logger.info(f"⬇️  Downloading LoRA: {filename}...")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            headers = {}
            hf_token = lora.get("hf_token", "")
            if hf_token:
                headers["Authorization"] = f"Bearer {hf_token}"
            resp = requests.get(url, stream=True, timeout=600, headers=headers)
            resp.raise_for_status()

            tmp = target.with_suffix(".download")
            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                    f.write(chunk)
            tmp.rename(target)

            size_mb = target.stat().st_size / (1024 * 1024)
            logger.info(f"✅ LoRA downloaded: {filename} ({size_mb:.0f}MB)")
        except Exception as e:
            logger.error(f"❌ Failed to download LoRA {filename}: {e}")
            return False

    return True


def queue_workflow(workflow: Dict[str, Any]) -> Optional[str]:
    """Queue a workflow on ComfyUI and return the prompt_id."""
    client_id = uuid.uuid4().hex
    payload = {"prompt": workflow, "client_id": client_id}

    try:
        resp = requests.post(f"{COMFYUI_URL}/prompt", json=payload, timeout=120)
        if resp.status_code == 200:
            prompt_id = resp.json().get("prompt_id")
            logger.info(f"📋 Workflow queued: {prompt_id}")
            return prompt_id
        else:
            error_text = resp.text[:500]
            logger.error(f"❌ Queue failed ({resp.status_code}): {error_text}")
            return None
    except Exception as e:
        logger.error(f"❌ Queue request failed: {e}")
        return None


def wait_for_completion(prompt_id: str, timeout: int = 0) -> bool:
    """Wait for a prompt to complete. Polls /history/{prompt_id}."""
    if timeout <= 0:
        timeout = WORKFLOW_TIMEOUT

    deadline = time.time() + timeout
    poll_interval = 2.0
    last_status = ""

    while time.time() < deadline:
        try:
            resp = requests.get(
                f"{COMFYUI_URL}/history/{prompt_id}", timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                if prompt_id in data:
                    status = data[prompt_id].get("status", {})
                    completed = status.get("completed", False)
                    status_str = status.get("status_str", "unknown")

                    if completed:
                        logger.info(f"✅ Workflow completed: {prompt_id}")
                        return True

                    if status_str == "error":
                        msgs = status.get("messages", [])
                        logger.error(f"❌ Workflow error: {msgs}")
                        return False

                    if status_str != last_status:
                        logger.info(f"⏳ Status: {status_str}")
                        last_status = status_str

        except Exception as e:
            logger.warning(f"⚠️ Poll error: {e}")

        # Adaptive polling
        elapsed = timeout - (deadline - time.time())
        if elapsed > 60:
            poll_interval = min(poll_interval + 0.5, 8.0)

        time.sleep(poll_interval)

    logger.error(f"❌ Workflow timed out after {timeout}s")
    return False


def collect_outputs(prompt_id: str) -> List[Dict[str, Any]]:
    """Collect output files (images) from a completed workflow."""
    outputs = []

    try:
        resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}", timeout=10)
        if resp.status_code != 200:
            logger.error(f"❌ Failed to get history: {resp.status_code}")
            return outputs

        data = resp.json()
        prompt_data = data.get(prompt_id, {})
        node_outputs = prompt_data.get("outputs", {})

        output_dir = Path(COMFYUI_PATH) / "output"

        for node_id, node_out in node_outputs.items():
            # Check for image files (SaveImage node output)
            images_out = node_out.get("images", [])
            for img in images_out:
                filename = img.get("filename", "")
                subfolder = img.get("subfolder", "")
                filepath = output_dir / subfolder / filename if subfolder else output_dir / filename

                if filepath.exists():
                    file_bytes = filepath.read_bytes()
                    b64 = base64.b64encode(file_bytes).decode("utf-8")

                    ext = filepath.suffix.lower()
                    mime = f"image/{ext.lstrip('.')}"

                    outputs.append({
                        "filename": filename,
                        "data": b64,
                        "type": mime,
                        "size_bytes": len(file_bytes),
                    })
                    logger.info(
                        f"📦 Collected output: {filename} "
                        f"({len(file_bytes) / 1024 / 1024:.1f} MB)"
                    )

            # Check for video files (unlikely for Qwen Edit, but handle anyway)
            gifs = node_out.get("gifs", [])
            for gif in gifs:
                filename = gif.get("filename", "")
                subfolder = gif.get("subfolder", "")
                filepath = output_dir / subfolder / filename if subfolder else output_dir / filename

                if filepath.exists():
                    file_bytes = filepath.read_bytes()
                    b64 = base64.b64encode(file_bytes).decode("utf-8")
                    ext = filepath.suffix.lower()
                    mime = "video/mp4" if ext == ".mp4" else f"video/{ext.lstrip('.')}"

                    outputs.append({
                        "filename": filename,
                        "data": b64,
                        "type": mime,
                        "size_bytes": len(file_bytes),
                    })

    except Exception as e:
        logger.error(f"❌ Failed to collect outputs: {e}")

    # Qwen Edit model produces reference reconstruction + edit result.
    # Only return the LAST image — that's the actual edited output.
    if len(outputs) > 1:
        image_outputs = [o for o in outputs if o.get("type", "").startswith("image/")]
        if len(image_outputs) > 1:
            logger.info(
                f"🔍 I2I produced {len(image_outputs)} images, "
                f"keeping only the last (edited result)"
            )
            outputs = [image_outputs[-1]]

    return outputs


# ---- Main Handler ----

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler for Image-to-Image workflows.

    Expected input:
    {
        "workflow": { ... ComfyUI API workflow ... },
        "images": { "name": "base64data", ... },
        "lora_downloads": [ {"filename": "...", "url": "https://..."} ],  # optional
    }
    """
    job_id = event.get("id", "unknown")
    job_input = event.get("input", {})
    log_lines = []

    logger.info(f"🎨 Job started: {job_id}")
    job_start = time.time()

    try:
        # 1. Extract inputs
        workflow = job_input.get("workflow")
        if not workflow:
            return {"error": "No workflow provided"}

        images = job_input.get("images", {})
        lora_downloads = job_input.get("lora_downloads", [])

        # 2. Check CUDA
        if not wait_for_cuda():
            return {"error": "No CUDA device available"}

        # 3. Ensure models are downloaded
        model_result = ensure_models(workflow)
        if not model_result.models_ready:
            return {"error": f"Model setup failed: {model_result.error}"}
        log_lines.append(
            f"Models ready in {model_result.download_time_s:.1f}s"
        )

        # 4. Start ComfyUI
        if not start_comfyui():
            return {"error": "Failed to start ComfyUI"}

        # 5. Save input images
        if images:
            saved_images = save_input_images(images)
            workflow_str = json.dumps(workflow)
            for orig_name, saved_name in saved_images.items():
                workflow_str = workflow_str.replace(orig_name, saved_name)
            workflow = json.loads(workflow_str)
            log_lines.append(f"Saved {len(saved_images)} input image(s)")

        # 6. Download LoRAs
        if lora_downloads:
            if not download_loras(lora_downloads):
                return {"error": "Failed to download required LoRAs"}
            log_lines.append(f"Downloaded {len(lora_downloads)} LoRA(s)")

        # 7. Queue workflow
        prompt_id = queue_workflow(workflow)
        if not prompt_id:
            return {"error": "Failed to queue workflow"}

        # 8. Wait for completion
        if not wait_for_completion(prompt_id):
            return {"error": "Workflow execution failed or timed out"}

        # 9. Collect outputs
        outputs = collect_outputs(prompt_id)
        if not outputs:
            return {"error": "No outputs produced"}

        elapsed = time.time() - job_start
        log_lines.append(f"Completed in {elapsed:.1f}s")
        logger.info(
            f"✅ Job {job_id} complete: {len(outputs)} outputs in {elapsed:.1f}s"
        )

        return {
            "files": outputs,
            "job_time_s": round(elapsed, 1),
            "model_time_s": round(model_result.download_time_s, 1),
            "log": log_lines,
        }

    except Exception as e:
        logger.exception(f"❌ Job {job_id} failed: {e}")
        return {"error": str(e)}


# ---- Entrypoint ----
if __name__ == "__main__":
    logger.info("🎨 I2I RunPod Worker starting...")
    logger.info(f"   COMFYUI_PATH: {COMFYUI_PATH}")
    logger.info(f"   MODEL_VOLUME: {MODEL_VOLUME}")

    runpod.serverless.start({"handler": handler})
