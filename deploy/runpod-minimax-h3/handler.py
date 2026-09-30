"""
RunPod Serverless Handler — MiniMax-H3 Worker
=============================================
Downloads MiniMax-H3 models from HuggingFace Hub, starts ComfyUI,
and processes video+audio generation workflows (t2v / i2v).

Target: 80 GB+ VRAM GPUs (A100, H100, B200, GB200); the int8/nvfp4
quantized files also fit on 48 GB tiers for short generations.

Models (Comfy-Org/MiniMax-H3 repack, all startup-required):
  - minimax_h3_fl2va_pruned_int8_convrot.safetensors  (20.97 GB, diffusion, t2v + i2v keyframes)
  - qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors      (15.69 GB, text encoder, no Blackwell needed)
  - minimax_h3_video_vae_fp16.safetensors             ( 5.21 GB, video VAE)
  - minimax_h3_audio_vae_fp32.safetensors             ( 0.61 GB, audio VAE — H3 generates audio too)

The workflow is provided by the Oelala backend (ComfyUIClient
build_cloud_minimax_h3_{t2v,i2v}_workflow) and uses ComfyUI core nodes
(MiniMaxH3ImageToVideo, BasicGuider, VAEDecodeAudio) + VHS_VideoCombine
for the final mp4 with muxed audio.
"""

import base64
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
logger = logging.getLogger("minimax-h3-handler")

# ---- Constants ----
COMFYUI_PORT = 8188
COMFYUI_URL = f"http://127.0.0.1:{COMFYUI_PORT}"
COMFYUI_PATH = os.environ.get("COMFYUI_PATH", "/comfyui")
MODEL_VOLUME = os.environ.get("MODEL_VOLUME", "/runpod-volume")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Maximum time to wait for ComfyUI startup
COMFYUI_STARTUP_TIMEOUT = 300  # 5 minutes (model loading is slow)

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

MINIMAX_H3_MODELS = [
    {
        "filename": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "repo": "Comfy-Org/MiniMax-H3",
        "hf_path": "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "target_dir": "diffusion_models",
        "size_gb": 20.97,
        "description": "MiniMax-H3 FL2VA diffusion model (int8+convrot, t2v + i2v keyframes)",
        "startup_required": False,
    },
    {
        "filename": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "repo": "Comfy-Org/MiniMax-H3",
        "hf_path": "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        "target_dir": "text_encoders",
        "size_gb": 15.69,
        "description": "MiniMax-H3 Qwen3-VL-32B text encoder (nvfp4_awq, no Blackwell needed)",
        "startup_required": True,
    },
    {
        "filename": "minimax_h3_video_vae_fp16.safetensors",
        "repo": "Comfy-Org/MiniMax-H3",
        "hf_path": "vae/minimax_h3_video_vae_fp16.safetensors",
        "target_dir": "vae",
        "size_gb": 5.21,
        "description": "MiniMax-H3 video VAE (fp16)",
        "startup_required": True,
    },
    {
        "filename": "minimax_h3_audio_vae_fp32.safetensors",
        "repo": "Comfy-Org/MiniMax-H3",
        "hf_path": "vae/minimax_h3_audio_vae_fp32.safetensors",
        "target_dir": "vae",
        "size_gb": 0.61,
        "description": "MiniMax-H3 audio VAE (fp32)",
        "startup_required": True,
    },
    {
        "filename": "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        "repo": "Comfy-Org/MiniMax-H3",
        "hf_path": "loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        "target_dir": "loras",
        "size_gb": 1.96,
        "description": "MiniMax-H3 FL2VA turbo LoRA 4-step 768p (draft quality mode)",
        "startup_required": False,
    },
    {
        "filename": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "repo": "Comfy-Org/MiniMax-H3",
        "hf_path": "loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "target_dir": "loras",
        "size_gb": 1.96,
        "description": "MiniMax-H3 FL2VA turbo LoRA 8-step (standard quality mode)",
        "startup_required": False,
    },
    {
        "filename": "h3ErosMax_beta5_3185144.safetensors",
        "repo": "bomehika/oelala-models",
        "hf_path": "diffusion_models/h3ErosMax_beta5_3185144.safetensors",
        "url": "https://civitai.com/api/download/models/3294059",
        "token_env": "CIVITAI_TOKEN",
        "target_dir": "diffusion_models",
        "size_gb": 13.67,
        "description": "H3 Eros Max beta5 TURBO-hybrid (NSFW finetune, turbo fused, model_variant='eros'; HF mirror primary, Civitai fallback)",
        "startup_required": False,
    },
    {
        "filename": "h3ErosMax_beta5_3185154.safetensors",
        "repo": "bomehika/oelala-models",
        "hf_path": "diffusion_models/h3ErosMax_beta5_3185154.safetensors",
        "url": "https://civitai.com/api/download/models/3294059?fileId=3185154",
        "token_env": "CIVITAI_TOKEN",
        "target_dir": "diffusion_models",
        "size_gb": 20.97,
        "description": "H3 Eros Max beta5 non-turbo int8 (all 200 quantized layers 8-bit; model_variant='eros_int8')",
        "startup_required": False,
    },
    {
        "filename": "h3ErosMax_beta5_3178732.safetensors",
        "repo": "bomehika/oelala-models",
        "hf_path": "diffusion_models/h3ErosMax_beta5_3178732.safetensors",
        "url": "https://civitai.com/api/download/models/3294059?fileId=3178732",
        "token_env": "CIVITAI_TOKEN",
        "target_dir": "diffusion_models",
        "size_gb": 20.97,
        "description": "H3 Eros Max beta5 turbo int8 (model_variant='eros_int8_turbo')",
        "startup_required": False,
    },
    {
        "filename": "DasiwaMinimaxH3_dasiwaHybridTurboV2_3203135.safetensors",
        "repo": "bomehika/oelala-models",
        "hf_path": "diffusion_models/DasiwaMinimaxH3_dasiwaHybridTurboV2_3203135.safetensors",
        "url": "https://civitai.com/api/download/models/3314686?fileId=3203135",
        "token_env": "CIVITAI_TOKEN",
        "target_dir": "diffusion_models",
        "size_gb": 20.97,
        "description": "DaSiWa Hybrid Turbo v2 int8 (4-8 step distillation; model_variant='dasiwa_turbo')",
        "startup_required": False,
    },
    {
        "filename": "DasiwaMinimaxH3_dasiwaHybridV2_3203130.safetensors",
        "repo": "bomehika/oelala-models",
        "hf_path": "diffusion_models/DasiwaMinimaxH3_dasiwaHybridV2_3203130.safetensors",
        "url": "https://civitai.com/api/download/models/3314675?fileId=3203130",
        "token_env": "CIVITAI_TOKEN",
        "target_dir": "diffusion_models",
        "size_gb": 20.97,
        "description": "DaSiWa Hybrid v2 int8 non-turbo (full step count; model_variant='dasiwa')",
        "startup_required": False,
    },
]

PUBLIC_MODEL_FILENAMES = {model["filename"] for model in MINIMAX_H3_MODELS}

# Directories where ComfyUI looks for models
NODE_MANAGED_MODEL_DIRS = [
    "diffusion_models",
    "checkpoints",
    "loras",
    "text_encoders",
    "vae",
]

# Workflow input keys that might reference model filenames
WORKFLOW_MODEL_INPUT_KEYS = (
    "unet_name",
    "clip_name",
    "vae_name",
    "audio_vae_name",
)


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


def _download_from_url(model: Dict[str, Any], target: Path) -> bool:
    """Stream a model file from a direct URL (e.g. Civitai). Token never logged."""
    filename = model["filename"]
    target_dir = model["target_dir"]
    size_gb = model["size_gb"]
    url = model["url"]
    token_env = model.get("token_env", "")
    token = os.getenv(token_env, "") if token_env else ""
    if token:
        url = f"{url}{'&' if '?' in url else '?'}token={token}"
    logger.info(f"⬇️  Downloading {filename} ({size_gb:.1f} GB) from URL source...")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".download")
        resp = requests.get(url, stream=True, timeout=600)
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                f.write(chunk)
        if tmp.stat().st_size < 1_000_000:
            logger.error(
                f"❌ Download too small ({tmp.stat().st_size} bytes), likely an auth page"
            )
            tmp.unlink()
            return False
        tmp.rename(target)
        logger.info(f"✅ Downloaded: {target_dir}/{filename}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to download {filename}: {e}")
        return False


def download_model(model: Dict[str, Any]) -> bool:
    """Download a single model file from HuggingFace Hub (or a direct URL)."""
    filename = model["filename"]
    repo = model.get("repo", "")
    hf_path = model.get("hf_path", "")
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

    # Direct-URL source when the entry has no HF repo (e.g. Civitai-only
    # entries). Entries with both repo and url download from HF first and fall
    # back to the URL in the except handler below.
    if model.get("url", "") and not repo:
        return _download_from_url(model, target)

    # Download from HuggingFace
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
        logger.error(f"❌ Failed to download {filename} from HF: {e}")
        if model.get("url", ""):
            logger.info(f"↩️ Falling back to URL source for {filename}...")
            return _download_from_url(model, target)
        return False


def ensure_models() -> WorkflowModelPrepResult:
    """Download the always-needed core models and return preparation result.

    Checkpoints and turbo LoRAs are NOT downloaded here: they are fetched on
    demand by ensure_workflow_models() for the files a job actually uses, so a
    job that runs one variant never pays the download time or disk for another.
    """
    result = WorkflowModelPrepResult()
    start = time.time()

    ensure_model_directories()
    setup_model_links()

    for model in MINIMAX_H3_MODELS:
        if not model.get("startup_required", False):
            continue
        if download_model(model):
            result.models_downloaded.append(model["filename"])
        else:
            result.error = f"Required model missing: {model['filename']}"
            result.download_time_s = time.time() - start
            return result

    result.models_ready = True
    result.download_time_s = time.time() - start
    logger.info(
        f"✅ Core models ready in {result.download_time_s:.1f}s "
        f"({len(result.models_downloaded)} files)"
    )
    return result


# Loader-node input names that reference a model file, mapped to the ComfyUI
# models subdirectory they resolve against.
_WORKFLOW_MODEL_INPUTS = {
    "unet_name": "diffusion_models",
    "lora_name": "loras",
    "clip_name": "text_encoders",
    "vae_name": "vae",
    "ckpt_name": "checkpoints",
}


def _workflow_model_names(workflow: Dict[str, Any]) -> Dict[str, str]:
    """Map every model filename referenced by a loader node to its models subdir."""
    found: Dict[str, str] = {}
    for node in (workflow or {}).values():
        if not isinstance(node, dict):
            continue
        for key, value in (node.get("inputs") or {}).items():
            if key in _WORKFLOW_MODEL_INPUTS and isinstance(value, str) and value:
                found[value] = _WORKFLOW_MODEL_INPUTS[key]
    return found


def ensure_workflow_models(workflow: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Download registry models referenced by this workflow; report what is missing.

    Returns (downloaded, missing) filenames. Unregistered names (user LoRAs,
    files baked into the image) are ignored here — job LoRAs are handled by
    download_loras().
    """
    registry = {m["filename"]: m for m in MINIMAX_H3_MODELS}
    downloaded: List[str] = []
    missing: List[str] = []
    for filename, subdir in _workflow_model_names(workflow).items():
        model = registry.get(filename)
        if not model:
            continue
        target = Path(COMFYUI_PATH) / "models" / subdir / filename
        if target.exists() and target.stat().st_size > 1_000_000:
            continue
        logger.info(f"📥 Workflow needs {filename} — fetching on demand...")
        if download_model(model):
            downloaded.append(filename)
        else:
            missing.append(filename)
    if downloaded:
        logger.info(f"✅ On-demand models: {', '.join(downloaded)}")
    if missing:
        logger.error(f"❌ Workflow models unavailable: {', '.join(missing)}")
    return downloaded, missing


# ---- ComfyUI Process Management ----

_comfyui_process: Optional[subprocess.Popen] = None


def start_comfyui() -> bool:
    """Start ComfyUI as a subprocess and wait for it to be ready."""
    global _comfyui_process

    if _comfyui_process and _comfyui_process.poll() is None:
        # Already running — check if reachable
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
        # Force disk-backed weight streaming: the auto-detection only enables
        # fast_disk when the model file's backing device is a real NVMe
        # (>= Gen4 x4) AND every file agrees — on RunPod hosts where the
        # container disk is not exposed as NVMe this would silently disable
        # streaming, pinning 42.5 GB of weights in host RAM (50 GB on
        # A6000/A40 hosts) and slowing large jobs to a crawl.
        "--fast-disk",
    ]
    _comfyui_process = subprocess.Popen(
        cmd,
        cwd=COMFYUI_PATH,
        stdout=sys.stdout,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Wait for ComfyUI to become responsive
    deadline = time.time() + COMFYUI_STARTUP_TIMEOUT
    while time.time() < deadline:
        if _comfyui_process.poll() is not None:
            # Process exited — read output
            output = "Check RunPod logs for details."
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


def get_cuda_device() -> Optional[str]:
    """Return the CUDA device name for per-job cost attribution in job output."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None


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
            # Strip data URI prefix if present
            if "," in b64data:
                b64data = b64data.split(",", 1)[1]
            # Strip whitespace/newlines and fix padding (avoid "Incorrect padding")
            b64data = "".join(b64data.split())
            missing = (-len(b64data)) % 4
            if missing:
                b64data = b64data + ("=" * missing)

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
    """Download LoRA files from backend URL for cloud jobs.

    Expects list of {"filename": "...", "url": "https://..."} dicts,
    matching the format sent by _build_lora_download_list() in the backend.
    """
    if not lora_downloads:
        return True

    lora_dir = Path(COMFYUI_PATH) / "models" / "loras"
    lora_dir.mkdir(parents=True, exist_ok=True)

    for lora in lora_downloads:
        filename = lora.get("filename", "")
        # Primary source + optional fallback (e.g. HF mirror first, then the
        # signed self-hosted backend URL).
        candidate_urls = [
            u for u in (lora.get("url", ""), lora.get("fallback_url", "")) if u
        ]

        if not filename or not candidate_urls:
            logger.warning(f"⚠️ Skipping LoRA entry with missing filename/url: {lora}")
            continue

        target = lora_dir / filename
        if target.exists():
            logger.info(f"✅ LoRA already present: {filename}")
            continue

        logger.info(f"⬇️  Downloading LoRA: {filename}...")
        downloaded = False
        last_error = None
        for url in candidate_urls:
            tmp = target.with_suffix(".download")
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                headers = {}
                hf_token = lora.get("hf_token", "")
                if hf_token and "huggingface.co" in url:
                    # Only send the HF token to huggingface.co hosts
                    headers["Authorization"] = f"Bearer {hf_token}"
                resp = requests.get(url, stream=True, timeout=600, headers=headers)
                resp.raise_for_status()

                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(chunk)
                tmp.rename(target)

                size_mb = target.stat().st_size / (1024 * 1024)
                logger.info(f"✅ LoRA downloaded: {filename} ({size_mb:.0f}MB)")
                downloaded = True
                break
            except Exception as e:
                last_error = e
                if tmp.exists():
                    tmp.unlink()
                if len(candidate_urls) > 1:
                    logger.warning(f"⚠️ LoRA source failed ({url}), trying next: {e}")
        if not downloaded:
            logger.error(f"❌ Failed to download LoRA {filename}: {last_error}")
            return False

    return True


def queue_workflow(workflow: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """Queue a workflow on ComfyUI; return (prompt_id, validation detail)."""
    client_id = uuid.uuid4().hex
    payload = {"prompt": workflow, "client_id": client_id}

    try:
        resp = requests.post(f"{COMFYUI_URL}/prompt", json=payload, timeout=120)
        if resp.status_code == 200:
            prompt_id = resp.json().get("prompt_id")
            logger.info(f"📋 Workflow queued: {prompt_id}")
            return prompt_id, ""
        detail = _summarize_comfy_error(resp)
        logger.error(f"❌ Queue failed ({resp.status_code}): {detail}")
        return None, f"HTTP {resp.status_code} {detail}"
    except Exception as e:
        logger.error(f"❌ Queue request failed: {e}")
        return None, str(e)


def _summarize_comfy_error(resp: Any) -> str:
    """Condense ComfyUI's validation payload into a short, loggable string."""
    try:
        body = resp.json()
    except Exception:
        return resp.text[:300]
    parts = []
    for node_id, err in (body.get("node_errors") or {}).items():
        for item in err.get("errors", []) or []:
            detail = item.get("details") or item.get("message") or ""
            parts.append(f"node {node_id} ({err.get('class_type')}): {detail}")
    if not parts:
        parts.append(json.dumps(body)[:300])
    return " | ".join(parts)[:600]


def verify_workflow_models(workflow: Dict[str, Any]) -> List[str]:
    """Return the workflow model files that are not present on this worker."""
    missing = []
    for filename, subdir in _workflow_model_names(workflow).items():
        target = Path(COMFYUI_PATH) / "models" / subdir / filename
        if not target.exists():
            missing.append(f"{subdir}/{filename}")
    return missing


def wait_for_completion(prompt_id: str, timeout: int = 2400) -> bool:
    """
    Wait for a prompt to complete. Polls /history/{prompt_id}.
    Timeout: 40 minutes default (MiniMax-H3 20-step 768p generation is slow).
    """
    deadline = time.time() + timeout
    poll_interval = 3.0
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

        # Adaptive polling — slower after first minute
        elapsed = timeout - (deadline - time.time())
        if elapsed > 60:
            poll_interval = min(poll_interval + 0.5, 10.0)

        time.sleep(poll_interval)

    logger.error(f"❌ Workflow timed out after {timeout}s")
    return False


def collect_outputs(prompt_id: str) -> List[Dict[str, Any]]:
    """Collect output files from a completed workflow."""
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
            # Check for video files (VHS_VideoCombine)
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
                    logger.info(
                        f"📦 Collected output: {filename} "
                        f"({len(file_bytes) / 1024 / 1024:.1f} MB)"
                    )

            # Check for image files
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

    except Exception as e:
        logger.error(f"❌ Failed to collect outputs: {e}")

    return outputs


# ---- Main Handler ----

def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    RunPod handler for MiniMax-H3 video generation.

    Expected input:
    {
        "workflow": { ... ComfyUI API workflow ... },
        "images": { "name": "base64data", ... },  # optional (i2v first frame)
        "lora_downloads": [ {"filename": "...", "url": "https://..."} ],  # optional
    }
    """
    job_id = event.get("id", "unknown")
    job_input = event.get("input", {})
    log_lines = []

    def _fail(message: str) -> Dict[str, Any]:
        """Return an error payload carrying the worker log tail for debugging."""
        logger.error(f"❌ Job failed: {message}")
        return {"ok": False, "error_message": message, "log": log_lines[-30:]}

    logger.info(f"🎬 Job started: {job_id}")
    job_start = time.time()

    try:
        # 1. Extract inputs
        workflow = job_input.get("workflow")
        if not workflow:
            return _fail("No workflow provided")

        images = job_input.get("images", {})
        lora_downloads = job_input.get("lora_downloads", [])

        # 2. Check CUDA
        if not wait_for_cuda():
            return _fail("No CUDA device available")
        gpu_name = get_cuda_device()
        log_lines.append(f"GPU: {gpu_name}")

        # 3. Ensure models are downloaded
        model_result = ensure_models()
        if not model_result.models_ready:
            return _fail(f"Model setup failed: {model_result.error}")
        log_lines.append(
            f"Models ready in {model_result.download_time_s:.1f}s"
        )

        # 4. Start ComfyUI
        if not start_comfyui():
            return _fail("Failed to start ComfyUI")

        # 5. Save input images
        if images:
            saved_images = save_input_images(images)
            # Replace image references in workflow
            workflow_str = json.dumps(workflow)
            for orig_name, saved_name in saved_images.items():
                workflow_str = workflow_str.replace(orig_name, saved_name)
            workflow = json.loads(workflow_str)
            log_lines.append(f"Saved {len(saved_images)} input image(s)")

        # 6. Download LoRAs
        if lora_downloads:
            if not download_loras(lora_downloads):
                return _fail("Failed to download required LoRAs")
            log_lines.append(f"Downloaded {len(lora_downloads)} LoRA(s)")

        # 7. Fetch any checkpoint/turbo-LoRA this workflow references but the
        #    worker does not have yet (variants are downloaded on demand).
        fetched, missing_models = ensure_workflow_models(workflow)
        if fetched:
            log_lines.append(f"On-demand models: {', '.join(fetched)}")
        if missing_models:
            return _fail(f"Workflow models unavailable: {', '.join(missing_models)}")

        # 8. Confirm every referenced model file is actually on disk
        absent = verify_workflow_models(workflow)
        log_lines.append(
            f"Models on disk: {len(_workflow_model_names(workflow)) - len(absent)}"
            f"/{len(_workflow_model_names(workflow))}"
        )
        if absent:
            return _fail(f"Models missing on worker: {', '.join(absent)}")

        # 9. Queue workflow
        prompt_id, queue_detail = queue_workflow(workflow)
        if not prompt_id:
            return _fail(f"Failed to queue workflow — {queue_detail}")

        # 10. Wait for completion
        if not wait_for_completion(prompt_id, timeout=2400):
            return _fail("Workflow execution failed or timed out")

        # 9. Collect outputs
        outputs = collect_outputs(prompt_id)
        if not outputs:
            return _fail("No outputs produced")

        elapsed = time.time() - job_start
        log_lines.append(f"Completed in {elapsed:.1f}s")
        logger.info(
            f"✅ Job {job_id} complete: {len(outputs)} outputs in {elapsed:.1f}s"
        )

        return {
            "files": outputs,
            "job_time_s": round(elapsed, 1),
            "model_time_s": round(model_result.download_time_s, 1),
            "gpu": gpu_name,
            "log": log_lines,
        }

    except Exception as e:
        logger.exception(f"❌ Job {job_id} failed: {e}")
        return _fail(str(e))


# ---- Entrypoint ----
if __name__ == "__main__":
    logger.info("🚀 MiniMax-H3 RunPod Worker starting...")
    logger.info(f"   COMFYUI_PATH: {COMFYUI_PATH}")
    logger.info(f"   MODEL_VOLUME: {MODEL_VOLUME}")

    runpod.serverless.start({"handler": handler})
