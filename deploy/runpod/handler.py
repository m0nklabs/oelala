#!/usr/bin/env python3
"""
RunPod Serverless Handler for Oelala ComfyUI Worker

Receives ComfyUI workflow JSON via RunPod API, executes it on the
local ComfyUI instance, and returns the output (images/videos).

Supports three asset-loading strategies:
1. RunPod cached models / HF cache for public models when available
2. Container-disk downloads from HuggingFace for missing public models
3. Optional RunPod Network Volume for LoRAs and hard-to-replace private assets

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
import shutil
from dataclasses import dataclass
from collections import deque
from pathlib import Path
from typing import Iterator

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
CACHED_MODEL_DIRS_ENV = "RUNPOD_CACHED_MODEL_DIRS"
HF_STAGING_DIR = Path(os.getenv("RUNPOD_HF_STAGING_DIR", "/tmp/hf_cache"))
DOWNLOAD_SAFETY_BUFFER_GB = float(os.getenv("RUNPOD_DOWNLOAD_SAFETY_BUFFER_GB", "2"))
DEFAULT_CACHED_MODEL_DIRS = [
    "/runpod-volume/huggingface-cache/hub",
    "/runpod-model-cache",
    "/runpod-models",
    "/cache/runpod-models",
    "/root/.cache/huggingface/hub",
    "/root/.cache/huggingface",
]

_cached_roots_logged = False


@dataclass(frozen=True)
class WorkflowModelPrepResult:
    requested_count: int = 0
    linked_count: int = 0
    downloaded_count: int = 0

    @property
    def prepared_count(self) -> int:
        return self.linked_count + self.downloaded_count

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
        "persist_on_volume": True,
        "startup_required": True,
    },
    {
        "hf_repo": HF_REPO_22,
        "hf_path": "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        "local_dir": "unet",
        "filename": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        "size_gb": 14.3,
        "description": "Wan 2.2 I2V low noise 14B fp8_scaled",
        "persist_on_volume": True,
        "startup_required": True,
    },
    {
        "hf_repo": HF_REPO_22,
        "hf_path": "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        "local_dir": "unet",
        "filename": "wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        "size_gb": 14.3,
        "description": "Wan 2.2 T2V high noise 14B fp8_scaled",
        "persist_on_volume": True,
        "startup_required": True,
    },
    {
        "hf_repo": HF_REPO_22,
        "hf_path": "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        "local_dir": "unet",
        "filename": "wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        "size_gb": 14.3,
        "description": "Wan 2.2 T2V low noise 14B fp8_scaled",
        "persist_on_volume": True,
        "startup_required": True,
    },
    {
        "hf_repo": HF_REPO_22,
        "hf_path": "split_files/text_encoders/umt5_xxl_fp16.safetensors",
        "local_dir": "clip",
        "filename": "umt5_xxl_fp16.safetensors",
        "size_gb": 11.4,
        "description": "UMT5-XXL fp16 text encoder",
        "persist_on_volume": True,
        "startup_required": True,
    },
    {
        "hf_repo": HF_REPO_21,
        "hf_path": "split_files/vae/wan_2.1_vae.safetensors",
        "local_dir": "vae",
        "filename": "wan_2.1_vae.safetensors",
        "size_gb": 0.40,
        "description": "Wan 2.1 VAE — required for 14B models (wan2.2_vae is 5B only)",
        "startup_required": True,
    },
    {
        "hf_repo": HF_REPO_21,
        "hf_path": "split_files/clip_vision/clip_vision_h.safetensors",
        "local_dir": "clip_vision",
        "filename": "clip_vision_h.safetensors",
        "size_gb": 1.26,
        "description": "CLIP Vision H (I2V conditioning)",
        "startup_required": True,
    },
]
PUBLIC_MODEL_FILENAMES = {model["filename"] for model in CLOUD_MAX_MODELS}
NODE_MANAGED_MODEL_DIRS = [
    "audio_vae",
    "checkpoints",
    "clip",
    "clip_vision",
    "diffusion_models",
    "latent_upscale_models",
    "loras",
    "text_encoders",
    "unet",
    "upscale_models",
    "vae",
]
NODE_MANAGED_MODEL_FILENAMES = {
    "LTX2_video_vae_bf16.safetensors",
    "gemma_3_12B_it_nvfp4.safetensors",
    "ltx-2-19b-dev-Q4_K_M.gguf",
    "ltx-2-19b-distilled-fp8.safetensors",
    "ltx-2-19b-distilled_Q4_K_M.gguf",
    "ltx-2-19b-embeddings_connector_bf16.safetensors",
}
WORKFLOW_MODEL_INPUT_KEYS = (
    "clip_name",
    "clip_name1",
    "clip_name2",
    "gemma_path",
    "ltxv_path",
    "unet_name",
    "vae_name",
)


# ---- Model Setup ----

def setup_model_links():
    """
    Create symlinks from network volume assets to ComfyUI model dirs.

    Public/general Hugging Face model files are never linked from the RunPod
    Network Volume. That volume is reserved for LoRAs and hard-to-recreate
    private/custom assets only.
    """
    volume_models = Path(MODEL_VOLUME) / "models"
    comfyui_models = Path("/comfyui/models")

    if not volume_models.exists():
        logger.info(f"ℹ️ No network volume at {volume_models}")
        return False

    model_dirs = NODE_MANAGED_MODEL_DIRS

    linked = 0
    for d in model_dirs:
        src = volume_models / d
        dst = comfyui_models / d
        if src.exists():
            dst.mkdir(parents=True, exist_ok=True)
            for f in src.iterdir():
                if f.is_file() and f.name in PUBLIC_MODEL_FILENAMES:
                    logger.info(f"⏭️ Skipping public model on RunPod volume per policy: {d}/{f.name}")
                    continue
                target = dst / f.name
                if not target.exists():
                    target.symlink_to(f)
                    logger.info(f"🔗 Linked: {d}/{f.name}")
                    linked += 1

    logger.info(f"✅ Model symlinks: {linked} files linked")
    return linked > 0


def ensure_model_directories() -> None:
    """Create the ComfyUI model folders used by the shared Wan + LTX worker."""
    comfyui_models = Path("/comfyui/models")
    for model_dir in NODE_MANAGED_MODEL_DIRS:
        (comfyui_models / model_dir).mkdir(parents=True, exist_ok=True)


def _is_startup_required(model: dict) -> bool:
    """Return whether a model should be prepared during worker startup."""
    return bool(model.get("startup_required", model.get("required", True)))


def _startup_models() -> list[dict]:
    """Return models that should be prepared during worker startup."""
    return [model for model in CLOUD_MAX_MODELS if _is_startup_required(model)]


def _deferred_models() -> list[dict]:
    """Return models that are intentionally deferred until workflow demand."""
    return [model for model in CLOUD_MAX_MODELS if not _is_startup_required(model)]


def download_models():
    """
    Download startup-required Cloud Max models from HuggingFace if not already present.
    Uses huggingface_hub for efficient downloading with resume support.

    Public/general models always download to the worker container disk.
    The RunPod Network Volume is reserved for LoRAs and private/custom assets.
    """
    linked = link_cached_models(required_only=True)
    if linked:
        logger.info(f"✅ Startup satisfied {linked} required model(s) from cache")

    total_to_download = 0
    models_needed = []

    # Check which models are missing (check both volume and comfyui dirs)
    for model in CLOUD_MAX_MODELS:
        # Skip mode-specific models at startup; they are resolved per workflow.
        if not _is_startup_required(model):
            logger.info(
                f"⏭️ {model['filename']} ({model['size_gb']}GB) — not startup-required, deferring to workflow"
            )
            continue

        if _is_model_present(model):
            logger.info(f"✅ {model['filename']} ({model['size_gb']}GB) — already present")
        else:
            models_needed.append(model)
            total_to_download += model["size_gb"]

    if not models_needed:
        logger.info("✅ All Cloud Max models already downloaded")
        return True

    ok, capacity_error = _check_download_capacity(models_needed)
    if not ok:
        logger.error(f"❌ {capacity_error}")
        return False

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.error("❌ huggingface_hub not installed, cannot download models")
        return False

    logger.info(f"📦 Downloading {len(models_needed)} models ({total_to_download:.1f}GB total)...")
    start = time.time()

    for i, model in enumerate(models_needed, 1):
        target_base = _target_base_for_model(model)
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
                local_dir=str(HF_STAGING_DIR),
                local_dir_use_symlinks=False,
            )
            shutil.move(downloaded_path, str(dest))
            elapsed = time.time() - dl_start
            speed = model["size_gb"] / elapsed * 1024 if elapsed > 0 else 0
            logger.info(f"✅ {model['filename']} downloaded in {elapsed:.0f}s "
                       f"({speed:.0f} MB/s)")

            # Clean HF cache after each download to free disk space
            shutil.rmtree(HF_STAGING_DIR, ignore_errors=True)
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


def _should_persist_model(model: dict) -> bool:
    """Public/general models must never persist on the RunPod Network Volume."""
    return False


def _writable_volume_models_dir() -> Path | None:
    """Return the writable volume-backed models dir, if available.

    RunPod cached-model mounts also live under /runpod-volume, but they do not
    guarantee a writable Network Volume for arbitrary files. Treat the path as a
    writable model volume only when the root exists and is writable.
    """
    volume_path = Path(MODEL_VOLUME)
    if not volume_path.exists():
        return None

    probe_path = volume_path / ".oelala-write-probe"
    try:
        probe_path.write_text("probe")
        probe_path.unlink(missing_ok=True)
    except OSError:
        logger.info(f"📁 Volume root present but not writable at {volume_path}; treating it as cache-only mount")
        return None

    return volume_path / "models"


def _target_base_for_model(model: dict) -> Path:
    """Always store public/general models on the worker container disk."""
    comfyui_models = Path("/comfyui/models")

    logger.info(f"📁 Using ephemeral container storage for model: {model['filename']}")
    return comfyui_models


def _model_size_bytes(model: dict) -> int:
    """Convert a configured model size from GiB to bytes."""
    return int(model["size_gb"] * (1024 ** 3))


def _existing_stats_path(path: Path) -> Path:
    """Return an existing path suitable for filesystem stats."""
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _storage_device_key(path: Path) -> int:
    """Return the filesystem device id for the given path."""
    return os.stat(_existing_stats_path(path)).st_dev


def _check_download_capacity(models: list[dict]) -> tuple[bool, str | None]:
    """Estimate whether current filesystems can hold the requested HF downloads.

    Final model files accumulate on the target filesystem, while the Hugging Face
    staging dir needs room for the largest in-flight download plus a small buffer.
    If both paths live on the same filesystem, those requirements add up.
    """
    if not models:
        return True, None

    requirements: dict[int, dict[str, int | Path]] = {}
    buffer_bytes = int(DOWNLOAD_SAFETY_BUFFER_GB * (1024 ** 3))

    for model in models:
        target_dir = _target_base_for_model(model) / model["local_dir"]
        device = _storage_device_key(target_dir)
        entry = requirements.setdefault(
            device,
            {"path": _existing_stats_path(target_dir), "required": 0},
        )
        entry["required"] += _model_size_bytes(model)

    staging_device = _storage_device_key(HF_STAGING_DIR)
    staging_entry = requirements.setdefault(
        staging_device,
        {"path": _existing_stats_path(HF_STAGING_DIR), "required": 0},
    )
    staging_entry["required"] += max(_model_size_bytes(model) for model in models) + buffer_bytes

    failures: list[str] = []
    for entry in requirements.values():
        stats_path = entry["path"]
        required_bytes = int(entry["required"])
        free_bytes = shutil.disk_usage(stats_path).free
        if free_bytes < required_bytes:
            failures.append(
                f"{stats_path}: need ~{required_bytes / (1024 ** 3):.1f}GB free, have {free_bytes / (1024 ** 3):.1f}GB"
            )

    if not failures:
        return True, None

    target_names = ", ".join(model["filename"] for model in models)
    return (
        False,
        "Insufficient disk for Hugging Face model download preflight. "
        f"Models: {target_names}. "
        f"Constraints: {'; '.join(failures)}. "
        "Increase containerDiskInGb or use RunPod cached models.",
    )


def _split_env_list(raw_value: str | None) -> list[str]:
    """Parse a comma-separated env var into a de-duplicated list."""
    if not raw_value:
        return []

    values: list[str] = []
    for part in raw_value.split(","):
        value = part.strip()
        if value and value not in values:
            values.append(value)
    return values


_cached_model_roots: list[Path] | None = None  # Cached result — env vars don't change mid-job


def _candidate_cached_model_roots() -> list[Path]:
    """Return ordered candidate roots for RunPod cached-model files.

    Results are cached after first call since env vars don't change during
    a container's lifetime. Avoids rebuilding the list on every model check.
    """
    global _cached_model_roots
    if _cached_model_roots is not None:
        return _cached_model_roots

    roots: list[Path] = []
    explicit_roots = _split_env_list(os.getenv(CACHED_MODEL_DIRS_ENV))
    env_roots = [
        os.getenv("RUNPOD_CACHED_MODEL_DIR"),
        os.getenv("HF_HOME"),
        os.getenv("HUGGINGFACE_HUB_CACHE"),
        os.getenv("HF_HUB_CACHE"),
    ]

    for raw_path in [*explicit_roots, *env_roots, *DEFAULT_CACHED_MODEL_DIRS]:
        if not raw_path:
            continue
        path = Path(raw_path)
        if path not in roots:
            roots.append(path)

    _cached_model_roots = roots
    return roots


def _log_cached_model_roots_once() -> None:
    """Emit one-time diagnostics about cached-model search roots."""
    global _cached_roots_logged
    if _cached_roots_logged:
        return

    roots = _candidate_cached_model_roots()
    if not roots:
        logger.warning("⚠️ No cached-model roots configured or discovered")
        _cached_roots_logged = True
        return

    for root in roots:
        try:
            if root.exists():
                logger.info(f"🗂️ Cached-model root available: {root}")
            else:
                logger.info(f"🗂️ Cached-model root missing: {root}")
        except OSError as exc:
            logger.info(f"🗂️ Cached-model root inaccessible: {root} ({exc})")

    _cached_roots_logged = True


def _iter_cached_model_candidates(root: Path, model: dict) -> Iterator[Path]:
    """Yield likely cached-model file paths for a single model."""
    filename = model["filename"]
    hf_path = Path(model["hf_path"])
    repo_slug = model.get("hf_repo", HF_REPO_22).replace("/", "--")
    snapshot_dirs = [
        root / f"models--{repo_slug}" / "snapshots",
        root / repo_slug / "snapshots",
        root / "snapshots",
    ]
    direct_candidates = [
        root / hf_path,
        root / model["local_dir"] / filename,
        root / "models" / model["local_dir"] / filename,
        root / filename,
    ]

    seen: set[Path] = set()
    for candidate in direct_candidates:
        if candidate not in seen:
            seen.add(candidate)
            yield candidate

    for snapshots_root in snapshot_dirs:
        if not snapshots_root.exists():
            continue
        for snapshot_dir in snapshots_root.iterdir():
            candidate = snapshot_dir / hf_path
            if candidate not in seen:
                seen.add(candidate)
                yield candidate

    if root.exists() and root.is_dir():
        for candidate in root.rglob(filename):
            if candidate not in seen:
                seen.add(candidate)
                yield candidate


def _find_cached_model_source(model: dict, *, emit_logs: bool = True) -> Path | None:
    """Locate a model file inside a RunPod cached-model mount or HF cache."""
    _log_cached_model_roots_once()
    for root in _candidate_cached_model_roots():
        try:
            root_exists = root.exists()
        except OSError as exc:
            if emit_logs:
                logger.info(f"🫥 Skipping inaccessible cached-model root {root}: {exc}")
            continue
        if not root_exists:
            continue
        for candidate in _iter_cached_model_candidates(root, model):
            if candidate.is_file():
                if emit_logs:
                    logger.info(f"💾 Found cached model for {model['filename']}: {candidate}")
                return candidate
    if emit_logs:
        logger.info(f"🫥 No cached model found for {model['filename']}")
    return None


def _model_state_for_log(model: dict) -> str:
    """Return a concise readiness state for diagnostics logging."""
    _, dest_local = _model_destinations(model)
    if dest_local.is_symlink():
        return "cached-linked"
    if _is_model_present(model):
        return "local"
    if _find_cached_model_source(model, emit_logs=False):
        return "cached-available"
    return "download-needed"


def _detect_workflow_family(referenced: set[str]) -> str:
    """Infer the Cloud Max workflow family from referenced model names."""
    has_ltx = any(name in NODE_MANAGED_MODEL_FILENAMES or name.startswith("ltx-") for name in referenced)
    has_i2v = any("i2v" in name for name in referenced) or "clip_vision_h.safetensors" in referenced
    has_t2v = any("t2v" in name for name in referenced)
    if has_ltx and (has_i2v or has_t2v):
        return "ltx-mixed"
    if has_ltx:
        return "ltx"
    if has_i2v and has_t2v:
        return "mixed"
    if has_i2v:
        return "i2v"
    if has_t2v:
        return "t2v"
    return "shared-core"


def _log_startup_model_plan() -> None:
    """Emit a clear startup/deferred model plan for worker diagnostics."""
    startup_states = ", ".join(
        f"{model['filename']}={_model_state_for_log(model)}"
        for model in _startup_models()
    )
    deferred_states = ", ".join(
        f"{model['filename']}={_model_state_for_log(model)}"
        for model in _deferred_models()
    )
    logger.info(f"🧭 Startup model plan: preload shared core only ({startup_states})")
    logger.info(f"🧭 Deferred workflow models: {deferred_states}")


def _link_model_into_comfyui(model: dict, source: Path) -> bool:
    """Symlink a cached model file into the expected ComfyUI model directory."""
    _, dest_local = _model_destinations(model)
    dest_local.parent.mkdir(parents=True, exist_ok=True)

    if _is_model_present(model):
        return False
    # Remove stale symlink or corrupt/empty placeholder before linking
    if dest_local.exists() or dest_local.is_symlink():
        dest_local.unlink()

    dest_local.symlink_to(source)
    logger.info(f"🔗 Linked cached model: {model['filename']} -> {source}")
    return True


def _missing_models(
    *,
    required_only: bool = False,
    filenames: set[str] | None = None,
) -> list[dict]:
    """Return models that are still missing from both volume and local paths."""
    missing = []
    for model in CLOUD_MAX_MODELS:
        if required_only and not _is_startup_required(model):
            continue
        if filenames and model["filename"] not in filenames:
            continue
        if not _is_model_present(model):
            missing.append(model)
    return missing


def link_cached_models(
    *,
    required_only: bool = False,
    filenames: set[str] | None = None,
) -> int:
    """Link models from cached storage before falling back to HF downloads."""
    linked = 0
    for model in _missing_models(required_only=required_only, filenames=filenames):
        source = _find_cached_model_source(model)
        if source and _link_model_into_comfyui(model, source):
            linked += 1

    if linked:
        logger.info(f"✅ Linked {linked} cached model(s) into ComfyUI")
    return linked


def _is_model_present(model: dict) -> bool:
    """Check whether a model already exists in either persistent or local storage.

    A model is considered present only if the file exists AND has a reasonable size
    (at least 50MB). This prevents treating corrupt/empty placeholder files as valid.
    """
    # Minimum file size: 50MB — catches empty placeholders and failed partial downloads
    min_bytes = 50 * 1024 * 1024

    def _valid(path: Path) -> bool:
        try:
            return path.exists() and path.stat().st_size >= min_bytes
        except OSError:
            return False

    dest_vol, dest_local = _model_destinations(model)
    volume_present = _valid(dest_vol) and _should_persist_model(model)
    return volume_present or _valid(dest_local)


def download_requested_models(filenames: list[str]) -> int:
    """Download a specific subset of models on demand."""
    requested = {name for name in filenames if name}
    if not requested:
        return 0

    linked = link_cached_models(filenames=requested)
    models = _missing_models(filenames=requested)
    if not models:
        return linked

    ok, capacity_error = _check_download_capacity(models)
    if not ok:
        raise RuntimeError(capacity_error)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError("huggingface_hub not installed, cannot download models")

    comfyui_models = Path("/comfyui/models")
    prepared = linked

    for model in models:
        if _is_model_present(model):
            logger.info(f"✅ {model['filename']} already present")
            continue

        target_base = _target_base_for_model(model)
        dest_dir = target_base / model["local_dir"]
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / model["filename"]

        # Remove any stale/corrupt placeholder before downloading
        if dest.exists() or dest.is_symlink():
            logger.info(f"🗑️ Removing corrupt/stale placeholder: {dest.name}")
            dest.unlink(missing_ok=True)

        logger.info(f"⬇️ On-demand model download: {model['filename']}")

        downloaded_path = hf_hub_download(
            repo_id=model.get("hf_repo", HF_REPO_22),
            filename=model["hf_path"],
            local_dir=str(HF_STAGING_DIR),
            local_dir_use_symlinks=False,
        )

        shutil.move(downloaded_path, str(dest))
        shutil.rmtree(HF_STAGING_DIR, ignore_errors=True)

        # If downloaded to volume, make sure ComfyUI sees it immediately.
        if target_base != comfyui_models:
            comfyui_target = comfyui_models / model["local_dir"] / model["filename"]
            comfyui_target.parent.mkdir(parents=True, exist_ok=True)
            if not comfyui_target.exists():
                comfyui_target.symlink_to(dest)
                logger.info(f"🔗 Symlinked on-demand model: {model['filename']}")

        prepared += 1

    return prepared


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


def ensure_workflow_models(workflow: dict, job=None) -> WorkflowModelPrepResult:
    """Ensure workflow-referenced optional models exist before queueing."""
    referenced = set()
    for node in workflow.values():
        inputs = node.get("inputs", {})
        for key in WORKFLOW_MODEL_INPUT_KEYS:
            value = inputs.get(key)
            if isinstance(value, str):
                referenced.add(value)

    if referenced:
        logger.info(
            "🎯 Workflow model check: family=%s referenced=%s",
            _detect_workflow_family(referenced),
            ", ".join(sorted(referenced)),
        )

    node_managed_missing = sorted(
        filename
        for filename in referenced
        if filename in NODE_MANAGED_MODEL_FILENAMES
        and not (Path("/comfyui/models") / _node_managed_model_dir(filename) / filename).exists()
    )
    if node_managed_missing:
        logger.info(
            "🧩 LTX node-managed models will resolve on first use: %s",
            ", ".join(node_managed_missing),
        )
        if job:
            _progress(
                job,
                f"Preparing LTX runtime for {len(node_managed_missing)} node-managed model(s)...",
            )

    requested = [
        model["filename"]
        for model in CLOUD_MAX_MODELS
        if model["filename"] in referenced and not _is_model_present(model)
    ]
    if not requested:
        if referenced:
            logger.info("✅ Workflow model check: all referenced models already available")
        return WorkflowModelPrepResult()

    cache_ready = []
    download_needed = []
    for filename in requested:
        model = next(item for item in CLOUD_MAX_MODELS if item["filename"] == filename)
        if _find_cached_model_source(model, emit_logs=False):
            cache_ready.append(filename)
        else:
            download_needed.append(filename)

    if cache_ready:
        logger.info("💾 Workflow cache hits pending link: %s", ", ".join(cache_ready))
    if download_needed:
        logger.warning("⬇️ Workflow models still require download: %s", ", ".join(download_needed))

    if job:
        if cache_ready and download_needed:
            _progress(
                job,
                f"Preparing {len(requested)} required model(s): linking {len(cache_ready)} from cache, downloading {len(download_needed)}...",
            )
        elif cache_ready:
            _progress(job, f"Linking {len(cache_ready)} required model(s) from cache...")
        else:
            _progress(job, f"Downloading {len(download_needed)} required model(s) for workflow...")

    prepared = download_requested_models(requested)
    linked = min(len(cache_ready), prepared)
    downloaded = max(prepared - linked, 0)
    if prepared > 0:
        restart_comfyui()
    return WorkflowModelPrepResult(
        requested_count=len(requested),
        linked_count=linked,
        downloaded_count=downloaded,
    )


def _node_managed_model_dir(filename: str) -> str:
    """Return the expected ComfyUI model directory for node-managed assets."""
    if filename == "LTX2_video_vae_bf16.safetensors":
        return "vae"
    if filename.endswith(".gguf"):
        return "unet"
    return "text_encoders"


def ensure_models():
    """
    Ensure all required models are available. Tries strategies in order:
    1. RunPod cached-model / HF cache symlinks (fast, no worker download time)
    2. Download from HuggingFace to container disk

    The RunPod Network Volume is reserved for LoRAs and private/custom assets.
    """
    comfyui_models = Path("/comfyui/models")
    volume_path = Path(MODEL_VOLUME)
    volume_models = _writable_volume_models_dir()

    ensure_model_directories()

    if not volume_path.exists():
        logger.info(f"📁 Network Volume not mounted at {volume_path}; public models will use container disk")
    elif volume_models is None:
        logger.info(f"📁 {volume_path} is not writable Network Volume storage; public models will use container disk")
    else:
        logger.info(f"📁 Writable Network Volume detected at {volume_path}; reserved for LoRAs/private assets only")

    _log_startup_model_plan()

    # Clean up old Wan 2.1 models to free space on volume
    _cleanup_old_models(volume_models or (volume_path / "models"), comfyui_models)

    # Strategy 1: Link any files already available via RunPod cached-model storage.
    cached_linked = link_cached_models(required_only=True)
    if cached_linked:
        logger.info("✅ Required models linked from cached-model storage")

    if not _missing_models(required_only=True):
        logger.info("✅ All required models are available before any HF download")
        return True

    logger.warning("⚠️ Some required models are still missing, falling back to Hugging Face download")

    # Strategy 2: Download from HuggingFace to container storage.
    logger.info("📥 Downloading missing models from HuggingFace...")
    if not download_models():
        return False

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
_comfyui_recent_logs = deque(maxlen=80)


def wait_for_cuda(max_wait: int = 60) -> bool:
    """
    Wait for CUDA to become available before starting ComfyUI.

    On RunPod serverless cold starts, the GPU may not be immediately
    available when the container starts. This check prevents ComfyUI
    from crashing with 'CUDA-capable device(s) is/are busy or unavailable'.

    Uses in-process torch check instead of spawning a subprocess per attempt
    (~2-5s overhead per subprocess avoided).
    """
    logger.info("🔍 Checking CUDA availability...")
    start = time.time()
    attempt = 0

    while (time.time() - start) < max_wait:
        attempt += 1
        try:
            import torch
            if torch.cuda.is_available():
                d = torch.cuda.device_count()
                name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                mem_bytes = getattr(props, "total_memory", getattr(props, "total_mem", 0))
                mem = mem_bytes / 1024**3
                logger.info(f"✅ CUDA ready (attempt {attempt}): "
                          f"{name}, {mem:.1f}GB VRAM, {d} device(s)")
                return True
            else:
                logger.warning(f"⚠️ CUDA check attempt {attempt}: not available yet")
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

        _comfyui_recent_logs.clear()

        # Stream ComfyUI logs in background thread
        def log_reader(proc=_comfyui_process):
            for line in iter(proc.stdout.readline, b''):
                decoded = line.decode(errors="replace").rstrip()
                _comfyui_recent_logs.append(decoded)
                logger.info(f"[ComfyUI] {decoded}")
        threading.Thread(target=log_reader, daemon=True).start()

        # Wait for ComfyUI to be ready
        max_wait = 120  # seconds
        start = time.time()
        while (time.time() - start) < max_wait:
            # Check if process crashed
            if _comfyui_process.poll() is not None:
                logger.error(f"❌ ComfyUI process exited with code {_comfyui_process.returncode}")
                if _comfyui_recent_logs:
                    logger.error("❌ Last ComfyUI log lines before exit:")
                    for recent_line in _comfyui_recent_logs:
                        logger.error(f"[ComfyUI][tail] {recent_line}")
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
            if _comfyui_recent_logs:
                logger.error("❌ Recent ComfyUI startup log lines before timeout:")
                for recent_line in _comfyui_recent_logs:
                    logger.error(f"[ComfyUI][tail] {recent_line}")
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
    Downloads to Network Volume (persistent cache) if writable,
    otherwise to container's ComfyUI loras dir (ephemeral).

    Skips LoRAs that already exist (cached from previous jobs).
    """
    volume_models = _writable_volume_models_dir()
    volume_loras = volume_models / "loras" if volume_models is not None else None
    comfyui_loras = Path("/comfyui/models/loras")

    # Prefer volume (persistent across jobs) over container disk
    target_dir = volume_loras if volume_loras is not None else comfyui_loras
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
            # Ensure parent dirs exist for LoRAs in subdirectories (e.g. "wan 2.2/file.safetensors")
            target.parent.mkdir(parents=True, exist_ok=True)

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
            if volume_loras is not None and target_dir == volume_loras and not comfyui_target.exists():
                comfyui_target.parent.mkdir(parents=True, exist_ok=True)
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


def _log_workflow_settings(workflow: dict) -> None:
    """Extract and log key generation settings from a ComfyUI workflow."""
    settings = {}
    for node_id, node in workflow.items():
        ct = node.get("class_type", "")
        inputs = node.get("inputs", {})
        if ct == "KSamplerAdvanced":
            label = "pass1" if inputs.get("add_noise") == "enable" else "pass2"
            settings[f"sampler_{label}"] = {
                "steps": inputs.get("steps"),
                "cfg": inputs.get("cfg"),
                "sampler": inputs.get("sampler_name"),
                "scheduler": inputs.get("scheduler"),
                "start": inputs.get("start_at_step"),
                "end": inputs.get("end_at_step"),
            }
        elif ct == "KSampler":
            settings["sampler"] = {
                "steps": inputs.get("steps"),
                "cfg": inputs.get("cfg"),
                "sampler": inputs.get("sampler_name"),
                "scheduler": inputs.get("scheduler"),
            }
        elif ct in ("WanImageToVideo", "EmptyWanLatentVideo"):
            settings["video"] = {
                "width": inputs.get("width"),
                "height": inputs.get("height"),
                "frames": inputs.get("length"),
            }
        elif ct == "VHS_VideoCombine":
            settings["output_fps"] = inputs.get("frame_rate")
        elif ct == "UNETLoader":
            name = inputs.get("unet_name", "")
            existing = settings.get("models", [])
            existing.append(name)
            settings["models"] = existing
        elif ct == "ModelSamplingSD3":
            settings["shift"] = inputs.get("shift")
    msg = f"📋 Workflow settings: {json.dumps(settings, default=str)}"
    logger.info(msg)
    print(msg, flush=True)


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
    Sends RunPod progress_update when new logs appear or every 30s.

    Uses adaptive polling: 2s during first 30s (startup), then 5s during
    generation, to reduce unnecessary ComfyUI API calls on long jobs.
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

        # Send progress when new ComfyUI logs appear or every 30s heartbeat
        elapsed = time.time() - start
        current_log_len = len(_log_buffer.getvalue()) if _log_buffer else 0
        has_new_logs = current_log_len > last_log_len
        if job and (has_new_logs or elapsed - last_progress >= 30):
            _progress(job, f"Generating... {elapsed:.0f}s elapsed", log_locally=has_new_logs)
            last_progress = elapsed
            last_log_len = len(_log_buffer.getvalue()) if _log_buffer else 0

        # Adaptive polling: fast during startup, slower during generation
        poll_interval = 2 if elapsed < 30 else 5
        time.sleep(poll_interval)

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
    """Encode output files as base64 for API response.

    Uses streaming base64 encoding to avoid holding both raw bytes
    and base64 string in memory simultaneously for large files.
    """
    encoded = []
    for f in files:
        path = Path(f["path"])
        if path.exists():
            # Stream base64 encoding in 3MB chunks to limit peak memory
            chunks = []
            with open(path, "rb") as fh:
                while True:
                    chunk = fh.read(3 * 1024 * 1024)  # 3MB (divisible by 3 for base64)
                    if not chunk:
                        break
                    chunks.append(base64.b64encode(chunk).decode("ascii"))
            b64 = "".join(chunks)
            encoded.append({
                "filename": f["filename"],
                "data": b64,
                "type": f["type"],
                "size": f["size"],
            })
            logger.info(f"📤 Encoded: {f['filename']} ({f['size']} bytes)")
    return encoded


# ---- RunPod Handler ----

_log_buffer = None       # Set by handler(), captures all logs for the job
_log_sent_pos = 0        # Track how much of the log buffer was already sent
_LOG_MAX_BYTES = 2 * 1024 * 1024  # 2MB cap — prevents unbounded memory growth


def _progress(job, message: str, log_locally: bool = True):
    """Send a progress update to RunPod with only NEW log lines since last update.

    Previous implementation sent the ENTIRE accumulated log buffer on every
    heartbeat, causing exponentially growing payloads on long jobs (20+ min).
    Now only the delta is sent, keeping payloads small and constant-sized.
    """
    global _log_buffer, _log_sent_pos
    try:
        if _log_buffer:
            full = _log_buffer.getvalue()
            new_logs = full[_log_sent_pos:]
            _log_sent_pos = len(full)
            payload = {"message": message, "logs": new_logs}
        else:
            payload = message
        runpod.serverless.progress_update(job, payload)
        if log_locally:
            logger.info(f"📡 Progress: {message}")
    except Exception as e:
        logger.warning(f"⚠️ progress_update failed: {e}")


def _cleanup_output_dir():
    """Remove stale output files from previous jobs to free disk space."""
    try:
        output_path = Path(OUTPUT_DIR)
        if not output_path.exists():
            return
        removed = 0
        for f in output_path.iterdir():
            if f.is_file():
                f.unlink()
                removed += 1
        if removed:
            logger.info(f"🗑️ Cleaned {removed} stale output file(s)")
    except Exception as e:
        logger.warning(f"⚠️ Output cleanup failed: {e}")


def handler(event: dict) -> dict:
    """
    Main RunPod handler function.

    Receives a workflow, queues it in ComfyUI, waits for completion,
    and returns the output files (base64 encoded).

    Progress updates are sent via RunPod API during execution.
    All logs are captured and returned in the 'logs' field.
    """
    # Capture logs during this job (also used by _progress for real-time log streaming)
    global _log_buffer, _log_sent_pos
    log_buffer = io.StringIO()
    _log_buffer = log_buffer
    _log_sent_pos = 0
    log_handler = logging.StreamHandler(log_buffer)
    log_handler.setLevel(logging.DEBUG)
    log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(log_handler)

    start_time = time.time()
    input_data = event.get("input", {})

    try:
        _progress(event, "Job received, validating workflow...")

        # Clean old output files from previous jobs
        _cleanup_output_dir()

        workflow = input_data.get("workflow")
        if not workflow:
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
        workflow_models = ensure_workflow_models(workflow, job=event)
        if workflow_models.prepared_count > 0:
            if workflow_models.linked_count and workflow_models.downloaded_count:
                _progress(
                    event,
                    f"Reloaded ComfyUI after linking {workflow_models.linked_count} cached and downloading {workflow_models.downloaded_count} model(s)",
                )
            elif workflow_models.linked_count:
                _progress(event, f"Reloaded ComfyUI after linking {workflow_models.linked_count} cached model(s)")
            else:
                _progress(event, f"Reloaded ComfyUI after downloading {workflow_models.downloaded_count} model(s)")

        # Log workflow settings for debugging
        _log_workflow_settings(workflow)

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
            return {"error": "No output files generated", "prompt_id": prompt_id, "logs": log_buffer.getvalue()}

        # Encode outputs as base64
        total_size = sum(f["size"] for f in files)
        _progress(event, f"Encoding {len(files)} file(s) ({total_size / 1024 / 1024:.1f} MB)...")
        encoded_files = encode_outputs(files)

        elapsed = time.time() - start_time
        logger.info(f"✅ Job complete in {elapsed:.1f}s — {len(encoded_files)} files")
        _progress(event, f"Done! {len(encoded_files)} file(s) in {elapsed:.0f}s")

        return {
            "files": encoded_files,
            "prompt_id": prompt_id,
            "execution_time_s": round(elapsed, 1),
            "logs": log_buffer.getvalue(),
        }

    except TimeoutError as e:
        _progress(event, f"❌ Timeout: {e}")
        return {"error": str(e), "logs": log_buffer.getvalue()}
    except RuntimeError as e:
        _progress(event, f"❌ Error: {e}")
        return {"error": str(e), "logs": log_buffer.getvalue()}
    except Exception as e:
        logger.exception(f"❌ Handler error: {e}")
        _progress(event, f"❌ Unexpected error: {e}")
        return {"error": f"Unexpected error: {str(e)}", "logs": log_buffer.getvalue()}
    finally:
        logger.removeHandler(log_handler)
        _log_buffer = None
        _log_sent_pos = 0
        # Cap: if buffer grew too large, warn
        buf_size = len(log_buffer.getvalue())
        if buf_size > _LOG_MAX_BYTES:
            logger.warning(f"⚠️ Job log buffer was {buf_size / 1024:.0f}KB (cap: {_LOG_MAX_BYTES / 1024:.0f}KB)")
        log_buffer.close()


# ---- Startup ----

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🎬 Oelala ComfyUI Worker starting...")
    logger.info("=" * 60)

    # Ensure public models are available (cached-model hit or download from HF)
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
