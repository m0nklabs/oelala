"""
Face LoRA Training Service
--------------------------
Trains a Dreambooth-style SDXL face LoRA using ai-toolkit (ostris).

Base model : ComfyUI/models/checkpoints/juggernautXL_ragnarok.safetensors
Output     : ComfyUI/models/loras/face_loras/<trigger>.safetensors
Toolkit    : external/ai-toolkit/

Job lifecycle:
  pending  → running  → done | failed

Jobs are stored in data/face_train_jobs/index.json
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)
DEBUG = os.getenv("DEBUG", "0") == "1"

# ─────────────────────────────────────────────────────────────────────────────
# Progress callback (set by app.py to broadcast via WebSocket)
# ─────────────────────────────────────────────────────────────────────────────
# Signature: callback(job_id: str, event: str, data: dict) -> None
# event: "started" | "progress" | "completed" | "failed"
_progress_callback = None


def set_progress_callback(callback):
    """Set a callback for training progress events (called from app.py)."""
    global _progress_callback
    _progress_callback = callback


def _emit_event(job_id: str, event: str, data: dict):
    """Emit a progress event if a callback is registered."""
    if _progress_callback:
        try:
            _progress_callback(job_id, event, data)
        except Exception as e:
            logger.warning(f"Progress callback error: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # repo root

TOOLKIT_RUN = BASE_DIR / "external" / "ai-toolkit" / "run.py"
TOOLKIT_DIR = BASE_DIR / "external" / "ai-toolkit"
TOOLKIT_PYTHON = TOOLKIT_DIR / ".venv" / "bin" / "python"  # isolated venv
JOBS_DIR = BASE_DIR / "data" / "face_train_jobs"
JOBS_INDEX = JOBS_DIR / "index.json"
LORAS_OUTPUT_DIR = BASE_DIR / "ComfyUI" / "models" / "loras" / "face_loras"
BASE_MODEL = (
    BASE_DIR
    / "ComfyUI"
    / "models"
    / "checkpoints"
    / "juggernautXL_ragnarok.safetensors"
)

JOBS_DIR.mkdir(parents=True, exist_ok=True)
LORAS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Caption variants used in Dreambooth training
CAPTION_TEMPLATES = [
    "a photo of {trigger}, person",
    "portrait of {trigger}",
    "photo of {trigger} person, face, realistic",
    "{trigger} person, natural light portrait",
    "close up photo of {trigger},  detailed face",
    "a clear photo of {trigger} person",
]

# ─────────────────────────────────────────────────────────────────────────────
# Index helpers
# ─────────────────────────────────────────────────────────────────────────────


def _load_index() -> dict:
    if not JOBS_INDEX.exists():
        return {}
    with open(JOBS_INDEX) as f:
        return json.load(f)


def _save_index(data: dict) -> None:
    with open(JOBS_INDEX, "w") as f:
        json.dump(data, f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Startup recovery — mark orphaned 'running' jobs as failed
# ─────────────────────────────────────────────────────────────────────────────

def recover_stuck_jobs() -> int:
    """
    Called at backend startup. Any job with status 'running' is orphaned
    because daemon training threads don't survive process restarts.
    Mark them as 'failed' so users can retry.
    Returns the number of recovered jobs.
    """
    index = _load_index()
    recovered = 0
    for job_id, job in index.items():
        if job.get("status") == "running":
            # Try to extract last known step from log
            log_path = JOBS_DIR / job_id / "training.log"
            last_step = _extract_last_step(log_path) if log_path.exists() else 0
            job["status"] = "failed"
            job["error"] = (
                f"Backend restarted during training "
                f"(orphaned at step ~{last_step}/{job.get('steps_total', '?')})"
            )
            if last_step > job.get("steps_done", 0):
                job["steps_done"] = last_step
            job["finished_at"] = time.time()
            recovered += 1
            logger.warning(
                f"🔄 Recovered orphaned training job {job_id} "
                f"(was at step {last_step})"
            )
    if recovered:
        _save_index(index)
    return recovered


def _extract_last_step(log_path: Path) -> int:
    """Parse the training log to find the highest step reached."""
    try:
        with open(log_path) as f:
            lines = f.readlines()
        step = 0
        for line in reversed(lines[-100:]):
            m = re.search(r"(\d+)/\d+.*lr:", line)
            if m:
                step = max(step, int(m.group(1)))
                break
        return step
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# GPU device selection
# ─────────────────────────────────────────────────────────────────────────────

def _select_training_device() -> str:
    """
    Pick the best CUDA device for training.
    Prefers the GPU with the most free VRAM.
    Falls back to cuda:0 if torch is unavailable.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("⚠️ No CUDA available, training will be very slow on CPU")
            return "cpu"

        best_device = "cuda:0"
        best_free = 0
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            name = torch.cuda.get_device_name(i)
            free_gb = free / (1024 ** 3)
            total_gb = total / (1024 ** 3)
            logger.info(
                f"🔍 GPU {i} ({name}): {free_gb:.1f}GB free / {total_gb:.1f}GB total"
            )
            if free > best_free:
                best_free = free
                best_device = f"cuda:{i}"

        logger.info(f"✅ Selected training device: {best_device} ({best_free / (1024**3):.1f}GB free)")
        return best_device
    except Exception as e:
        logger.warning(f"⚠️ GPU detection failed ({e}), defaulting to cuda:0")
        return "cuda:0"


def _sanitize_trigger(name: str) -> str:
    """Convert name to a safe Dreambooth trigger word like ohwx_john_doe."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return f"ohwx_{slug}"


# ─────────────────────────────────────────────────────────────────────────────
# Config generation
# ─────────────────────────────────────────────────────────────────────────────


def _build_training_config(
    job_id: str,
    trigger: str,
    images_dir: Path,
    output_dir: Path,
    steps: int = 1000,
) -> dict:
    """Build an ai-toolkit YAML config dict for SDXL face LoRA."""
    device = _select_training_device()
    return {
        "job": "extension",
        "config": {
            "name": f"face_lora_{trigger}",
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": str(output_dir),
                    "device": device,
                    "trigger_word": trigger,
                    "network": {
                        "type": "lora",
                        "linear": 16,
                        "linear_alpha": 16,
                    },
                    "save": {
                        "dtype": "float16",
                        "save_every": 250,
                        "max_step_saves_to_keep": 2,
                        "push_to_hub": False,
                    },
                    "datasets": [
                        {
                            "folder_path": str(images_dir),
                            "caption_ext": "txt",
                            "caption_dropout_rate": 0.05,
                            "shuffle_tokens": False,
                            "cache_latents_to_disk": True,
                            "resolution": [512, 768, 1024],
                        }
                    ],
                    "train": {
                        "batch_size": 1,
                        "steps": steps,
                        "gradient_accumulation_steps": 1,
                        "train_unet": True,
                        "train_text_encoder": False,
                        "gradient_checkpointing": True,
                        "noise_scheduler": "ddpm",
                        "optimizer": "adamw8bit",
                        "lr": 1e-4,
                        "ema_config": {
                            "use_ema": True,
                            "ema_decay": 0.99,
                        },
                        "dtype": "bf16",
                    },
                    "model": {
                        "name_or_path": str(BASE_MODEL),
                        "is_xl": True,
                        "arch": "sdxl",
                    },
                    "sample": {
                        "sampler": "ddpm",
                        "sample_every": 250,
                        "width": 1024,
                        "height": 1024,
                        "prompts": [
                            "professional portrait photo of [trigger] person, sharp focus, detailed face, natural lighting",
                            "candid photo of [trigger] person, realistic, natural expression",
                        ],
                        "neg": "blurry, deformed face, bad anatomy, extra limbs",
                        "seed": 42,
                        "walk_seed": True,
                        "guidance_scale": 7.5,
                        "sample_steps": 20,
                    },
                }
            ],
        },
        "meta": {
            "name": f"face_lora_{trigger}",
            "version": "1.0",
        },
    }


def _write_captions(images_dir: Path, trigger: str) -> None:
    """Create .txt caption file for every image in the dataset directory."""
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    imgs = [p for p in images_dir.iterdir() if p.suffix.lower() in image_exts]
    for i, img in enumerate(imgs):
        cap_path = img.with_suffix(".txt")
        template = CAPTION_TEMPLATES[i % len(CAPTION_TEMPLATES)]
        cap_path.write_text(template.format(trigger=trigger))
    if DEBUG:
        logger.debug(f"🐛 Wrote {len(imgs)} captions in {images_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────


def create_training_job(
    name: str,
    images: list,  # list of PIL Image or bytes, saved to disk
    description: str = "",
    steps: int = 1000,
) -> dict:
    """
    Create and START a face LoRA training job.

    Returns the job dict.
    Raises ValueError if base model is missing.
    """
    if not BASE_MODEL.exists():
        raise ValueError(
            f"Base model not found: {BASE_MODEL}. "
            "juggernautXL_ragnarok.safetensors is required for SDXL face LoRA training."
        )
    if not TOOLKIT_RUN.exists():
        raise ValueError(f"ai-toolkit not found at {TOOLKIT_DIR}")
    if not TOOLKIT_PYTHON.exists():
        raise ValueError(
            f"ai-toolkit venv not found at {TOOLKIT_PYTHON}. "
            "Run: python3.12 -m venv external/ai-toolkit/.venv && "
            "external/ai-toolkit/.venv/bin/pip install -r external/ai-toolkit/requirements.txt"
        )

    job_id = str(uuid.uuid4())[:8]
    trigger = _sanitize_trigger(name)
    job_dir = JOBS_DIR / job_id
    images_dir = job_dir / "images"
    output_dir = job_dir / "output"
    images_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save images to training dataset dir
    from PIL import Image as PILImage
    import io

    for idx, img_data in enumerate(images):
        img_path = images_dir / f"face_{idx:03d}.png"
        if isinstance(img_data, bytes):
            img = PILImage.open(io.BytesIO(img_data)).convert("RGB")
        elif hasattr(img_data, "save"):  # PIL Image
            img = img_data.convert("RGB")
        else:
            raise ValueError(f"Unknown image type: {type(img_data)}")
        # Resize to max 1024x1024 while maintaining aspect ratio
        img.thumbnail((1024, 1024), PILImage.LANCZOS)
        img.save(img_path, "PNG")

    _write_captions(images_dir, trigger)

    # Write YAML config
    config = _build_training_config(job_id, trigger, images_dir, output_dir, steps)
    config_path = job_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    # Persist job metadata
    index = _load_index()
    job = {
        "id": job_id,
        "name": name,
        "description": description,
        "trigger": trigger,
        "status": "pending",
        "steps_total": steps,
        "steps_done": 0,
        "lora_path": None,
        "error": None,
        "created_at": time.time(),
        "started_at": None,
        "finished_at": None,
        "images_count": len(images),
    }
    index[job_id] = job
    _save_index(index)

    # Launch training subprocess
    _launch_training(job_id, job_dir, config_path)
    return job


def _launch_training(job_id: str, job_dir: Path, config_path: Path) -> None:
    """Launch ai-toolkit training as a background subprocess."""
    # Use ai-toolkit's own isolated venv to avoid dependency conflicts
    python_bin = str(TOOLKIT_PYTHON) if TOOLKIT_PYTHON.exists() else sys.executable
    log_path = job_dir / "training.log"
    lora_dest = LORAS_OUTPUT_DIR  # ai-toolkit writes to output/ inside job_dir

    def _run():
        import subprocess as sp

        index = _load_index()
        job = index.get(job_id, {})
        job["status"] = "running"
        job["started_at"] = time.time()
        _save_index({**index, job_id: job})

        _emit_event(job_id, "started", {
            "name": job.get("name", ""),
            "trigger": job.get("trigger", ""),
            "steps_total": job.get("steps_total", 0),
        })

        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(TOOLKIT_DIR) + ":" + env.get("PYTHONPATH", "")

            with open(log_path, "w") as logf:
                proc = sp.Popen(
                    [python_bin, str(TOOLKIT_RUN), str(config_path)],
                    cwd=str(TOOLKIT_DIR),
                    env=env,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                )

                # Poll progress by watching log lines
                while proc.poll() is None:
                    time.sleep(5)
                    _update_progress_from_log(job_id, log_path)

                ret = proc.returncode

            if ret == 0:
                # Find the output LoRA file
                trigger = job.get("trigger", "unknown")
                lora_file = _find_output_lora(job_dir, trigger)

                if lora_file:
                    # Copy to canonical loras dir
                    dest = LORAS_OUTPUT_DIR / f"{trigger}.safetensors"
                    shutil.copy2(lora_file, dest)
                    job["lora_path"] = str(dest)
                    job["status"] = "done"
                    logger.info(f"✅ Face LoRA training done: {dest}")
                    _emit_event(job_id, "completed", {
                        "lora_path": str(dest),
                        "trigger": trigger,
                    })
                else:
                    job["status"] = "failed"
                    job["error"] = "Training finished but no output LoRA found"
                    _emit_event(job_id, "failed", {"error": job["error"]})
            else:
                job["status"] = "failed"
                job["error"] = f"Process exited with code {ret}"
                logger.error(f"❌ Face LoRA training failed (code {ret}): {job_id}")
                _emit_event(job_id, "failed", {"error": job["error"]})

        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            logger.exception(f"❌ Face LoRA training exception: {job_id}")
            _emit_event(job_id, "failed", {"error": str(e)})

        job["finished_at"] = time.time()
        index = _load_index()
        _save_index({**index, job_id: job})

    import threading

    t = threading.Thread(target=_run, daemon=True, name=f"face-train-{job_id}")
    t.start()
    logger.info(f"🎯 Face LoRA training started: job={job_id}, log={log_path}")


def _update_progress_from_log(job_id: str, log_path: Path) -> None:
    """Parse training log to extract current step count."""
    if not log_path.exists():
        return
    try:
        with open(log_path) as f:
            lines = f.readlines()
        # ai-toolkit logs lines like: "step: 100/1000 loss: 0.123"
        step = 0
        for line in reversed(lines[-50:]):  # last 50 lines
            m = re.search(r"step[:\s]+(\d+)", line, re.IGNORECASE)
            if m:
                step = int(m.group(1))
                break
        if step > 0:
            index = _load_index()
            job = index.get(job_id, {})
            if job.get("steps_done", 0) != step:
                job["steps_done"] = step
                _save_index({**index, job_id: job})
                # Broadcast progress via WebSocket
                steps_total = job.get("steps_total", 0)
                progress = round((step / steps_total) * 100) if steps_total else 0
                _emit_event(job_id, "progress", {
                    "steps_done": step,
                    "steps_total": steps_total,
                    "progress": progress,
                })
    except Exception:
        pass


def _find_output_lora(job_dir: Path, trigger: str) -> Path | None:
    """Find the final output LoRA safetensors in the job output directory."""
    output_dir = job_dir / "output"
    # ai-toolkit saves to output/{config_name}/{config_name}_{steps}.safetensors
    for candidate in output_dir.rglob("*.safetensors"):
        if "sample" not in str(candidate).lower():
            return candidate
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Status / list
# ─────────────────────────────────────────────────────────────────────────────


def get_job(job_id: str) -> dict | None:
    return _load_index().get(job_id)


def list_jobs() -> list[dict]:
    return list(_load_index().values())


def cancel_job(job_id: str) -> bool:
    """Mark job as cancelled (cannot kill subprocess retroactively but prevents re-launch)."""
    index = _load_index()
    job = index.get(job_id)
    if not job:
        return False
    if job["status"] in ("pending", "running"):
        job["status"] = "cancelled"
        job["finished_at"] = time.time()
        _save_index({**index, job_id: job})
        return True
    return False


def retry_job(job_id: str) -> dict | None:
    """Retry a failed training job by resetting status and re-launching."""
    index = _load_index()
    job = index.get(job_id)
    if not job:
        return None
    if job["status"] not in ("failed", "cancelled"):
        return None

    job_dir = JOBS_DIR / job_id
    config_path = job_dir / "config.yaml"
    if not config_path.exists():
        return None

    # Reset job state
    job["status"] = "pending"
    job["steps_done"] = 0
    job["error"] = None
    job["lora_path"] = None
    job["started_at"] = None
    job["finished_at"] = None
    _save_index({**index, job_id: job})

    # Re-launch
    _launch_training(job_id, job_dir, config_path)
    logger.info(f"🔄 Retrying face LoRA training: {job_id}")
    return job


def list_trained_loras() -> list[dict]:
    """Return all .safetensors files in the face_loras output dir."""
    loras = []
    for f in sorted(LORAS_OUTPUT_DIR.glob("*.safetensors")):
        stat = f.stat()
        loras.append(
            {
                "filename": f.name,
                "path": str(f),
                "size_mb": round(stat.st_size / 1024 / 1024, 1),
                "trigger": f.stem,  # e.g. ohwx_john_doe
                "modified": stat.st_mtime,
            }
        )
    return loras
