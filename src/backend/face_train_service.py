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
from typing import Literal

import yaml

logger = logging.getLogger(__name__)
DEBUG = os.getenv("DEBUG", "0") == "1"

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent.parent  # repo root

TOOLKIT_RUN = BASE_DIR / "external" / "ai-toolkit" / "run.py"
TOOLKIT_DIR = BASE_DIR / "external" / "ai-toolkit"
JOBS_DIR = BASE_DIR / "data" / "face_train_jobs"
JOBS_INDEX = JOBS_DIR / "index.json"
LORAS_OUTPUT_DIR = BASE_DIR / "ComfyUI" / "models" / "loras" / "face_loras"
BASE_MODEL = BASE_DIR / "ComfyUI" / "models" / "checkpoints" / "juggernautXL_ragnarok.safetensors"

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
    return {
        "job": "extension",
        "config": {
            "name": f"face_lora_{trigger}",
            "process": [
                {
                    "type": "sd_trainer",
                    "training_folder": str(output_dir),
                    "device": "cuda:0",
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
                            f"professional portrait photo of [trigger] person, sharp focus, detailed face, natural lighting",
                            f"candid photo of [trigger] person, realistic, natural expression",
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
    python_bin = sys.executable
    log_path = job_dir / "training.log"
    lora_dest = LORAS_OUTPUT_DIR  # ai-toolkit writes to output/ inside job_dir

    def _run():
        import subprocess as sp
        index = _load_index()
        job = index.get(job_id, {})
        job["status"] = "running"
        job["started_at"] = time.time()
        _save_index({**index, job_id: job})

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
                else:
                    job["status"] = "failed"
                    job["error"] = "Training finished but no output LoRA found"
            else:
                job["status"] = "failed"
                job["error"] = f"Process exited with code {ret}"
                logger.error(f"❌ Face LoRA training failed (code {ret}): {job_id}")

        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)
            logger.exception(f"❌ Face LoRA training exception: {job_id}")

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


def list_trained_loras() -> list[dict]:
    """Return all .safetensors files in the face_loras output dir."""
    loras = []
    for f in sorted(LORAS_OUTPUT_DIR.glob("*.safetensors")):
        stat = f.stat()
        loras.append({
            "filename": f.name,
            "path": str(f),
            "size_mb": round(stat.st_size / 1024 / 1024, 1),
            "trigger": f.stem,  # e.g. ohwx_john_doe
            "modified": stat.st_mtime,
        })
    return loras
