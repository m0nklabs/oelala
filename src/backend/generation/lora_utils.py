"""
LoRA utility functions extracted from app.py.

These functions handle LoRA path resolution, format sanitization,
model compatibility filtering, and cloud download URL generation.

The originals in app.py are kept as re-exports so nothing breaks.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional
from urllib.parse import quote

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────
# These match the paths used in app.py
LORA_DIR = Path(os.getenv("LORA_DIR", "/home/flip/oelala/ComfyUI/models/loras"))
LORA_SSD_DIR = Path(os.getenv("LORA_SSD_DIR", "/mnt/ssd/loras"))


def resolve_lora_path(name: str) -> tuple[Optional[Path], Optional[str]]:
    """
    Resolve a LoRA name to its actual file path.

    Handles names with or without .safetensors extension,
    and LoRAs in subdirectories. Searches both primary LORA_DIR
    and SSD fallback directory.

    Returns (full_path, filename) or (None, None) if not found.
    """
    # Prevent directory traversal from user-supplied names
    if ".." in name or name.startswith("/"):
        logger.warning(f"⚠️ Rejected suspicious LoRA name: {name}")
        return None, None

    for base_dir in [LORA_DIR, LORA_SSD_DIR]:
        if not base_dir.exists():
            continue

        # Try exact path first
        exact = base_dir / name
        if exact.is_file():
            return exact, name

        # Try adding .safetensors extension
        with_ext = base_dir / f"{name}.safetensors"
        if with_ext.is_file():
            return with_ext, f"{name}.safetensors"

        # Search subdirectories for exact filename match
        for match in base_dir.rglob(name):
            if match.is_file():
                return match, str(match.relative_to(base_dir))

        # Search subdirectories with extension added
        for match in base_dir.rglob(f"{name}.safetensors"):
            if match.is_file():
                return match, str(match.relative_to(base_dir))

    return None, None


def sanitize_lora_configs_for_single_stage(lora_configs: list[dict]) -> list[dict]:
    """
    Convert Wan2.2 dual-stage LoRA configs ({high, low, strength}) to
    single-stage format ({name, strength}) for LTX-2.3 / Qwen workflows.

    If a Wan-format config is detected, only the 'high' key is kept as 'name'.
    Already-correct single-stage configs ({name, strength}) pass through unchanged.
    """
    sanitized: list[dict] = []
    for config in lora_configs:
        if "name" in config and config["name"]:
            sanitized.append(config)
        elif "high" in config and config["high"]:
            logger.warning(
                f"⚠️ Converting Wan2.2 dual-stage LoRA to single-stage: "
                f"high={config.get('high')} (low={config.get('low')} dropped)"
            )
            sanitized.append(
                {"name": config["high"], "strength": config.get("strength", 1.0)}
            )
        else:
            logger.warning(f"⚠️ Skipping LoRA config with no name/high key: {config}")
    return sanitized


def filter_loras_by_model_compat(
    lora_configs: list[dict], target_model: str
) -> list[dict]:
    """
    Filter LoRA configs to only include LoRAs compatible with the target model.

    Uses filename-based base_model derivation (same logic as lora_scanner).
    """
    try:
        from lora_scanner import _derive_base_model
    except ImportError:
        logger.warning("⚠️ lora_scanner not available, skipping LoRA compat filter")
        return lora_configs

    compatible: list[dict] = []
    for config in lora_configs:
        lora_name = config.get("name") or config.get("high") or ""
        if not lora_name:
            continue
        base_model = _derive_base_model(lora_name)
        if base_model == target_model or base_model == "":
            compatible.append(config)
        else:
            logger.warning(
                f"🚫 LoRA '{lora_name}' is for {base_model}, incompatible with "
                f"{target_model} — skipping"
            )
    if len(compatible) < len(lora_configs):
        logger.info(
            f"🔍 LoRA compat filter: {len(compatible)}/{len(lora_configs)} "
            f"passed for target={target_model}"
        )
    return compatible


def lora_download_token(filename: str) -> str:
    """
    Generate HMAC-SHA256 token for LoRA download URL validation.

    Must match the check performed by app.py's /loras/download endpoint:
    same key (RUNPOD_API_KEY), same digest truncation (first 32 hex chars),
    computed over the same (normalized) filename that appears in the URL path.
    """
    import hashlib
    import hmac

    key = os.getenv("RUNPOD_API_KEY", "fallback-lora-key").encode()
    return hmac.new(key, filename.encode(), hashlib.sha256).hexdigest()[:32]


def build_lora_download_list(
    lora_configs: list[dict],
    *,
    backend_public_url: Optional[str] = None,
    hf_sources: Optional[dict] = None,
    hf_token: Optional[str] = None,
) -> list[dict]:
    """
    Build download URLs for LoRAs needed by a cloud job.

    Source priority per LoRA:
      1. explicit `hf_sources` mapping (curated public HF repos)
      2. the flat mirror repo (LORA_HF_FLAT_MIRROR_REPO env, basename-mapped)
      3. the subdir mirror repo (LORA_HF_MIRROR_REPO env, layout-preserving)
      4. the backend's signed /loras/download endpoint
    A non-HF primary URL always carries the signed self-hosted URL as
    `fallback_url`, so workers can recover when the file is missing on HF.
    """
    base_url = backend_public_url or os.getenv(
        "BACKEND_PUBLIC_URL", "https://api.oelala.xyz"
    )
    # HuggingFace mirror options, tried before the self-hosted download:
    #  - flat mirror (LORA_HF_FLAT_MIRROR_REPO): public repo whose files are
    #    stored by basename (no subdirs) — e.g. a shared dump repo.
    #  - subdir mirror (LORA_HF_MIRROR_REPO): private repo preserving the
    #    local subdirectory layout, populated by scripts/sync_loras_hf.py.
    hf_flat_mirror_repo = os.getenv("LORA_HF_FLAT_MIRROR_REPO", "")
    hf_mirror_repo = os.getenv("LORA_HF_MIRROR_REPO", "")
    hf_mirror_token = os.getenv("HF_LORA_TOKEN", "")
    downloads: list[dict] = []
    seen: set[str] = set()

    for config in lora_configs:
        keys_to_check = ["high", "low", "name"]
        for key in keys_to_check:
            name = config.get(key, "")
            if not name or name in seen:
                continue
            seen.add(name)
            lora_path, resolved_name = resolve_lora_path(name)
            if not lora_path or not resolved_name:
                logger.warning(f"⚠️ LoRA not found locally for cloud upload: {name}")
                continue
            config[key] = resolved_name

            # HuggingFace CDN first (much faster than the self-hosted home
            # uplink); the signed self-hosted URL rides along as fallback_url
            # so workers can recover when the file is missing on HF (404/401).
            signed_url = (
                f"{base_url}/loras/download/{resolved_name}"
                f"?token={lora_download_token(resolved_name)}"
            )
            if hf_sources and resolved_name in hf_sources:
                hf_source = hf_sources[resolved_name]
                hf_repo = hf_source["repo"]
                hf_path = hf_source.get("path", resolved_name)
                hf_url = f"https://huggingface.co/{hf_repo}/resolve/main/{hf_path}"
                entry: dict = {
                    "filename": resolved_name,
                    "url": hf_url,
                    "fallback_url": signed_url,
                }
                if hf_token:
                    entry["hf_token"] = hf_token
                downloads.append(entry)
            elif hf_flat_mirror_repo:
                # Flat mirror repo: files stored by basename (no subdirs).
                # Public repos need no token.
                flat_url = (
                    f"https://huggingface.co/{hf_flat_mirror_repo}"
                    f"/resolve/main/{quote(resolved_name.split('/')[-1])}"
                )
                downloads.append(
                    {
                        "filename": resolved_name,
                        "url": flat_url,
                        "fallback_url": signed_url,
                    }
                )
            elif hf_mirror_repo:
                mirror_url = (
                    f"https://huggingface.co/{hf_mirror_repo}"
                    f"/resolve/main/{resolved_name}"
                )
                entry = {
                    "filename": resolved_name,
                    "url": mirror_url,
                    "fallback_url": signed_url,
                }
                if hf_mirror_token:
                    entry["hf_token"] = hf_mirror_token
                downloads.append(entry)
            else:
                downloads.append(
                    {
                        "filename": resolved_name,
                        "url": signed_url,
                    }
                )
    return downloads
