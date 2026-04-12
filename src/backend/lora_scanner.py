"""
LoRA Scanner — Scans and caches LoRA model metadata.

Discovers LoRA files from ComfyUI model paths and /mnt/ssd/loras/,
extracts metadata from safetensors headers, and caches results.
"""

import json
import logging
import os
import struct
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("lora_scanner")

DEBUG = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")


def debug_log(msg: str):
    if DEBUG:
        logger.info(f"🔍 [LoRA Scanner] {msg}")


# LoRA directories to scan
LORA_DIRS: List[Path] = [
    Path("/mnt/ssd/loras"),
    Path("/home/flip/oelala/ComfyUI/models/loras"),
]


@dataclass
class LoRAInfo:
    """Metadata for a single LoRA model."""

    id: str  # Unique ID (hash of relative path)
    filename: str  # Original filename
    name: str  # Display name (derived from filename)
    path: str  # Relative path within lora dir
    full_path: str  # Absolute path
    size_bytes: int
    size_mb: float
    modified: float  # Unix timestamp
    category: str  # Derived from subdirectory or filename
    tags: List[str] = field(default_factory=list)
    base_model: str = ""  # e.g., "wan2.2", "sdxl", "pony"
    noise_level: str = ""  # "high", "low", or ""
    format: str = ""  # From safetensors metadata
    rank: str = ""  # LoRA rank if detectable


def _derive_name(filename: str) -> str:
    """Derive a display name from filename."""
    name = Path(filename).stem
    # Remove common suffixes
    for suffix in [".safetensors", ".pt", ".ckpt"]:
        name = name.replace(suffix, "")
    # Replace underscores and hyphens with spaces
    name = name.replace("_", " ").replace("-", " ")
    # Clean up multiple spaces
    name = " ".join(name.split())
    return name


def _derive_tags(filename: str, category: str) -> List[str]:
    """Derive tags from filename and category."""
    tags = []
    lower = filename.lower()

    # Noise level
    if "high" in lower or "hn" in lower or "highnoise" in lower:
        tags.append("high-noise")
    if "low" in lower or "ln" in lower or "lownoise" in lower:
        tags.append("low-noise")

    # Model type
    if "wan" in lower or "w22" in lower:
        tags.append("wan2.2")
    if "ltx" in lower:
        tags.append("ltx")
    if "sdxl" in lower:
        tags.append("sdxl")
    if "pony" in lower:
        tags.append("pony")

    # Mode
    if "i2v" in lower:
        tags.append("i2v")
    if "t2v" in lower:
        tags.append("t2v")

    # Style markers
    if "realism" in lower or "realistic" in lower or "realskin" in lower:
        tags.append("realism")
    if "detail" in lower:
        tags.append("detail")
    if "skin" in lower:
        tags.append("skin")
    if "bounce" in lower:
        tags.append("motion")
    if "slider" in lower:
        tags.append("slider")
    if "style" in lower:
        tags.append("style")
    if "step" in lower or "distill" in lower:
        tags.append("speed")

    # Category as tag
    if category and category != "root":
        tags.append(category)

    return list(set(tags))


def _derive_noise_level(filename: str) -> str:
    """Derive noise level from filename."""
    lower = filename.lower()
    if any(x in lower for x in ["highnoise", "high_noise", "high noise", "_hn", "-hn"]):
        return "high"
    if any(x in lower for x in ["lownoise", "low_noise", "low noise", "_ln", "-ln"]):
        return "low"
    if "high" in lower and "noise" not in lower:
        # Check if "high" is part of the name (e.g., "HIGH-v1")
        if "high" in lower.split("-") or "high" in lower.split("_"):
            return "high"
    if "low" in lower and "noise" not in lower:
        if "low" in lower.split("-") or "low" in lower.split("_"):
            return "low"
    return ""


def _derive_base_model(filename: str) -> str:
    """Derive base model from filename."""
    lower = filename.lower()
    if "wan" in lower or "w22" in lower:
        return "wan2.2"
    if "ltx" in lower:
        return "ltx"
    if "sdxl" in lower or "xl" in lower:
        return "sdxl"
    if "pony" in lower:
        return "pony"
    if "sd15" in lower or "sd1.5" in lower:
        return "sd1.5"
    return ""


def _read_safetensors_metadata(filepath: str) -> Dict:
    """Read metadata from safetensors file header."""
    try:
        with open(filepath, "rb") as f:
            header_size = struct.unpack("<Q", f.read(8))[0]
            # Limit header read to 1MB to avoid memory issues
            header_bytes = f.read(min(header_size, 1_048_576))
            header = json.loads(header_bytes)
            return header.get("__metadata__", {})
    except Exception as e:
        debug_log(f"Failed to read metadata from {filepath}: {e}")
        return {}


def _make_id(path: str) -> str:
    """Create a stable ID from a path."""
    import hashlib

    return hashlib.md5(path.encode()).hexdigest()[:12]


def scan_lora_directory(lora_dir: Path) -> List[LoRAInfo]:
    """Scan a single LoRA directory for models."""
    results = []
    if not lora_dir.exists():
        return results

    for filepath in lora_dir.rglob("*.safetensors"):
        try:
            stat = filepath.stat()
            rel_path = str(filepath.relative_to(lora_dir))

            # Category from subdirectory
            parts = Path(rel_path).parts
            category = parts[0] if len(parts) > 1 else "root"

            # Read safetensors metadata
            meta = _read_safetensors_metadata(str(filepath))

            info = LoRAInfo(
                id=_make_id(rel_path),
                filename=filepath.name,
                name=_derive_name(filepath.name),
                path=rel_path,
                full_path=str(filepath),
                size_bytes=stat.st_size,
                size_mb=round(stat.st_size / (1024 * 1024), 1),
                modified=stat.st_mtime,
                category=category,
                tags=_derive_tags(filepath.name, category),
                base_model=_derive_base_model(filepath.name),
                noise_level=_derive_noise_level(filepath.name),
                format=meta.get("format", ""),
                rank=meta.get("rank", ""),
            )
            results.append(info)
        except Exception as e:
            debug_log(f"Error scanning {filepath}: {e}")

    return results


class LoRACache:
    """Cached LoRA scanner with TTL."""

    def __init__(self, ttl_seconds: int = 300):
        self._cache: List[LoRAInfo] = []
        self._last_scan: float = 0
        self._ttl = ttl_seconds

    def get_all(self, force_refresh: bool = False) -> List[LoRAInfo]:
        """Get all LoRAs, scanning if cache is stale."""
        now = time.time()
        if force_refresh or not self._cache or (now - self._last_scan) > self._ttl:
            self._scan()
        return self._cache

    def _scan(self):
        """Scan all LoRA directories."""
        debug_log("Starting LoRA scan...")
        start = time.time()
        all_loras = []
        for lora_dir in LORA_DIRS:
            loras = scan_lora_directory(lora_dir)
            all_loras.extend(loras)
            debug_log(f"Found {len(loras)} LoRAs in {lora_dir}")

        # Sort by name
        all_loras.sort(key=lambda x: x.name.lower())
        self._cache = all_loras
        self._last_scan = time.time()
        debug_log(
            f"LoRA scan complete: {len(all_loras)} total in {time.time() - start:.2f}s"
        )

    def search(self, query: str) -> List[LoRAInfo]:
        """Search LoRAs by name, tags, or category."""
        all_loras = self.get_all()
        if not query:
            return all_loras

        q = query.lower()
        results = []
        for lora in all_loras:
            if (
                q in lora.name.lower()
                or q in lora.filename.lower()
                or q in lora.category.lower()
                or any(q in tag for tag in lora.tags)
                or q in lora.base_model.lower()
            ):
                results.append(lora)
        return results

    def get_by_id(self, lora_id: str) -> Optional[LoRAInfo]:
        """Get a specific LoRA by ID."""
        for lora in self.get_all():
            if lora.id == lora_id:
                return lora
        return None

    def get_categories(self) -> List[Dict]:
        """Get unique categories with counts."""
        cats: Dict[str, int] = {}
        for lora in self.get_all():
            cats[lora.category] = cats.get(lora.category, 0) + 1
        return [{"name": k, "count": v} for k, v in sorted(cats.items())]

    def get_tags(self) -> List[Dict]:
        """Get unique tags with counts."""
        tag_counts: Dict[str, int] = {}
        for lora in self.get_all():
            for tag in lora.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1
        return [
            {"name": k, "count": v}
            for k, v in sorted(tag_counts.items(), key=lambda x: -x[1])
        ]

    def to_dict(self, lora: LoRAInfo) -> Dict:
        """Convert LoRAInfo to API-friendly dict."""
        return asdict(lora)


# Global singleton
lora_cache = LoRACache(ttl_seconds=300)
