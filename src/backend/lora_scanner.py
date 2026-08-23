"""
LoRA Scanner — Scans and caches LoRA model metadata.

Discovers LoRA files from ComfyUI model paths and /mnt/ssd/loras/,
extracts metadata from safetensors headers, enriches with registry
data (trigger words, strength, source URLs), and caches results.
"""

import json
import logging
import os
import struct
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

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

# Registry file with usage metadata (trigger words, strengths, source URLs)
REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent.parent / "docs" / "lora_registry.yaml"
)


@dataclass
class LoRARegistry:
    """Usage metadata from the LoRA registry for a single LoRA."""

    trigger_words: List[str] = field(default_factory=list)
    trigger_mode: str = ""  # "none" | "required" | "natural_language"
    trigger_format: str = ""  # Pattern for structured triggers
    trigger_examples: List[str] = field(default_factory=list)
    recommended_strength: float = 1.0
    strength_range: List[float] = field(default_factory=lambda: [0.5, 1.2])
    source_url: Optional[str] = None
    civitai_model_id: Optional[int] = None
    display_name: str = ""
    creator: str = ""
    version: str = ""
    usage_notes: str = ""
    noise_type: str = ""  # "single" | "dual"
    paired_with: Optional[str] = None
    modes: List[str] = field(default_factory=list)
    base_model: str = ""  # e.g., "wan2.2", "ltx", "sdxl"
    last_checked: str = ""


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
    registry: Optional[LoRARegistry] = None  # Enriched metadata from registry


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
    import re

    # Strip extension before analysis
    stem = filename.rsplit(".", 1)[0]
    stem_lower = stem.lower()
    if any(
        x in stem_lower for x in ["highnoise", "high_noise", "high noise", "_hn", "-hn"]
    ):
        return "high"
    if any(
        x in stem_lower for x in ["lownoise", "low_noise", "low noise", "_ln", "-ln"]
    ):
        return "low"
    # Split on delimiters AND CamelCase boundaries (split BEFORE lowering)
    expanded = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", stem)
    tokens = [t.lower() for t in re.split(r"[-_\s]+", expanded) if t]
    if "high" in tokens:
        return "high"
    if "low" in tokens:
        return "low"
    # Check trailing H/L suffixes (e.g., 5750H, 5750L)
    if re.search(r"\d+h$", stem_lower):
        return "high"
    if re.search(r"\d+l$", stem_lower):
        return "low"
    return ""


def _derive_base_model(filename: str) -> str:
    """Derive base model from filename or path."""
    lower = filename.lower()
    # Check full path including subdirectories
    if "wan" in lower or "w22" in lower:
        return "wan2.2"
    if "ltx" in lower:
        return "ltx"
    # MiniMax-H3 — check subdirectory and name markers before the generic
    # i2v/t2v → wan2.2 fallback (an H3 LoRA can carry i2v/t2v in its name).
    if lower.startswith("minimax-h3/") or lower.startswith("minimax-h3\\"):
        return "minimax_h3"
    if "minimax" in lower or "fl2va" in lower:
        return "minimax_h3"
    # I2V/T2V without ltx → wan2.2 (only Wan uses these LoRA modes)
    if ("i2v" in lower or "t2v" in lower) and "ltx" not in lower:
        return "wan2.2"
    if "sdxl" in lower or "xl" in lower:
        return "sdxl"
    if "pony" in lower or lower.endswith("-pn.safetensors") or "_pn." in lower:
        return "pony"
    if "sd15" in lower or "sd1.5" in lower:
        return "sd1.5"
    # Subdirectory-based detection
    if lower.startswith("qwen_image/") or lower.startswith("qwen_image\\"):
        return "qwen_image_edit"
    if lower.startswith("ltx/") or lower.startswith("ltx\\"):
        return "ltx"
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


def _load_registry() -> Dict[str, LoRARegistry]:
    """Load the LoRA registry YAML and return a dict keyed by filename."""
    if not REGISTRY_PATH.exists():
        logger.warning(f"⚠️ LoRA registry not found at {REGISTRY_PATH}")
        return {}
    try:
        with open(REGISTRY_PATH, "r") as f:
            entries = yaml.safe_load(f)
        if not entries or not isinstance(entries, list):
            return {}

        registry: Dict[str, LoRARegistry] = {}
        for entry in entries:
            fn = entry.get("filename", "")
            if not fn:
                continue
            registry[fn] = LoRARegistry(
                trigger_words=entry.get("trigger_words", []),
                trigger_mode=entry.get("trigger_mode", "none"),
                trigger_format=entry.get("trigger_format", ""),
                trigger_examples=entry.get("trigger_examples", []),
                recommended_strength=float(entry.get("recommended_strength", 1.0)),
                strength_range=entry.get("strength_range", [0.5, 1.2]),
                source_url=entry.get("source_url"),
                civitai_model_id=entry.get("civitai_model_id"),
                display_name=entry.get("display_name", ""),
                creator=entry.get("creator", ""),
                version=entry.get("version", ""),
                usage_notes=entry.get("usage_notes", "").strip(),
                noise_type=entry.get("noise_type", ""),
                paired_with=entry.get("paired_with"),
                modes=entry.get("modes", []),
                base_model=entry.get("base_model", ""),
                last_checked=entry.get("last_checked", ""),
            )
        debug_log(f"Loaded {len(registry)} entries from LoRA registry")
        return registry
    except Exception as e:
        logger.error(f"❌ Failed to load LoRA registry: {e}")
        return {}


def _enrich_with_registry(lora: LoRAInfo, registry: Dict[str, LoRARegistry]) -> None:
    """Enrich a LoRAInfo with registry metadata (in-place)."""
    # Try exact path match first, then filename-only match
    reg = registry.get(lora.path) or registry.get(lora.filename)
    if reg:
        lora.registry = reg
        # Override display name from registry if available
        if reg.display_name:
            lora.name = reg.display_name
        # Override base_model from registry if scanner couldn't detect it
        if not lora.base_model and reg.base_model:
            lora.base_model = reg.base_model
        # Override noise_level from registry noise_type if scanner missed it
        if (
            not lora.noise_level
            and reg.noise_type
            and reg.noise_type in ("high", "low")
        ):
            lora.noise_level = reg.noise_type


@dataclass
class LoRAValidation:
    """Result of validating a LoRA configuration against its registry entry."""

    lora_filename: str
    is_valid: bool
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)


def validate_lora_usage(
    lora_filename: str,
    positive_prompt: str,
    strength: float,
    registry: Optional[Dict[str, LoRARegistry]] = None,
) -> LoRAValidation:
    """Validate that a LoRA is being used correctly.

    Checks:
    - Required trigger words are in the positive prompt
    - Strength is within recommended range
    - Dual-noise pair is being used if applicable
    """
    result = LoRAValidation(lora_filename=lora_filename, is_valid=True)

    if registry is None:
        registry = _load_registry()

    reg = registry.get(lora_filename)
    if not reg:
        result.warnings.append(f"No registry entry found for '{lora_filename}'")
        return result

    prompt_lower = positive_prompt.lower()

    # Check trigger words
    if reg.trigger_mode == "required" and reg.trigger_words:
        found = any(tw.lower() in prompt_lower for tw in reg.trigger_words)
        if not found:
            result.is_valid = False
            result.errors.append(
                f"Missing required trigger word(s): {reg.trigger_words}. "
                f"At least one must appear in the positive prompt."
            )
            if reg.trigger_format:
                result.suggestions.append(f"Prompt format: {reg.trigger_format}")
            elif reg.trigger_examples:
                result.suggestions.append(f"Example: {reg.trigger_examples[0]}")

    elif reg.trigger_mode == "natural_language":
        # Can't enforce, but suggest
        if reg.trigger_examples:
            result.suggestions.append(
                f"Natural language trigger — describe the action. "
                f"Example: {reg.trigger_examples[0]}"
            )

    # Check strength range
    if reg.strength_range and len(reg.strength_range) == 2:
        lo, hi = reg.strength_range
        if strength < lo or strength > hi:
            result.warnings.append(
                f"Strength {strength} is outside recommended range "
                f"[{lo}, {hi}]. Recommended: {reg.recommended_strength}"
            )

    # Check dual-noise pair
    if reg.noise_type == "dual" and reg.paired_with:
        result.suggestions.append(
            f"Dual-noise LoRA — should be used together with '{reg.paired_with}'"
        )

    return result


def validate_lora_batch(
    loras: List[Dict[str, Any]],
    positive_prompt: str,
) -> List[LoRAValidation]:
    """Validate multiple LoRA configs at once.

    Each lora dict should have 'filename' and 'strength' keys.
    """
    registry = _load_registry()
    results = []
    for lora_cfg in loras:
        fn = lora_cfg.get("filename", "")
        strength = float(lora_cfg.get("strength", 1.0))
        results.append(validate_lora_usage(fn, positive_prompt, strength, registry))
    return results


def scan_lora_directory(
    lora_dir: Path,
    registry: Optional[Dict[str, LoRARegistry]] = None,
) -> List[LoRAInfo]:
    """Scan a single LoRA directory for models."""
    results = []
    if not lora_dir.exists():
        return results

    if registry is None:
        registry = _load_registry()

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
                base_model=_derive_base_model(rel_path),
                noise_level=_derive_noise_level(filepath.name),
                format=meta.get("format", ""),
                rank=meta.get("rank", ""),
            )
            results.append(info)
            _enrich_with_registry(info, registry)
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
        registry = _load_registry()
        all_loras = []
        for lora_dir in LORA_DIRS:
            loras = scan_lora_directory(lora_dir, registry)
            all_loras.extend(loras)
            debug_log(f"Found {len(loras)} LoRAs in {lora_dir}")

        # Second pass: inherit base_model/modes from paired_with references
        # If LoRA A has paired_with=B, then B should inherit A's base_model and modes
        by_filename = {l.filename: l for l in all_loras}
        by_path = {l.path: l for l in all_loras}
        for lora in all_loras:
            if lora.registry and lora.registry.paired_with:
                pair = by_filename.get(lora.registry.paired_with) or by_path.get(
                    lora.registry.paired_with
                )
                if pair and not pair.base_model:
                    source_base = lora.base_model or (
                        lora.registry.base_model if lora.registry else ""
                    )
                    if source_base:
                        pair.base_model = source_base
                        debug_log(
                            f"Inherited base_model '{source_base}' for {pair.path} from pair {lora.path}"
                        )

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
        d = asdict(lora)
        # Flatten registry into top-level keys for API convenience
        reg = d.pop("registry", None)
        if reg:
            d["trigger_words"] = reg.get("trigger_words", [])
            d["trigger_mode"] = reg.get("trigger_mode", "none")
            d["trigger_format"] = reg.get("trigger_format", "")
            d["trigger_examples"] = reg.get("trigger_examples", [])
            d["recommended_strength"] = reg.get("recommended_strength", 1.0)
            d["strength_range"] = reg.get("strength_range", [])
            d["source_url"] = reg.get("source_url")
            d["civitai_model_id"] = reg.get("civitai_model_id")
            d["creator"] = reg.get("creator", "")
            d["version"] = reg.get("version", "")
            d["usage_notes"] = reg.get("usage_notes", "")
            d["noise_type"] = reg.get("noise_type", "")
            d["paired_with"] = reg.get("paired_with")
            d["modes"] = reg.get("modes", [])
            d["has_registry"] = True
        else:
            d["has_registry"] = False
        return d


# Global singleton
lora_cache = LoRACache(ttl_seconds=300)
