#!/usr/bin/env python3
"""Build/verify docs/model_catalog.yaml — the model-component source of truth.

The catalog unifies what ComfyUI can load across every compute location:

  * worker registries parsed from deploy/runpod*/handler.py (AST, no imports —
    the worker modules need packages this box does not have),
  * checkpoint variants and turbo presets from src/backend/comfyui_client.py,
  * curated LoRA HF sources from src/backend/app.py,
  * the local ComfyUI model tree (root resolved via systemd, never hardcoded),
  * the HF mirrors (read-only listings: private m0nk111/oelala-models and
    m0nk111/oelala-loras, public Serenak/chilloutmix),
  * the Windows-PC ComfyUI (availability probe through the compute backend
    inventory in src/backend/generation/compute_backends.json; short timeout,
    graceful "unverified" when the box is asleep).

Outputs:
  docs/model_catalog.yaml  — machine-readable catalog (fully generated)
  docs/MODEL_CATALOG.md    — human-readable tables per family

`--check` regenerates the YAML and exits non-zero when it differs from the
committed file, so drift in registries/disk/mirrors is caught before it lies.

Usage:
  python scripts/build_model_catalog.py            # regenerate both outputs
  python scripts/build_model_catalog.py --check    # verify only, exit code
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_YAML = REPO_ROOT / "docs" / "model_catalog.yaml"
CATALOG_MD = REPO_ROOT / "docs" / "MODEL_CATALOG.md"
COMPUTE_BACKENDS = REPO_ROOT / "src" / "backend" / "generation" / "compute_backends.json"

ENV_FILE = REPO_ROOT / ".env"
HF_PRIVATE_MODELS_REPO = "m0nk111/oelala-models"
HF_PRIVATE_LORAS_REPO = "m0nk111/oelala-loras"
HF_PUBLIC_LORA_DUMP_REPO = "Serenak/chilloutmix"

# ComfyUI models subdirectory -> catalog role. Unknown directories fall back
# to "other"; role is derived from where ComfyUI looks, not from guesswork.
ROLE_BY_DIR = {
    "diffusion_models": "diffusion_model",
    "checkpoints": "checkpoint",
    "text_encoders": "text_encoder",
    "vae": "vae",
    "loras": "lora",
    "upscale_models": "upscaler",
    "controlnet": "controlnet",
    "audio_encoders": "audio",
    "unet": "diffusion_model",
}

# filename token -> quantization label, checked in order (first hit wins).
QUANT_TOKENS = [
    ("nvfp4_awq", "nvfp4_awq"),
    ("nvfp4", "nvfp4"),
    ("fp8_scaled", "fp8_scaled"),
    ("fp8mixed", "fp8_mixed"),
    ("fp8_e4m3fn", "fp8_e4m3fn"),
    ("e4m3fn", "fp8_e4m3fn"),
    ("fp8", "fp8"),
    ("int8_convrot", "int8_convrot"),
    ("int8", "int8"),
    ("w4a8", "w4a8"),
    ("bf16", "bf16"),
    ("fp16", "fp16"),
    ("fp32", "fp32"),
    ("q4_0", "q4_0"),
]

# (regex on filename, family, base_model). First match wins; anything else is
# reported with family "unknown" instead of a guess.
FAMILY_PATTERNS = [
    (r"(?i)minimax[-_]h3|h3erosmax|dasiwaminimaxh3", "minimax_h3", "MiniMax-H3"),
    (r"(?i)wan2\.?2|^wan2_2|^wan2\.2_|wan_2\.1|umt5_xxl_fp16|clip_vision_h", "wan22", "Wan 2.2"),
    (r"(?i)^ltx|ltx2|ltx-|gemma_3_12b", "ltx23", "LTX-2.3"),
    (r"(?i)qwen_image|qwen_2\.5_vl", "qwen_image_edit", "Qwen-Image-Edit 2511"),
    (r"(?i)^ae\.safetensors|flux2", "flux2", "FLUX.2"),
    (r"(?i)flux(?!2)", "flux", "FLUX"),
    (r"(?i)krea2", "krea2", "Krea 2"),
    (r"(?i)sdxl", "sdxl", "SDXL"),
    (r"(?i)wan2\.?1|wanvideo", "wan21", "Wan 2.1"),
]

# Worker handler file -> worker label used in available_on / used_by fields.
WORKER_REGISTRIES = [
    ("deploy/runpod-i2i/handler.py", "CLOUD_I2I_MODELS", "runpod_i2i_worker"),
    ("deploy/runpod-ltx23/handler.py", "LTX23_MODELS", "runpod_ltx23_worker"),
    ("deploy/runpod-minimax-h3/handler.py", "MINIMAX_H3_MODELS", "runpod_h3_worker"),
]


# ─── AST extraction (no imports of the parsed modules) ───────────────────────

def _collect_name_literals(ast_tree: ast.AST) -> Dict[str, Any]:
    """Collect simple top-level literal constants (strings/ints) for Name resolution."""
    known: Dict[str, Any] = {}
    for node in ast_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            continue
        for name in targets:
            known[name] = value
    return known


def _eval_with_knowns(node: ast.AST, known: Dict[str, Any]) -> Optional[Any]:
    """literal_eval that resolves known Name references (repo constants etc.)."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        pass
    if isinstance(node, ast.Name) and node.id in known:
        return known[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        out = []
        for el in node.elts:
            v = _eval_with_knowns(el, known)
            if v is None:
                return None
            out.append(v)
        return out
    if isinstance(node, ast.Dict):
        out = {}
        for k, v in zip(node.keys, node.values):
            key = ast.literal_eval(k) if k is not None else None
            val = _eval_with_knowns(v, known)
            if key is None or val is None:
                return None
            out[key] = val
        return out
    return None


def extract_assignment(ast_tree: ast.AST, var_name: str) -> Optional[Any]:
    """Return the value of a top-level `VAR = <expr>` when it resolves to data."""
    known = _collect_name_literals(ast_tree)
    for node in ast_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(t, ast.Name) and t.id == var_name for t in node.targets):
            return _eval_with_knowns(node.value, known)
    return None


def parse_module(path: Path, var_names: List[str]) -> Dict[str, Any]:
    """Parse top-level literal assignments without importing the module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {name: extract_assignment(tree, name) for name in var_names}


def parse_env_names(source: str) -> List[str]:
    """Extract os.getenv("NAME", ...) default names used for mirror config."""
    return sorted(set(re.findall(r'os\.getenv\(\s*"([A-Z0-9_]+)"', source)))


# ─── Ground-truth scans ──────────────────────────────────────────────────────

def resolve_local_comfy_root() -> Path:
    """Find the local ComfyUI root from systemd; env override wins."""
    override = os.environ.get("LOCAL_COMFYUI_DIR")
    if override:
        return Path(override)
    try:
        out = subprocess.run(
            ["systemctl", "show", "comfyui", "-p", "WorkingDirectory", "--value"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        out = ""
    if out:
        return Path(out)
    raise SystemExit(
        "Cannot resolve the local ComfyUI root: systemctl has no comfyui "
        "WorkingDirectory and LOCAL_COMFYUI_DIR is not set."
    )


def scan_extra_model_roots(comfy_root: Path) -> Dict[str, Dict[str, Any]]:
    """Scan the roots ComfyUI adds via extra_model_paths.yaml (SSD, nvme, ...)."""
    found: Dict[str, Dict[str, Any]] = {}
    cfg = comfy_root / "extra_model_paths.yaml"
    if not cfg.exists():
        return found
    try:
        import yaml as _yaml

        doc = _yaml.safe_load(cfg.read_text()) or {}
    except (OSError, _yaml.YAMLError):
        return found
    for group in doc.values():
        if not isinstance(group, dict):
            continue
        base = group.get("base_path")
        if not base:
            continue
        base = Path(os.path.expandvars(str(base)))
        if not base.is_dir():
            continue
        for subdir, rel in group.items():
            if subdir == "base_path" or not isinstance(rel, str):
                continue
            root = base / rel
            if not root.is_dir():
                continue
            for f in sorted(root.rglob("*")):
                if f.is_file() and f.suffix in (".safetensors", ".gguf", ".pt", ".pth", ".bin"):
                    rel_key = f"{subdir}/{f.name}"
                    found.setdefault(rel_key, {"size": f.stat().st_size, "root": str(base.name)})
    return found


def scan_local_models(comfy_root: Path) -> Dict[str, Dict[str, Any]]:
    """Scan <comfy_root>/models/** for files; key = '<subdir>/<filename>'."""
    models_root = comfy_root / "models"
    found: Dict[str, Dict[str, Any]] = {}
    if not models_root.is_dir():
        return found
    for subdir in sorted(p.name for p in models_root.iterdir() if p.is_dir()):
        for f in sorted((models_root / subdir).rglob("*")):
            if f.is_file() and f.suffix in (".safetensors", ".gguf", ".pt", ".pth", ".bin"):
                rel = f"{subdir}/{f.name}"
                try:
                    size = f.stat().st_size
                except OSError:
                    size = None
                found[rel] = {"size": size}
    return found


def load_env_token() -> Optional[str]:
    """Read HF_LORA_TOKEN from .env for read-only HF listings (never printed)."""
    if not ENV_FILE.exists():
        return None
    for line in ENV_FILE.read_text().splitlines():
        if line.startswith("HF_LORA_TOKEN="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def list_hf_repo(repo: str, token: Optional[str]) -> Dict[str, int]:
    """Read-only file listing (path -> bytes) of a HF model repo."""
    from huggingface_hub import HfApi  # lazy: keeps tests import-light

    api = HfApi(token=token)
    info = api.repo_info(repo, repo_type="model", files_metadata=True)
    return {s.rfilename: (s.size or 0) for s in info.siblings if (s.size or 0) > 0}


def probe_remote_models(base_url: str, dirs: List[str], timeout_s: float = 5.0) -> Dict[str, bool]:
    """Probe a ComfyUI server's /models/<dir> listings; {} when unreachable.

    One retry on connection errors: a sleeping/waking box often drops the
    first contact.
    """
    import requests  # lazy
    import time

    result: Dict[str, bool] = {}
    url_base = base_url.rstrip("/")
    for d in dirs:
        for attempt in (1, 2):
            try:
                r = requests.get(f"{url_base}/models/{d}", timeout=timeout_s)
                if r.status_code == 200:
                    names = r.json()
                    result[d] = bool(names) and isinstance(names, list)
                else:
                    result[d] = False
                break
            except requests.RequestException:
                if attempt == 2:
                    return {}
                time.sleep(3)
    return result


def load_backends() -> List[Dict[str, Any]]:
    if not COMPUTE_BACKENDS.exists():
        return []
    return json.loads(COMPUTE_BACKENDS.read_text()).get("backends", [])


# ─── Catalog assembly ────────────────────────────────────────────────────────

def classify(rel_key: str) -> Tuple[str, str, str]:
    """(target_dir, role, (family, base_model)) from '<subdir>/<filename>'."""
    subdir, filename = rel_key.split("/", 1)
    role = ROLE_BY_DIR.get(subdir, "other")
    family, base_model = "unknown", "unknown"
    for pattern, fam, base in FAMILY_PATTERNS:
        if re.search(pattern, filename):
            family, base_model = fam, base
            break
    return subdir, role, family, base_model


def detect_quantization(filename: str, description: str) -> str:
    for token, label in QUANT_TOKENS:
        if token in filename.lower():
            return label
    for token, label in QUANT_TOKENS:
        if token in description.lower():
            return label
    return "unknown"


def load_lora_usage_registry() -> Dict[str, Dict[str, Any]]:
    """Read docs/lora_registry.yaml (curated usage metadata) keyed by filename."""
    path = REPO_ROOT / "docs" / "lora_registry.yaml"
    if not path.exists():
        return {}
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    except yaml.YAMLError:
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for item in doc:
        if isinstance(item, dict) and item.get("filename"):
            out[str(item["filename"])] = item
    return out


def civitai_ids_from_url(url: str) -> Tuple[Optional[str], Optional[str]]:
    """Extract (civitai_model_version_url, file_id) from a registry download URL."""
    m_file = re.search(r"fileId=(\d+)", url)
    m_ver = re.search(r"/api/download/models/(\d+)", url)
    return (m_ver.group(1) if m_ver else None), (m_file.group(1) if m_file else None)


def build_catalog() -> Dict[str, Any]:
    registry_entries: List[Dict[str, Any]] = []
    registry_src: Dict[str, Dict[str, Any]] = {}

    worker_labels: Dict[str, List[str]] = defaultdict(list)  # filename -> labels
    for rel_path, var, label in WORKER_REGISTRIES:
        parsed = parse_module(REPO_ROOT / rel_path, [var])[var] or []
        for entry in parsed:
            registry_entries.append({**entry, "_worker": label, "_registry_file": rel_path})
            worker_labels[entry["filename"]].append(label)

    client = parse_module(
        REPO_ROOT / "src" / "backend" / "comfyui_client.py",
        ["MINIMAX_H3_VARIANTS", "MINIMAX_H3_TURBO_PRESETS"],
    )
    variants = client["MINIMAX_H3_VARIANTS"] or {}
    presets = client["MINIMAX_H3_TURBO_PRESETS"] or {}

    app_src = (REPO_ROOT / "src" / "backend" / "app.py").read_text(encoding="utf-8")
    lora_sources = parse_module(REPO_ROOT / "src" / "backend" / "app.py", ["LORA_HF_SOURCES"])[
        "LORA_HF_SOURCES"
    ] or {}
    utils_src = (
        (REPO_ROOT / "src" / "backend" / "generation" / "lora_utils.py").read_text(encoding="utf-8")
    )
    mirror_envs = sorted(
        {n for src in (app_src, utils_src) for n in parse_env_names(src)
         if n.startswith("LORA_HF") or n == "HF_LORA_TOKEN"}
    )

    # Ground truth scans (read-only).
    comfy_root = resolve_local_comfy_root()
    local_files = scan_local_models(comfy_root)
    local_files.update(scan_extra_model_roots(comfy_root))

    token = load_env_token()
    hf_private_models = list_hf_repo(HF_PRIVATE_MODELS_REPO, token)
    hf_private_loras = list_hf_repo(HF_PRIVATE_LORAS_REPO, token)
    hf_public_dump = list_hf_repo(HF_PUBLIC_LORA_DUMP_REPO, token)

    lora_usage = load_lora_usage_registry()

    backends = load_backends()
    windows_backend = next(
        (b for b in backends if b.get("type") == "comfyui" and "windows" in b.get("id", "").lower()),
        None,
    )
    probe_dirs = [
        "diffusion_models", "checkpoints", "text_encoders", "vae", "loras",
        "upscale_models", "controlnet", "clip_vision", "unet",
    ]
    windows_probe = (
        probe_remote_models(windows_backend["base_url"], probe_dirs)
        if windows_backend and windows_backend.get("enabled")
        else {}
    )
    windows_verified = bool(windows_probe)

    # Union of every known component, keyed '<target_dir>/<filename>'.
    keys: Dict[str, Dict[str, Any]] = {}

    def ensure(rel_key: str) -> Dict[str, Any]:
        if rel_key not in keys:
            subdir, role, family, base = classify(rel_key)
            keys[rel_key] = {
                "filename": rel_key.split("/", 1)[1],
                "target_dir": subdir,
                "role": role,
                "family": family,
                "base_model": base,
                "quantization": "unknown",
                "size_gb": None,
                "sources": {},
                "available_on": {},
                "used_by": [],
                "notes": [],
            }
        return keys[rel_key]

    for entry in registry_entries:
        target_dir = entry.get("target_dir") or entry.get("local_dir") or "unknown"
        entry = {**entry, "target_dir": target_dir}
        rel_key = f"{target_dir}/{entry['filename']}"
        item = ensure(rel_key)
        item["quantization"] = detect_quantization(entry["filename"], entry.get("description", ""))
        if entry.get("size_gb"):
            item["size_gb"] = round(float(entry["size_gb"]), 2)
        if entry.get("description"):
            item["notes"].append(entry["description"])
        sources = item["sources"]
        repo = entry.get("repo") or entry.get("hf_repo")
        if repo:
            sources["hf_upstream"] = {"repo": repo, "path": entry["hf_path"]}
        if entry.get("url"):
            ver, file_id = civitai_ids_from_url(entry["url"])
            if ver:
                sources["civitai"] = {"version_id": ver, "file_id": file_id}
        if entry.get("repo") == HF_PRIVATE_MODELS_REPO:
            sources["private_mirror"] = {
                "repo": HF_PRIVATE_MODELS_REPO, "path": entry["hf_path"],
            }
        delivery = "on_demand_download"
        if entry.get("startup_required"):
            delivery = "startup_required"
        elif entry.get("persist_on_volume"):
            delivery = "persisted_on_worker_volume"
        item["available_on"][entry["_worker"]] = delivery
        item["used_by"].append(f"{entry['_worker']} registry ({entry['_registry_file']})")

    for rel_key, info in local_files.items():
        item = ensure(rel_key)
        item.setdefault("_local_size", info["size"])
        if item["size_gb"] is None and info["size"]:
            item["size_gb"] = round(info["size"] / 1e9, 2)
        item["available_on"]["local_comfyui_ai_kvm2"] = True
        if item["quantization"] == "unknown":
            item["quantization"] = detect_quantization(item["filename"], "")

    for rel_path in hf_private_models:
        rel_key = f"diffusion_models/{rel_path.split('/', 1)[-1]}" if rel_path.startswith("diffusion_models/") else rel_path
        if "/" not in rel_path or rel_path.count("/") != 1:
            continue
        item = ensure(rel_key)
        item["sources"].setdefault(
            "private_mirror", {"repo": HF_PRIVATE_MODELS_REPO, "path": rel_path}
        )
        if item["size_gb"] is None and hf_private_models[rel_path]:
            item["size_gb"] = round(hf_private_models[rel_path] / 1e9, 2)

    for rel_path, size in hf_private_loras.items():
        if not rel_path.endswith(".safetensors") or rel_path.count("/") > 1:
            continue
        item = ensure(rel_path)
        item["sources"].setdefault(
            "private_mirror", {"repo": HF_PRIVATE_LORAS_REPO, "path": rel_path}
        )
        if item["size_gb"] is None and size:
            item["size_gb"] = round(size / 1e9, 2)

    dump_by_basename = {p.split("/")[-1]: p for p in hf_public_dump}
    for rel_key in list(keys):
        item = keys[rel_key]
        base = dump_by_basename.get(item["filename"])
        if base:
            item["sources"].setdefault(
                "public_dump", {"repo": HF_PUBLIC_LORA_DUMP_REPO, "path": base}
            )

    for rel_key, item in keys.items():
        # Windows availability (probe results apply per target dir).
        if windows_verified:
            item["available_on"]["windows_pc_comfyui"] = bool(
                windows_probe.get(item["target_dir"])
                and f"/models/{item['target_dir']}" is not None
            )
        else:
            item["available_on"]["windows_pc_comfyui"] = "unverified"
        # LoRA sources from the curated backend mapping.
        if item["role"] == "lora":
            if rel_key in lora_sources:
                item["sources"].setdefault("hf_curated", lora_sources[rel_key])
            curated = lora_usage.get(item["filename"])
            if curated:
                if item["family"] == "unknown" and curated.get("base_model"):
                    item["family"] = str(curated["base_model"]).replace(".", "_")
                    item["base_model"] = str(curated["base_model"])
                if curated.get("civitai_model_id"):
                    item["sources"].setdefault(
                        "civitai", {"model_id": str(curated["civitai_model_id"])}
                    )
                if curated.get("source_url"):
                    item["sources"].setdefault("civitai_page", {"url": curated["source_url"]})
                item["used_by"].append("curated usage metadata in docs/lora_registry.yaml")
            else:
                item["used_by"].append("generation API user loras (no usage metadata)")
        if item["family"] == "minimax_h3":
            for variant, spec in variants.items():
                if spec["checkpoint"] == item["filename"]:
                    kind = "turbo" if spec.get("turbo") else "non-turbo"
                    item["used_by"].append(
                        f"minimax_h3 adapters: model_variant={variant} ({kind})"
                    )
            for preset_name, spec in presets.items():
                if spec.get("lora") == item["filename"]:
                    item["used_by"].append(
                        f"minimax_h3 turbo preset '{preset_name}' "
                        f"({spec['steps']} steps, shift {spec.get('shift_video')}/{spec.get('shift_audio')})"
                    )
        # Mirror-aware availability markers.
        if any(s.get("repo") == HF_PRIVATE_MODELS_REPO for s in item["sources"].values()):
            item["available_on"]["hf_mirror_private"] = True
        if "public_dump" in item["sources"]:
            item["available_on"]["hf_public_dump"] = True

    entries = [keys[k] for k in sorted(keys)]
    for item in entries:
        if item["family"] == "unknown":
            item["notes"].append("family unverified (no registry source, no pattern match)")
        if windows_verified and item["available_on"]["windows_pc_comfyui"] is False:
            item["notes"].append("not listed on windows_pc_comfyui (probe)")
        if item["size_gb"] is None:
            item["notes"].append("size unverified")
        item["notes"] = item["notes"] or ["—"]

    return {
        "meta": {
            "generated_by": "scripts/build_model_catalog.py",
            "generator": "--check regenerates this file; hand-edits are drift",
            "local_comfyui_root_resolved": "systemctl show comfyui -p WorkingDirectory (or LOCAL_COMFYUI_DIR env)",
            "hf_private_models_repo": HF_PRIVATE_MODELS_REPO,
            "hf_private_loras_repo": HF_PRIVATE_LORAS_REPO,
            "hf_public_lora_dump_repo": HF_PUBLIC_LORA_DUMP_REPO,
            "mirror_env_names_from_app_py": mirror_envs,
            "windows_pc_probe": "verified" if windows_verified else "unreachable during generation (marked unverified)",
            "counts": {
                "entries": len(entries),
                "by_role": dict(sorted(_count(entries, "role").items())),
                "by_family": dict(sorted(_count(entries, "family").items())),
            },
        },
        "entries": entries,
    }


def _count(entries: List[Dict[str, Any]], field: str) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for e in entries:
        counts[e[field]] += 1
    return counts


# ─── Rendering ───────────────────────────────────────────────────────────────

def _slim(entry: Dict[str, Any]) -> Dict[str, Any]:
    """YAML-safe entry: drop internal keys, keep a stable field order."""
    return {
        "filename": entry["filename"],
        "role": entry["role"],
        "target_dir": entry["target_dir"],
        "family": entry["family"],
        "base_model": entry["base_model"],
        "quantization": entry["quantization"],
        "size_gb": entry["size_gb"],
        "sources": entry["sources"] or {"none": "local file only, upstream unknown"},
        "available_on": entry["available_on"],
        "used_by": entry["used_by"] or ["—"],
        "notes": entry["notes"],
    }


def render_marketing(catalog: Dict[str, Any]) -> str:
    meta = catalog["meta"]
    lines = [
        "# Model Catalog",
        "",
        "> Generated by `scripts/build_model_catalog.py` — do not hand-edit;",
        "> run the generator or `--check`. Local root resolved via systemd,",
        "> mirrors read via the HuggingFace API (read-only).",
        "",
        f"Entries: **{meta['counts']['entries']}** — "
        + ", ".join(f"{k}: {v}" for k, v in meta["counts"]["by_role"].items()),
        "",
        "Families: " + ", ".join(f"{k}: {v}" for k, v in meta["counts"]["by_family"].items()),
        "",
        f"Windows-PC probe during generation: {meta['windows_pc_probe']}.",
        "",
    ]
    by_family: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in catalog["entries"]:
        by_family[e["family"]].append(e)
    for family in sorted(by_family):
        lines += [f"## {family}", "", "| filename | role | dir | GB | quant | cloud | local | windows | mirror |", "|---|---|---|---|---|---|---|---|---|"]
        for e in sorted(by_family[family], key=lambda x: (x["role"], x["filename"])):
            cloud = next((v for k, v in e["available_on"].items() if k.startswith("runpod")), "—")
            local = "yes" if e["available_on"].get("local_comfyui_ai_kvm2") else "—"
            win = str(e["available_on"].get("windows_pc_comfyui", "—")).lower()
            mirror = "private" if e["available_on"].get("hf_mirror_private") else ("dump" if e["available_on"].get("hf_public_dump") else "—")
            size = f"{e['size_gb']:.2f}" if e["size_gb"] else "?"
            lines.append(
                f"| {e['filename']} | {e['role']} | {e['target_dir']} | {size} | {e['quantization']} | {cloud} | {local} | {win} | {mirror} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="verify only; exit 1 on drift")
    args = parser.parse_args()

    catalog = build_catalog()
    yaml_text = yaml.safe_dump(
        {"meta": catalog["meta"], "entries": [_slim(e) for e in catalog["entries"]]},
        sort_keys=False, allow_unicode=True, width=120,
    )

    if args.check:
        if not CATALOG_YAML.exists():
            print(f"❌ {CATALOG_YAML} does not exist — run the generator first")
            return 1
        committed = CATALOG_YAML.read_text(encoding="utf-8")
        if committed == yaml_text:
            print(f"✅ catalog up to date ({catalog['meta']['counts']['entries']} entries)")
            return 0
        print("❌ docs/model_catalog.yaml is stale — regenerate with scripts/build_model_catalog.py")
        import difflib
        for line in list(difflib.unified_diff(
                committed.splitlines(), yaml_text.splitlines(), "committed", "regenerated", lineterm=""))[:60]:
            print(line)
        return 1

    CATALOG_YAML.write_text(yaml_text, encoding="utf-8")
    CATALOG_MD.write_text(render_marketing(catalog), encoding="utf-8")
    print(f"✅ wrote {CATALOG_YAML.name} and {CATALOG_MD.name} "
          f"({catalog['meta']['counts']['entries']} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
