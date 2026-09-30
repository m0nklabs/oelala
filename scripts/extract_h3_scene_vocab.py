#!/usr/bin/env python3
"""Extract the MiniMax-H3 scene vocabulary from the Ref2Vid randomizer page.

The page is a single-file studio that composes timed shots for H3 (who does
what, camera, sound). Its value for our prompt generator is the structured
vocabulary plus the stage/camera/audio knowledge it encodes, so this script
turns that page into committed JSON the backend can use at runtime.

Usage:
    scripts/extract_h3_scene_vocab.py --source "/path/to/h3_scene_randomizer.html"
    scripts/extract_h3_scene_vocab.py --source ... --out src/backend/h3_scene_vocab.json

The generated file is committed, so the runtime never depends on the page.
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "src" / "backend" / "h3_scene_vocab.json"

# Dimensions we keep: value -> human label. Everything else in the page is UI.
DIMENSIONS = {
    "cast": "cast",
    "relate": "relationship",
    "place": "place",
    "placeLight": "lighting",
    "clothes": "her clothes",
    "manCloth": "his clothes",
    "mood": "mood",
    "stylePack": "style pack",
    "sceneFormat": "scene format",
    "artifactFix": "artifact fix",
    "talkTrack": "dialogue",
    "cutMode": "cut mode",
    "introDo": "intro action",
    "teaseDo": "tease action",
    "hjPos": "handjob position",
    "hjGrip": "handjob grip",
    "bjPos": "blowjob position",
    "bjDepth": "blowjob depth",
    "brKind": "breast action",
    "sxPos": "sex position",
    "sxKind": "sex kind",
    "sxBounce": "bounce",
    "cumWhere": "cum target",
    "cuFinish": "cum finish",
    "cuQty": "cum amount",
    "distAll": "camera distance",
    "heightAll": "camera height",
    "inAngle": "camera angle",
    "inMove": "camera move",
    "inSpeed": "camera speed",
    "inFace": "face",
    "inLook": "look",
    "inLH": "left hand",
    "inRH": "right hand",
    "fAge": "her age",
    "fFrame": "her frame",
    "fHeight": "her height",
    "fBreast": "her breasts",
    "fAss": "her ass",
    "manAge": "his age",
    "manBody": "his body",
    "manLook": "his look",
    "manMood": "his mood",
    "manSkin": "his skin",
    "manLen": "his length",
    "manGirth": "his girth",
}

# Stages of the act arc, in order, with the camera/audio knowledge the page
# encodes per stage.
STAGE_ORDER = ["intro", "tease", "hj", "bj", "breast", "sex", "cum"]


def strip_tags(raw: str) -> str:
    return html_mod.unescape(re.sub(r"<[^>]+>", " ", raw)).strip()


def parse_selects(source: str) -> dict[str, list[dict[str, str]]]:
    """Return {select_id: [{value, label}, ...]} for every <select> in the page."""
    out: dict[str, list[dict[str, str]]] = {}
    for match in re.finditer(r'<select id="([A-Za-z]+)"[^>]*>(.*?)</select>', source, re.S):
        select_id, body = match.group(1), match.group(2)
        options = []
        for opt in re.finditer(r'<option value="([^"]*)"[^>]*>(.*?)</option>', body, re.S):
            value, label = opt.group(1), strip_tags(opt.group(2))
            if value:
                options.append({"value": value, "label": label})
        if options:
            out[select_id] = options
    return out


def js_object_body(source: str, name: str) -> str:
    """Return the body of `const NAME = { ... };` using brace matching.

    Regex alone breaks on single-line nested objects, and the page mixes both
    styles, so the block is located by counting braces.
    """
    marker = f"const {name} = {{"
    start = source.find(marker)
    if start < 0:
        return ""
    open_index = source.index("{", start)
    depth = 0
    for index in range(open_index, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[open_index + 1 : index]
    return ""


def iter_nested_objects(body: str) -> list[tuple[str, str]]:
    """Yield (key, body) for each `key: { ... }` entry in *body*."""
    entries: list[tuple[str, str]] = []
    for match in re.finditer(r'"?([\w-]+)"?\s*:\s*\{', body):
        key = match.group(1)
        open_index = body.index("{", match.start())
        depth = 0
        for index in range(open_index, len(body)):
            char = body[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    entries.append((key, body[open_index + 1 : index]))
                    break
    return entries


def _fields(body: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for entry in re.finditer(r'(\w+)\s*:\s*"((?:[^"\\]|\\.)*)"', body, re.S):
        fields[entry.group(1)] = entry.group(2).replace('\\"', '"').strip()
    return fields


def parse_js_object(source: str, name: str) -> dict[str, str]:
    """Parse a flat `const NAME = { key: "string", ... };` block."""
    body = js_object_body(source, name)
    return _fields(body) if body else {}


def parse_styles(source: str) -> dict[str, dict[str, str]]:
    """Parse the STYLE pack table (label/camera/audio per pack)."""
    body = js_object_body(source, "STYLE")
    styles: dict[str, dict[str, str]] = {}
    for key, entry_body in iter_nested_objects(body):
        entry = _fields(entry_body)
        if entry:
            styles[key] = entry
    return styles


def parse_formats(source: str) -> dict[str, dict[str, str]]:
    """Parse the FORMAT (story wrapper) table."""
    body = js_object_body(source, "FORMAT")
    formats: dict[str, dict[str, str]] = {}
    for key, entry_body in iter_nested_objects(body):
        entry = _fields(entry_body)
        if entry:
            formats[key] = entry
    return formats


def build(source_path: Path) -> dict:
    source = source_path.read_text(encoding="utf-8", errors="replace")
    selects = parse_selects(source)

    vocab = {}
    for select_id, dimension in DIMENSIONS.items():
        options = selects.get(select_id, [])
        if options:
            vocab[select_id] = {"dimension": dimension, "options": options}

    audio_stage = parse_js_object(source, "AUDIO_STAGE")
    styles = parse_styles(source)
    formats = parse_formats(source)

    artifact_match = re.search(
        r'const light = "(ARTIFACT LOCK:.*?)";\s*\n\s*if\(cfg\.artifactFix==="light"\) return light;\s*\n\s*return light \+ "((?:[^"\\]|\\.)*)";',
        source,
        re.S,
    )
    artifact = {}
    if artifact_match:
        artifact = {
            "light": artifact_match.group(1).replace('\\"', '"').strip(),
            "full_extra": artifact_match.group(2).replace('\\"', '"').strip(),
        }

    return {
        "meta": {
            "source": source_path.name,
            "generated_by": "scripts/extract_h3_scene_vocab.py",
            "purpose": (
                "Structured scene vocabulary and H3 stage/audio/camera knowledge, "
                "used by the prompt generator to write concrete, varied scenes."
            ),
            "dimensions": len(vocab),
            "options": sum(len(v["options"]) for v in vocab.values()),
        },
        "vocab": vocab,
        "stage_order": STAGE_ORDER,
        "audio_stage": audio_stage,
        "styles": styles,
        "formats": formats,
        "artifact_lock": artifact,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="path to the randomizer HTML page")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output JSON path")
    parser.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = parser.parse_args()

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"❌ source not found: {source_path}")
        return 1

    data = build(source_path)
    meta = data["meta"]
    print(f"📚 {meta['dimensions']} dimensions, {meta['options']} options")
    print(f"   audio stages: {len(data['audio_stage'])} | styles: {len(data['styles'])} | formats: {len(data['formats'])}")
    print(f"   artifact lock: {'yes' if data['artifact_lock'] else 'no'}")
    if args.dry_run:
        return 0

    out_path = Path(args.out)
    out_path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"✅ wrote {out_path.relative_to(PROJECT_ROOT) if out_path.is_relative_to(PROJECT_ROOT) else out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
