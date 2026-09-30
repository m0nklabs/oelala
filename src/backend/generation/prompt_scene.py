"""Structured scene rolling for MiniMax-H3 prompts.

The vocabulary and the H3-specific stage/audio knowledge come from the Ref2Vid
randomizer page via ``scripts/extract_h3_scene_vocab.py`` (committed as
``src/backend/h3_scene_vocab.json``). A rolled scene hands the prompt LLM
concrete details — place, light, clothes, act arc, per-shot camera — instead of
letting it invent generic filler, and carries the artifact guards that keep H3's
audio and anatomy from drifting.
"""

from __future__ import annotations

import json
import random
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

DATA_PATH = Path(__file__).resolve().parents[1] / "h3_scene_vocab.json"

# Stage arc used when the caller does not pass one: an act flows intro → tease →
# one main act → sex → cum, and short clips drop stages from the middle out.
# Which vocabulary dimensions describe each stage, in the order they read.
STAGE_ACT_FIELDS = {
    "intro": ("introDo",),
    "tease": ("teaseDo",),
    "hj": ("hjPos", "hjGrip"),
    "bj": ("bjPos", "bjDepth"),
    "breast": ("brKind",),
    "sex": ("sxPos", "sxKind"),
    "cum": ("cumWhere", "cuFinish"),
}

# Camera dimensions (the page repeats them per act with identical option sets).
CAMERA_FIELDS = {
    "distance": "inDist",
    "angle": "inAngle",
    "move": "inMove",
    "speed": "inSpeed",
}

# Values that mean "no choice" — never roll these.
_PLACEHOLDER_VALUES = {"any", "asref", "undef", "off", "random", "none", ""}


@lru_cache(maxsize=1)
def load_vocab() -> Dict[str, Any]:
    """Load the extracted scene vocabulary (cached)."""
    with open(DATA_PATH, encoding="utf-8") as handle:
        return json.load(handle)


def _options(vocab: Dict[str, Any], select_id: str) -> List[Dict[str, str]]:
    entry = vocab.get("vocab", {}).get(select_id)
    if not entry:
        return []
    return [
        option
        for option in entry["options"]
        if option["value"] not in _PLACEHOLDER_VALUES
    ]


def _pick(rng: random.Random, vocab: Dict[str, Any], select_id: str) -> Optional[str]:
    options = _options(vocab, select_id)
    return rng.choice(options)["label"] if options else None


def stage_arc(
    duration_s: float, main_act: Optional[str] = None, seed: Optional[int] = None
) -> List[str]:
    """Stages for a clip of *duration_s*: ~1.6 s per shot, 2-6 shots.

    "main" resolves to a concrete main act (handjob / blowjob / breast play) so
    every shot in the arc has real action vocabulary behind it.
    """
    shots = max(2, min(6, round(duration_s / 1.6) or 2))
    if main_act in (None, "main"):
        main_act = random.Random(seed).choice(["hj", "bj", "breast"])
    stages = ["intro", main_act, "sex", "cum"]
    if shots >= len(stages):
        return stages
    if shots == 2:
        return [stages[0], stages[-1]]
    if shots == 3:
        return [stages[0], stages[1], stages[-1]]
    return stages[-shots:]


def shortlist(
    vocab: Dict[str, Any], select_id: str, count: int = 10, seed: Optional[int] = None
) -> List[str]:
    """Return up to *count* labels for a dimension, for an LLM to choose from."""
    options = [option["label"] for option in _options(vocab, select_id)]
    if len(options) <= count:
        return options
    return random.Random(seed).sample(options, count)


# Dimensions the "LLM picks the scene" mode asks about:
# vocab select id -> (scene key, label shown to the model).
PICK_DIMENSIONS = {
    "place": ("place", "place"),
    "placeLight": ("lighting", "lighting"),
    "clothes": ("clothes", "her clothes"),
    "mood": ("mood", "mood"),
    "stylePack": ("style_pack", "camera/format style"),
    "sceneFormat": ("scene_format", "story wrapper"),
}


def scene_from_picks(
    picks: Dict[str, str],
    seed: Optional[int] = None,
    duration_s: float = 5.0,
    main_act: Optional[str] = None,
    artifact: str = "light",
) -> Dict[str, Any]:
    """Build a scene around values the prompt model chose (creative mode)."""
    vocab = load_vocab()
    scene = roll_scene(
        seed=seed, duration_s=duration_s, main_act=main_act, artifact=artifact
    )
    for dimension, (scene_key, _label) in PICK_DIMENSIONS.items():
        chosen = (picks or {}).get(dimension)
        if not chosen:
            continue
        scene[scene_key] = chosen
    scene["style_camera"] = (
        vocab.get("styles", {}).get(scene["style_pack"], {}) if scene.get("style_pack") else {}
    )
    scene["format_spine"] = (
        vocab.get("formats", {}).get(scene["scene_format"], {})
        if scene.get("scene_format")
        else {}
    )
    scene["picked_by"] = "llm"
    return scene


# Alternative keys the picker model answers with instead of the vocabulary ids:
# the friendlier label from the prompt ("lighting", "her clothes", "story
# wrapper") or a close variant. Keys are normalized via _normalize_pick_key.
_PICK_KEY_ALIASES = {
    # vocabulary ids, case/spacing tolerant
    "place": "place",
    "placelight": "placeLight",
    "place_light": "placeLight",
    "clothes": "clothes",
    "mood": "mood",
    "stylepack": "stylePack",
    "style_pack": "stylePack",
    "sceneformat": "sceneFormat",
    "scene_format": "sceneFormat",
    # scene/label variants seen from the model
    "light": "placeLight",
    "lighting": "placeLight",
    "clothing": "clothes",
    "her_clothes": "clothes",
    "outfit": "clothes",
    "wardrobe": "clothes",
    "vibe": "mood",
    "atmosphere": "mood",
    "style": "stylePack",
    "camera_style": "stylePack",
    "camera_format_style": "stylePack",
    "format": "sceneFormat",
    "wrapper": "sceneFormat",
    "story_wrapper": "sceneFormat",
}

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _normalize_pick_key(key: str) -> str:
    """Fold a key to its comparable form: lowercase, non-alnum runs to '_'."""
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")


def _picks_from_object(candidate: Any) -> Optional[Dict[str, str]]:
    """Keep known scene keys from a decoded object, mapped to vocabulary ids.

    Values must be non-empty strings; unknown keys and other types are dropped.
    """
    if not isinstance(candidate, dict):
        return None
    picks: Dict[str, str] = {}
    for raw_key, raw_value in candidate.items():
        if not isinstance(raw_key, str) or not isinstance(raw_value, str):
            continue
        dimension = _PICK_KEY_ALIASES.get(_normalize_pick_key(raw_key))
        value = raw_value.strip()
        if dimension and value:
            picks[dimension] = value
    return picks or None


def _scan_json_objects(text: str) -> Iterator[dict]:
    """Yield dicts from a tolerant right-to-left scan for parseable JSON."""
    decoder = json.JSONDecoder()
    for index in range(len(text) - 1, -1, -1):
        if text[index] != "{":
            continue
        try:
            candidate, _end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            yield candidate


def parse_scene_picks(text: Optional[str]) -> Optional[Dict[str, str]]:
    """Extract creative-mode picks from a raw picker answer.

    Tolerant by design: the JSON may sit after reasoning prose or inside a
    ```json fence, and the model may answer with scene labels instead of the
    vocabulary ids (mapped back). Returns picks keyed by PICK_DIMENSIONS ids
    with non-empty string values, or None when nothing usable was found.
    """
    if not text or "{" not in text:
        return None
    fenced = [match.group(1) for match in _FENCE_RE.finditer(text)]
    for source in [text] + fenced:
        for candidate in _scan_json_objects(source):
            picks = _picks_from_object(candidate)
            if picks:
                return picks
    return None


def roll_scene(
    seed: Optional[int] = None,
    duration_s: float = 5.0,
    main_act: Optional[str] = None,
    cast: Optional[str] = None,
    style: Optional[str] = None,
    scene_format: Optional[str] = None,
    artifact: str = "light",
) -> Dict[str, Any]:
    """Roll a concrete scene: setting, wardrobe, act arc and per-shot camera."""
    vocab = load_vocab()
    rng = random.Random(seed)
    stages = stage_arc(duration_s, main_act, seed=seed)

    shots: List[Dict[str, Any]] = []
    per_shot = duration_s / len(stages)
    for index, stage in enumerate(stages):
        actions = [
            label
            for field in STAGE_ACT_FIELDS.get(stage, ())
            if (label := _pick(rng, vocab, field))
        ]
        camera = {key: _pick(rng, vocab, field) for key, field in CAMERA_FIELDS.items()}
        # A locked-off shot has no meaningful speed; drop the contradiction.
        if camera.get("move") and "static" in camera["move"].lower():
            camera["speed"] = None
        shots.append(
            {
                "index": index + 1,
                "stage": stage,
                "start_s": round(index * per_shot, 2),
                "end_s": round((index + 1) * per_shot, 2),
                "actions": actions,
                "camera": camera,
            }
        )

    scene: Dict[str, Any] = {
        "seed": seed,
        "duration_s": duration_s,
        "cast": cast or _pick(rng, vocab, "cast"),
        "relationship": _pick(rng, vocab, "relate"),
        "place": _pick(rng, vocab, "place"),
        "lighting": _pick(rng, vocab, "placeLight"),
        "clothes": _pick(rng, vocab, "clothes"),
        "his_clothes": _pick(rng, vocab, "manCloth"),
        "mood": _pick(rng, vocab, "mood"),
        "style_pack": style or _pick(rng, vocab, "stylePack"),
        "scene_format": scene_format or _pick(rng, vocab, "sceneFormat"),
        "artifact_fix": artifact,
        "shots": shots,
        "stages": stages,
    }
    scene["soundscape"] = foley_guard(stages, vocab)
    scene["style_camera"] = (
        vocab.get("styles", {}).get(scene["style_pack"], {}) if scene["style_pack"] else {}
    )
    scene["format_spine"] = (
        vocab.get("formats", {}).get(scene["scene_format"], {}) if scene["scene_format"] else {}
    )
    return scene


def foley_guard(stages: Sequence[str], vocab: Optional[Dict[str, Any]] = None) -> str:
    """Per-stage close-mic foley text with H3's anti-artifact guards."""
    vocab = vocab or load_vocab()
    audio = vocab.get("audio_stage", {})
    parts = [audio[stage] for stage in stages if stage in audio]
    return " ".join(parts)


def artifact_lock(level: str = "light", vocab: Optional[Dict[str, Any]] = None) -> str:
    """Anatomy/artifact guard text for the requested level (off|light|full)."""
    if level == "off":
        return ""
    vocab = vocab or load_vocab()
    lock = vocab.get("artifact_lock", {})
    text = lock.get("light", "")
    if level == "full":
        text = f"{text} {lock.get('full_extra', '')}".strip()
    return text


def scene_brief(scene: Dict[str, Any], include_guards: bool = True) -> str:
    """Render a rolled scene as a compact constraint block for the prompt LLM."""
    lines = ["SCENE CONSTRAINTS — use these concrete details, do not swap the setting:"]
    for label, key in (
        ("cast", "cast"),
        ("relationship", "relationship"),
        ("place", "place"),
        ("lighting", "lighting"),
        ("her clothes", "clothes"),
        ("his clothes", "his_clothes"),
        ("mood", "mood"),
        ("scene format", "scene_format"),
        ("style pack", "style_pack"),
    ):
        value = scene.get(key)
        if value:
            lines.append(f"- {label}: {value}")

    spine = (scene.get("format_spine") or {}).get("spine")
    if spine:
        lines.append(f"- story wrapper: {spine}")
    style_camera = (scene.get("style_camera") or {}).get("camera")
    if style_camera:
        lines.append(f"- style camera: {style_camera}")

    lines.append(
        f"- timeline: {len(scene['shots'])} shots covering "
        f"{scene['duration_s']:.1f}s exactly (first shot has no timestamp)"
    )
    for shot in scene["shots"]:
        camera = ", ".join(
            f"{key} {value}" for key, value in shot["camera"].items() if value
        )
        actions = "; ".join(shot["actions"]) or "hold the moment"
        lines.append(
            f"- shot {shot['index']} [{shot['stage']}] {shot['start_s']:.1f}-{shot['end_s']:.1f}s: "
            f"{actions} | camera: {camera}"
        )

    if include_guards:
        lock = artifact_lock(scene.get("artifact_fix", "light"))
        if lock:
            lines.append(f"- anatomy guard: {lock}")
        if scene.get("soundscape"):
            lines.append(f"- sound foley (close-mic): {scene['soundscape']}")
    return "\n".join(lines)
