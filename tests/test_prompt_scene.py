"""Scene rolling and H3 guard text for the prompt generator.

Covers the vocabulary extraction, the deterministic scene roll, the rendered
constraint block, and the presence of the sound/anatomy guards in the H3 skill.
"""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src", "backend"))

from generation.prompt_scene import (  # noqa: E402
    artifact_lock,
    foley_guard,
    load_vocab,
    parse_scene_picks,
    roll_scene,
    scene_brief,
    stage_arc,
)


def _load_script(name: str):
    path = PROJECT_ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def extractor():
    return _load_script("extract_h3_scene_vocab.py")


def test_vocab_file_is_present_and_shaped():
    vocab = load_vocab()
    assert vocab["vocab"], "no dimensions extracted"
    assert vocab["audio_stage"], "no per-stage audio knowledge"
    assert vocab["artifact_lock"]["light"].startswith("ARTIFACT LOCK")
    for entry in vocab["vocab"].values():
        assert entry["options"], f"{entry['dimension']} has no options"


def test_roll_is_deterministic_per_seed():
    first = roll_scene(seed=1234, duration_s=5.0)
    second = roll_scene(seed=1234, duration_s=5.0)
    assert first == second
    assert roll_scene(seed=99, duration_s=5.0) != first


def test_roll_covers_the_duration_and_has_concrete_stages():
    scene = roll_scene(seed=7, duration_s=6.0)
    shots = scene["shots"]
    assert shots[0]["start_s"] == 0
    assert shots[-1]["end_s"] == pytest.approx(6.0, abs=0.05)
    assert "main" not in scene["stages"]
    for shot in shots:
        assert shot["actions"], f"stage {shot['stage']} rolled no action"
    # gaps must not exist: every shot starts where the previous one ended
    for previous, current in zip(shots, shots[1:]):
        assert current["start_s"] == pytest.approx(previous["end_s"], abs=0.01)


def test_locked_static_shots_drop_the_speed():
    for seed in range(40):
        for shot in roll_scene(seed=seed, duration_s=6.0)["shots"]:
            move = (shot["camera"].get("move") or "").lower()
            if "static" in move:
                assert not shot["camera"].get("speed"), "static shot kept a speed"


def test_stage_arc_scales_with_duration():
    assert len(stage_arc(2.0)) == 2
    assert len(stage_arc(5.0)) == 3
    assert len(stage_arc(12.0)) == 4
    assert stage_arc(5.0, main_act="bj")[1] == "bj"


def test_scene_brief_contains_setting_shots_and_guards():
    brief = scene_brief(roll_scene(seed=5, duration_s=5.0))
    assert brief.startswith("SCENE CONSTRAINTS")
    assert "anatomy guard: ARTIFACT LOCK" in brief
    assert "sound foley (close-mic)" in brief
    assert "shot 1" in brief and "camera:" in brief


def test_guards_render_per_level():
    assert artifact_lock("off") == ""
    light = artifact_lock("light")
    full = artifact_lock("full")
    assert "five separate fingers" in light
    assert len(full) > len(light) and "face smear" in full


def test_foley_guard_follows_the_stages():
    guard = foley_guard(["intro", "bj", "cum"])
    assert "No chewing" in guard or "No chew" in guard
    assert "No mouths eating" in guard
    assert foley_guard([]) == ""


def test_extraction_parses_a_page(extractor, tmp_path):
    page = tmp_path / "mini.html"
    page.write_text(
        """
        <select id="place"><option value="any">any</option>
          <option value="sofa">living-room sofa</option></select>
        <select id="sxPos"><option value="doggy">doggy</option></select>
        <script>
        const AUDIO_STAGE = { sex: "Skin slap at the hips. No chew." };
        const STYLE = { handheld: { label: "handheld", camera: "handheld", audio: "roomy" } };
        const FORMAT = { interview: { label: "interview", spine: "she answers the lens" } };
        function artifactLine(cfg){
          const light = "ARTIFACT LOCK: two arms.";
          if(cfg.artifactFix==="light") return light;
          return light + " No face smear.";
        }
        </script>
        """,
        encoding="utf-8",
    )
    data = extractor.build(page)
    assert data["vocab"]["place"]["options"][1]["label"] == "living-room sofa"
    assert data["audio_stage"]["sex"].startswith("Skin slap")
    assert data["styles"]["handheld"]["camera"] == "handheld"
    assert data["formats"]["interview"]["spine"].startswith("she answers")
    assert data["artifact_lock"]["full_extra"] == "No face smear."
    json.dumps(data)  # must stay serialisable


def test_h3_skill_carries_the_guards():
    """Regression: the H3 system prompt must keep the sound/anatomy guards."""
    source = (PROJECT_ROOT / "src" / "backend" / "app.py").read_text(encoding="utf-8")
    assert "SOUND AND BODY GUARDS" in source
    for phrase in (
        "never let wet sound read as chewing",
        "five separate fingers",
        "Continuity carries between shots",
        "One camera move per shot",
    ):
        assert phrase in source, f"H3 guard missing: {phrase}"


def test_shortlist_is_bounded_and_deterministic():
    from generation.prompt_scene import shortlist

    vocab = load_vocab()
    first = shortlist(vocab, "place", 6, seed=11)
    assert first == shortlist(vocab, "place", 6, seed=11)
    assert len(first) <= 6
    assert all(isinstance(label, str) and label for label in first)


def test_scene_from_picks_lands_on_the_scene_keys():
    """Creative mode picks must overwrite the scene values, not add new keys."""
    from generation.prompt_scene import PICK_DIMENSIONS, scene_from_picks

    picks = {
        "place": "rooftop at night",
        "placeLight": "cheap neon / sign spill",
        "clothes": "open robe",
        "mood": "eager",
        "stylePack": "phone selfie",
        "sceneFormat": "paid webcam",
    }
    scene = scene_from_picks(picks, seed=3, duration_s=5.0)
    for dimension, (scene_key, _label) in PICK_DIMENSIONS.items():
        assert scene[scene_key] == picks[dimension], f"{dimension} did not reach {scene_key}"
        if scene_key != dimension:
            assert dimension not in scene, f"{dimension} leaked in as its own key"
    assert scene["picked_by"] == "llm"
    assert scene["shots"], "creative scene still needs a rolled act arc"


def test_scene_from_picks_survives_empty_picks():
    from generation.prompt_scene import scene_from_picks

    scene = scene_from_picks({}, seed=9, duration_s=5.0)
    assert scene["place"] and scene["shots"]


# ── Creative-mode picker answer parsing ──────────────────────────────────────


def test_parse_scene_picks_plain_json():
    picks = parse_scene_picks('{"place": "rooftop at night", "mood": "eager"}')
    assert picks == {"place": "rooftop at night", "mood": "eager"}


def test_parse_scene_picks_after_reasoning_inside_fence():
    """The thinking model wraps its answer in prose and a ```json fence."""
    text = (
        "Thinking Process:\n\n1. Analyze the idea...\n2. Choose options...\n\n"
        "Here is my answer:\n"
        "```json\n"
        "{\n"
        '  "place": "rooftop at night",\n'
        '  "placeLight": "cheap neon / sign spill",\n'
        '  "clothes": "open robe",\n'
        '  "mood": "eager",\n'
        '  "stylePack": "phone selfie",\n'
        '  "sceneFormat": "paid webcam"\n'
        "}\n"
        "```\n"
        "That covers every category."
    )
    picks = parse_scene_picks(text)
    assert picks == {
        "place": "rooftop at night",
        "placeLight": "cheap neon / sign spill",
        "clothes": "open robe",
        "mood": "eager",
        "stylePack": "phone selfie",
        "sceneFormat": "paid webcam",
    }


def test_parse_scene_picks_maps_scene_keys_back():
    """Answers keyed by the friendly labels land on the vocabulary ids."""
    text = json.dumps(
        {
            "place": "rooftop at night",
            "lighting": "warm lamps",
            "her clothes": "open robe",
            "mood": "eager",
            "camera/format style": "phone selfie",
            "story wrapper": "paid webcam",
        }
    )
    picks = parse_scene_picks(text)
    assert set(picks) == {"place", "placeLight", "clothes", "mood", "stylePack", "sceneFormat"}
    assert picks["placeLight"] == "warm lamps"
    assert picks["clothes"] == "open robe"
    assert picks["stylePack"] == "phone selfie"
    assert picks["sceneFormat"] == "paid webcam"


def test_parse_scene_picks_is_tolerant_of_key_noise():
    """Case, spacing and underscores in ids must not break the mapping."""
    picks = parse_scene_picks(
        '{"Place": "bed", "PLACE_LIGHT": "daylight, window", "Scene Format": "paid webcam"}'
    )
    assert picks == {
        "place": "bed",
        "placeLight": "daylight, window",
        "sceneFormat": "paid webcam",
    }


def test_parse_scene_picks_rejects_unusable_answers():
    assert parse_scene_picks("") is None
    assert parse_scene_picks("no json here at all") is None
    assert parse_scene_picks('{"idea": "x", "notes": "y"}') is None
    # non-string / empty values are dropped, usable ones survive
    picks = parse_scene_picks('{"place": "bed", "shots": 3, "mood": ""}')
    assert picks == {"place": "bed"}
    # a complete answer wins over a later truncated one
    text = '{"place": "bed"} trailing junk {"mood": "eager", "clothes": "robe'
    assert parse_scene_picks(text) == {"place": "bed"}


def test_picker_wiring_uses_the_pure_parser():
    """The live picker must parse through parse_scene_picks, with fallbacks."""
    source = (PROJECT_ROOT / "src" / "backend" / "app.py").read_text(encoding="utf-8")
    assert "parse_scene_picks(content)" in source
    assert "reasoning_content" in source, "picker must also scan the thinking field"
    assert "Scene picker gave no usable JSON" in source, "failure needs a greppable warning"
    assert "_SCENE_PICK_TIMEOUT_S" in source


def test_prompt_requests_accept_the_new_controls():
    """The API models must carry the new knobs, or they vanish silently."""
    import importlib.util

    app_path = PROJECT_ROOT / "src" / "backend" / "app.py"
    source = app_path.read_text(encoding="utf-8")
    for field in ("scene_mode", "temperature", "llm_seed"):
        assert f"{field}:" in source, f"PromptGenerateRequest is missing {field}"
    assert "class PromptCompareRequest" in source
    assert '"/generate-prompt/compare"' in source
    # the new fields must reach the job dict, not just the request model
    for forwarded in ('"scene_mode": req.scene_mode', '"temperature": req.temperature', '"llm_seed": req.llm_seed'):
        assert forwarded in source, f"not forwarded to the LLM job: {forwarded}"
    assert "importlib" in importlib.util.__name__ or True
