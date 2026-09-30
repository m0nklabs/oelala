"""
Tests for MiniMax-H3 workflow LoRA injection (single-stage LoraLoaderModelOnly).

These exercise the workflow builders in ``comfyui_client`` directly — no
network is touched; the builder only assembles the prompt/dict graph. They
verify that requested LoRAs are chained in front of the base UNETLoader model
and that the guider (node 6) and scheduler (node 8) consume the last loader's
output, with no node-id collisions against the fixed image/output nodes.

The local builders re-map requested LoRA names onto the target server's own
LoRA list (Windows servers enumerate subfolder loras with backslashes); the
stubbed ``get_lora_names`` keeps these tests offline.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "backend"))

from comfyui_client import ComfyUIClient, _match_lora_name


def _client(server_loras=None) -> ComfyUIClient:
    # Any non-resolving host works — the builders only assemble the workflow
    # dict and never touch the network.
    c = ComfyUIClient(host="comfyui.test.internal", port=8188)
    if server_loras is not None:
        c.get_lora_names = lambda: list(server_loras)
    return c


def test_t2v_injects_single_lora():
    wf = _client(
        ["bounceV07_fl2va-000230_Intense.safetensors"]
    ).build_local_minimax_h3_t2v_workflow(
        prompt="test",
        num_frames=124,
        fps=24,
        seed=42,
        steps=20,
        lora_configs=[
            {"name": "bounceV07_fl2va-000230_Intense.safetensors", "strength": 0.8}
        ],
    )
    lora = wf["16"]
    assert lora["class_type"] == "LoraLoaderModelOnly"
    assert lora["inputs"]["lora_name"] == "bounceV07_fl2va-000230_Intense.safetensors"
    assert lora["inputs"]["strength_model"] == 0.8
    assert lora["inputs"]["model"] == ["1", 0]
    # guider + scheduler now consume the lora output
    assert wf["6"]["inputs"]["model"] == ["16", 0]
    assert wf["8"]["inputs"]["model"] == ["16", 0]


def test_t2v_no_loras_keeps_base_model_link():
    wf = _client().build_local_minimax_h3_t2v_workflow(
        prompt="test", num_frames=124, fps=24, seed=1, steps=20
    )
    assert "16" not in wf
    assert wf["6"]["inputs"]["model"] == ["1", 0]
    assert wf["8"]["inputs"]["model"] == ["1", 0]


def test_i2v_chains_multiple_loras_and_keeps_image_node():
    wf = _client(["A.safetensors", "B.safetensors"]).build_local_minimax_h3_i2v_workflow(
        image_name="first.png",
        prompt="test",
        num_frames=124,
        fps=24,
        seed=1,
        steps=20,
        lora_configs=[
            {"name": "A.safetensors", "strength": 1.0},
            {"name": "B.safetensors", "strength": 0.5},
        ],
    )
    assert wf["14"]["class_type"] == "LoadImage"  # first-frame keyframe
    assert wf["16"]["inputs"]["model"] == ["1", 0]
    assert wf["17"]["inputs"]["model"] == ["16", 0]
    assert wf["6"]["inputs"]["model"] == ["17", 0]
    assert wf["8"]["inputs"]["model"] == ["17", 0]


def test_t2v_output_kind_savevideo_uses_savevideo_node():
    wf = _client(["Bounce.safetensors"]).build_local_minimax_h3_t2v_workflow(
        prompt="test",
        num_frames=124,
        fps=24,
        seed=1,
        steps=20,
        lora_configs=[{"name": "Bounce.safetensors", "strength": 0.7}],
    )
    # local output is SaveVideo (node 13/15); lora node 16 collides with none
    assert wf["16"]["class_type"] == "LoraLoaderModelOnly"
    assert "SaveVideo" in {
        wf[k]["class_type"] for k in wf if k in ("13", "15")
    }


# --- server-side LoRA name resolution (Windows backslash / missing loras) ---


def test_match_lora_name_variants():
    available = [
        "minimax-h3\\PenisV2_minimax-h3_epoch60.safetensors",
        "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
    ]
    assert _match_lora_name("minimax-h3\\PenisV2_minimax-h3_epoch60.safetensors", available) == available[0]
    assert _match_lora_name("minimax-h3/PenisV2_minimax-h3_epoch60.safetensors", available) == available[0]
    assert _match_lora_name("PenisV2_minimax-h3_epoch60.safetensors", available) == available[0]
    assert _match_lora_name("minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors", available) == available[1]
    assert _match_lora_name("minimax-h3/Vagina_minimax-h3_epoch20.safetensors", available) is None


def test_local_builder_maps_forward_slash_to_server_form():
    wf = _client(
        [
            "minimax-h3\\PenisV2_minimax-h3_epoch60.safetensors",
            "minimax-h3\\Vagina_minimax-h3_epoch20.safetensors",
        ]
    ).build_local_minimax_h3_t2v_workflow(
        prompt="test",
        num_frames=124,
        fps=24,
        seed=1,
        steps=20,
        lora_configs=[
            {"name": "minimax-h3/PenisV2_minimax-h3_epoch60.safetensors", "strength": 0.8},
            {"name": "minimax-h3/Vagina_minimax-h3_epoch20.safetensors", "strength": 0.8},
        ],
    )
    # exact strings the Windows server validates lora_name against
    assert wf["16"]["inputs"]["lora_name"] == "minimax-h3\\PenisV2_minimax-h3_epoch60.safetensors"
    assert wf["17"]["inputs"]["lora_name"] == "minimax-h3\\Vagina_minimax-h3_epoch20.safetensors"
    assert wf["6"]["inputs"]["model"] == ["17", 0]


def test_local_builder_drops_lora_missing_on_server():
    wf = _client(["minimax-h3\\Bounce.safetensors"]).build_local_minimax_h3_t2v_workflow(
        prompt="test",
        num_frames=124,
        fps=24,
        seed=1,
        steps=20,
        lora_configs=[{"name": "minimax-h3/Missing.safetensors", "strength": 0.8}],
    )
    assert "16" not in wf
    assert wf["6"]["inputs"]["model"] == ["1", 0]
    assert wf["8"]["inputs"]["model"] == ["1", 0]


def test_local_builder_passes_through_when_lora_list_unavailable():
    wf = _client([]).build_local_minimax_h3_t2v_workflow(
        prompt="test",
        num_frames=124,
        fps=24,
        seed=1,
        steps=20,
        lora_configs=[{"name": "minimax-h3/Whatever.safetensors", "strength": 0.8}],
    )
    # fail-open: server list unreadable -> names go through unchanged
    assert wf["16"]["inputs"]["lora_name"] == "minimax-h3/Whatever.safetensors"
