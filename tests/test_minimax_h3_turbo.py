"""
Tests for MiniMax-H3 turbo quality modes (draft / standard / full).

The turbo presets (ModelTC/Minimax-H3-Turbo) chain a turbo LoRA in front of
any user LoRAs, override the step count and sampler (euler), and — for the
768p-trained 4-step variant — override the sigma shift via MiniMaxH3SigmaShift
(shift 6/3 instead of the checkpoint-baked 12/3).

Covers: workflow shape per quality mode, LoRA chain ordering with user LoRAs,
and unchanged behaviour for the default full mode.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "backend"))

from comfyui_client import ComfyUIClient, MINIMAX_H3_TURBO_PRESETS  # noqa: E402


def _build(quality_mode: str, lora_configs=None) -> dict:
    client = ComfyUIClient()
    return client.build_cloud_minimax_h3_t2v_workflow(
        prompt="test prompt",
        aspect_ratio="16:9",
        megapixels=0.4,
        num_frames=73,
        seed=42,
        quality_mode=quality_mode,
        lora_configs=lora_configs,
    )


def test_full_mode_unchanged():
    """Default full mode: 20 steps, res_multistep, no turbo LoRA / shift node."""
    wf = _build("full")
    assert wf["7"]["inputs"]["sampler_name"] == "res_multistep"
    assert wf["8"]["inputs"]["steps"] == 20
    assert wf["6"]["inputs"]["model"] == ["1", 0]  # straight from UNETLoader
    lora_nodes = [n for n in wf.values() if n["class_type"] == "LoraLoaderModelOnly"]
    assert lora_nodes == []
    assert not any(n["class_type"] == "MiniMaxH3SigmaShift" for n in wf.values())


def test_draft_mode_4step_shift63():
    """Draft: 4-step 768p LoRA + euler + MiniMaxH3SigmaShift 6/3."""
    wf = _build("draft")
    preset = MINIMAX_H3_TURBO_PRESETS["draft"]
    assert wf["7"]["inputs"]["sampler_name"] == "euler"
    assert wf["8"]["inputs"]["steps"] == 4
    assert wf["16"]["class_type"] == "LoraLoaderModelOnly"
    assert wf["16"]["inputs"]["lora_name"] == preset["lora"]
    assert wf["16"]["inputs"]["strength_model"] == 1.0
    assert wf["17"]["class_type"] == "MiniMaxH3SigmaShift"
    assert wf["17"]["inputs"]["shift_video"] == 6.0
    assert wf["17"]["inputs"]["shift_audio"] == 3.0
    # guider + scheduler consume the sigma-shift output
    assert wf["6"]["inputs"]["model"] == ["17", 0]
    assert wf["8"]["inputs"]["model"] == ["17", 0]


def test_standard_mode_8step_no_shift_node():
    """Standard: 8-step LoRA + euler, base shift 12/3 (no SigmaShift node)."""
    wf = _build("standard")
    preset = MINIMAX_H3_TURBO_PRESETS["standard"]
    assert wf["7"]["inputs"]["sampler_name"] == "euler"
    assert wf["8"]["inputs"]["steps"] == 8
    assert wf["16"]["inputs"]["lora_name"] == preset["lora"]
    assert not any(n["class_type"] == "MiniMaxH3SigmaShift" for n in wf.values())
    assert wf["6"]["inputs"]["model"] == ["16", 0]


def test_turbo_lora_chains_before_user_loras():
    """Turbo LoRA first, then user LoRAs, then the sigma-shift node."""
    user_lora = {"name": "user_style_lora.safetensors", "strength": 0.8}
    wf = _build("draft", lora_configs=[user_lora])
    assert wf["16"]["inputs"]["lora_name"] == MINIMAX_H3_TURBO_PRESETS["draft"]["lora"]
    assert wf["17"]["class_type"] == "LoraLoaderModelOnly"
    assert wf["17"]["inputs"]["lora_name"] == "user_style_lora.safetensors"
    assert wf["17"]["inputs"]["model"] == ["16", 0]
    assert wf["18"]["class_type"] == "MiniMaxH3SigmaShift"
    assert wf["18"]["inputs"]["model"] == ["17", 0]


def test_invalid_quality_mode_falls_back_to_full():
    """Unknown quality mode behaves like full (no turbo nodes)."""
    wf = _build("does-not-exist")
    assert wf["7"]["inputs"]["sampler_name"] == "res_multistep"
    assert wf["8"]["inputs"]["steps"] == 20
    assert not any(n["class_type"] == "LoraLoaderModelOnly" for n in wf.values())


def test_local_builder_passes_quality_mode():
    """Local (Windows) builder forwards the quality mode to the cloud builder."""
    client = ComfyUIClient()
    wf = client.build_local_minimax_h3_t2v_workflow(
        prompt="test prompt",
        num_frames=73,
        seed=42,
        quality_mode="draft",
    )
    assert wf["7"]["inputs"]["sampler_name"] == "euler"
    assert wf["8"]["inputs"]["steps"] == 4


def test_eros_draft_baked_turbo():
    """Eros draft: turbo fused in the checkpoint — no LoRA/shift nodes."""
    client = ComfyUIClient()
    wf = client.build_cloud_minimax_h3_t2v_workflow(
        prompt="test prompt",
        aspect_ratio="16:9",
        megapixels=0.4,
        num_frames=73,
        seed=42,
        quality_mode="draft",
        model_variant="eros",
    )
    assert wf["1"]["inputs"]["unet_name"] == "h3ErosMax_beta5_3185144.safetensors"
    assert wf["7"]["inputs"]["sampler_name"] == "euler"
    assert wf["8"]["inputs"]["steps"] == 4
    assert not any(n["class_type"] == "LoraLoaderModelOnly" for n in wf.values())
    assert not any(n["class_type"] == "MiniMaxH3SigmaShift" for n in wf.values())


def test_eros_standard_8steps_no_nodes():
    client = ComfyUIClient()
    wf = client.build_cloud_minimax_h3_t2v_workflow(
        prompt="test prompt",
        aspect_ratio="16:9",
        megapixels=0.4,
        num_frames=73,
        seed=42,
        quality_mode="standard",
        model_variant="eros",
    )
    assert wf["1"]["inputs"]["unet_name"] == "h3ErosMax_beta5_3185144.safetensors"
    assert wf["8"]["inputs"]["steps"] == 8
    assert not any(n["class_type"] == "LoraLoaderModelOnly" for n in wf.values())


def test_eros_full_keeps_its_own_checkpoint():
    """Full quality never swaps the model: same checkpoint, base sampler, full steps."""
    client = ComfyUIClient()
    wf = client.build_cloud_minimax_h3_t2v_workflow(
        prompt="test prompt",
        aspect_ratio="16:9",
        megapixels=0.4,
        num_frames=73,
        steps=20,
        seed=42,
        quality_mode="full",
        model_variant="eros",
    )
    assert wf["1"]["inputs"]["unet_name"] == "h3ErosMax_beta5_3185144.safetensors"
    assert wf["7"]["inputs"]["sampler_name"] == "res_multistep"
    assert wf["8"]["inputs"]["steps"] == 20
    assert not any(n["class_type"] == "LoraLoaderModelOnly" for n in wf.values())


def test_eros_int8_non_turbo_uses_full_steps():
    """Non-turbo checkpoints ignore the turbo presets and keep the step count."""
    client = ComfyUIClient()
    wf = client.build_cloud_minimax_h3_t2v_workflow(
        prompt="test prompt",
        aspect_ratio="16:9",
        megapixels=0.4,
        num_frames=73,
        steps=20,
        seed=42,
        quality_mode="standard",
        model_variant="eros_int8",
    )
    assert wf["1"]["inputs"]["unet_name"] == "h3ErosMax_beta5_3185154.safetensors"
    assert wf["7"]["inputs"]["sampler_name"] == "res_multistep"
    assert wf["8"]["inputs"]["steps"] == 20


def test_eros_int8_turbo_is_a_four_step_variant():
    client = ComfyUIClient()
    wf = client.build_cloud_minimax_h3_t2v_workflow(
        prompt="test prompt",
        aspect_ratio="16:9",
        megapixels=0.4,
        num_frames=73,
        seed=42,
        quality_mode="draft",
        model_variant="eros_int8_turbo",
    )
    assert wf["1"]["inputs"]["unet_name"] == "h3ErosMax_beta5_3178732.safetensors"
    assert wf["7"]["inputs"]["sampler_name"] == "euler"
    assert wf["8"]["inputs"]["steps"] == 4


def test_dasiwa_variants():
    """DaSiWa turbo runs few euler steps; the non-turbo file keeps full steps."""
    client = ComfyUIClient()
    turbo = client.build_cloud_minimax_h3_t2v_workflow(
        prompt="test prompt",
        aspect_ratio="16:9",
        megapixels=0.4,
        num_frames=73,
        seed=42,
        quality_mode="standard",
        model_variant="dasiwa_turbo",
    )
    assert turbo["1"]["inputs"]["unet_name"] == (
        "DasiwaMinimaxH3_dasiwaHybridTurboV2_3203135.safetensors"
    )
    assert turbo["7"]["inputs"]["sampler_name"] == "euler"
    assert turbo["8"]["inputs"]["steps"] == 8

    plain = client.build_cloud_minimax_h3_t2v_workflow(
        prompt="test prompt",
        aspect_ratio="16:9",
        megapixels=0.4,
        num_frames=73,
        steps=20,
        seed=42,
        quality_mode="standard",
        model_variant="dasiwa",
    )
    assert plain["1"]["inputs"]["unet_name"] == (
        "DasiwaMinimaxH3_dasiwaHybridV2_3203130.safetensors"
    )
    assert plain["7"]["inputs"]["sampler_name"] == "res_multistep"
    assert plain["8"]["inputs"]["steps"] == 20
