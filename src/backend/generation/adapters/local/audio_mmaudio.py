"""
MMAudio adapter — text-to-audio generation via ComfyUI.

Supports TTS (ChatterBox), music (MMAudio), and SFX (MMAudio) modes.
"""

from __future__ import annotations

import logging
import random
import uuid
from typing import Any

from ...adapter import GenerationAdapter, ProgressCallback
from ...types import (
    AdapterConstraints,
    ComputeTarget,
    GenerationRequest,
    GenerationResult,
    LoraFormat,
    MediaType,
    Operation,
)

logger = logging.getLogger(__name__)

# Voice presets for ChatterBox TTS engine
VOICE_PRESETS = {
    "alloy": {"language": "English", "exaggeration": 0.4, "temperature": 0.7},
    "echo": {"language": "English", "exaggeration": 0.6, "temperature": 0.9},
    "fable": {"language": "English", "exaggeration": 0.8, "temperature": 1.0},
    "onyx": {"language": "English", "exaggeration": 0.3, "temperature": 0.6},
    "nova": {"language": "English", "exaggeration": 0.5, "temperature": 0.8},
    "shimmer": {"language": "English", "exaggeration": 0.35, "temperature": 0.75},
}


class MMAudioAdapter(GenerationAdapter):
    """
    Local audio generation via ComfyUI.

    Three modes:
    - tts: Text-to-speech via ChatterBox engine
    - music: Music generation via MMAudio model
    - sfx: Sound effects via MMAudio model
    """

    name = "local-mmaudio"
    model_family = ""
    supported_ops = {Operation.GENERATE}
    input_types = {MediaType.TEXT}
    output_type = MediaType.AUDIO
    compute = ComputeTarget.LOCAL
    lora_format = LoraFormat.NONE

    def __init__(self, comfyui_client_fn: Any = None) -> None:
        self._get_comfyui = comfyui_client_fn

    def constraints(self) -> AdapterConstraints:
        return AdapterConstraints(
            max_duration_seconds=60.0,
            supports_negative_prompt=False,
        )

    def _build_tts_workflow(self, req: GenerationRequest) -> dict:
        """Build ChatterBox TTS workflow."""
        voice = req.voice or "nova"
        voice_settings = VOICE_PRESETS.get(voice, VOICE_PRESETS["nova"])
        output_id = uuid.uuid4().hex[:8]

        return {
            "1": {
                "class_type": "ChatterBoxEngineNode",
                "inputs": {
                    "language": voice_settings["language"],
                    "device": "auto",
                    "exaggeration": voice_settings["exaggeration"],
                    "temperature": voice_settings["temperature"],
                    "cfg_weight": 0.5,
                    "crash_protection_template": "hmm ,, {seg} hmm ,,",
                },
            },
            "2": {
                "class_type": "UnifiedTTSTextNode",
                "inputs": {
                    "TTS_engine": ["1", 0],
                    "text": req.prompt or "",
                    "narrator_voice": "none",
                    "seed": random.randint(0, 2**32 - 1),
                    "enable_chunking": True,
                    "max_chars_per_chunk": 400,
                    "chunk_combination_method": "auto",
                    "silence_between_chunks_ms": 100,
                    "enable_audio_cache": True,
                    "batch_size": 0,
                },
            },
            "3": {
                "class_type": "SaveAudio",
                "inputs": {
                    "audio": ["2", 0],
                    "filename_prefix": f"tts_{output_id}",
                },
            },
        }

    def _build_music_workflow(self, req: GenerationRequest) -> dict:
        """Build MMAudio music generation workflow."""
        style = req.audio_style or "cinematic"
        music_prompt = f"{style} music, {req.prompt or ''}"
        duration = req.duration or 10.0
        output_id = uuid.uuid4().hex[:8]

        return {
            "1": {
                "class_type": "MMAudioModelLoader",
                "inputs": {
                    "mmaudio_model": "mmaudio_large_44k_v2_fp16.safetensors",
                    "base_precision": "fp16",
                },
            },
            "2": {
                "class_type": "MMAudioFeatureUtilsLoader",
                "inputs": {
                    "synchformer_model": "mmaudio_synchformer_fp16.safetensors",
                    "vae_model": "mmaudio_vae_44k_fp16.safetensors",
                    "clip_model": "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors",
                    "mode": "44k",
                    "precision": "fp16",
                },
            },
            "3": {
                "class_type": "MMAudioSampler",
                "inputs": {
                    "mmaudio_model": ["1", 0],
                    "feature_utils": ["2", 0],
                    "prompt": music_prompt,
                    "negative_prompt": "noise, distortion, glitch, silence",
                    "duration": float(duration),
                    "steps": 25,
                    "cfg": 4.5,
                    "seed": random.randint(0, 2**32 - 1),
                    "mask_away_clip": False,
                    "force_offload": True,
                },
            },
            "4": {
                "class_type": "SaveAudio",
                "inputs": {
                    "audio": ["3", 0],
                    "filename_prefix": f"music_{output_id}",
                },
            },
        }

    def _build_sfx_workflow(self, req: GenerationRequest) -> dict:
        """Build MMAudio SFX workflow (shorter duration, different negative prompt)."""
        duration = min(req.duration or 10.0, 10.0)
        output_id = uuid.uuid4().hex[:8]

        return {
            "1": {
                "class_type": "MMAudioModelLoader",
                "inputs": {
                    "mmaudio_model": "mmaudio_large_44k_v2_fp16.safetensors",
                    "base_precision": "fp16",
                },
            },
            "2": {
                "class_type": "MMAudioFeatureUtilsLoader",
                "inputs": {
                    "synchformer_model": "mmaudio_synchformer_fp16.safetensors",
                    "vae_model": "mmaudio_vae_44k_fp16.safetensors",
                    "clip_model": "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors",
                    "mode": "44k",
                    "precision": "fp16",
                },
            },
            "3": {
                "class_type": "MMAudioSampler",
                "inputs": {
                    "mmaudio_model": ["1", 0],
                    "feature_utils": ["2", 0],
                    "prompt": req.prompt or "",
                    "negative_prompt": "music, speech, voice, singing",
                    "duration": float(duration),
                    "steps": 25,
                    "cfg": 4.5,
                    "seed": random.randint(0, 2**32 - 1),
                    "mask_away_clip": False,
                    "force_offload": True,
                },
            },
            "4": {
                "class_type": "SaveAudio",
                "inputs": {
                    "audio": ["3", 0],
                    "filename_prefix": f"sfx_{output_id}",
                },
            },
        }

    def build_workflow(self, req: GenerationRequest) -> dict:
        """Build the appropriate ComfyUI workflow based on audio mode."""
        mode = req.audio_mode or "music"
        if mode == "tts":
            return self._build_tts_workflow(req)
        elif mode == "sfx":
            return self._build_sfx_workflow(req)
        return self._build_music_workflow(req)

    def cost(self, req: GenerationRequest) -> int:
        duration = req.duration or 10.0
        if duration <= 10:
            return 3
        return 5

    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        if self._get_comfyui is None:
            raise RuntimeError("ComfyUI client not available")

        client = self._get_comfyui()
        mode = req.audio_mode or "music"

        workflow = self.build_workflow(req)
        prompt_id = client.queue_prompt(workflow)

        if not prompt_id:
            raise RuntimeError(f"Failed to queue {mode} audio workflow")

        logger.info(f"🎵 {mode.upper()} queued: {prompt_id}")

        return GenerationResult(
            prompt_id=prompt_id,
            status="queued_local",
            compute_target=ComputeTarget.LOCAL,
            credits_used=0,  # Router fills this in
            adapter_name=self.name,
            meta={
                "mode": mode,
                "duration": req.duration or 10.0,
            },
        )
