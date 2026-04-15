"""
Tests for Phase 6 adapters — remaining local adapters.

14 adapters: Lightning I2V, I2I, V2V, upscale image/video, interpolate,
face swap image/video, MMAudio, voice clone, lipsync, inpaint,
caption image/video.
"""

from __future__ import annotations

import sys
import os
import unittest
from unittest.mock import MagicMock, AsyncMock

# Ensure generation package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "backend"))

from generation.types import (
    AdapterConstraints,
    ComputeTarget,
    GenerationRequest,
    GenerationResult,
    LoraFormat,
    MediaType,
    Operation,
)

from generation.adapters.local.i2v_wan22_lightning import Wan22LocalI2VLightningAdapter
from generation.adapters.local.i2i_transform import I2ITransformAdapter
from generation.adapters.local.v2v import V2VStyleTransferAdapter
from generation.adapters.local.upscale_image import ImageUpscaleAdapter
from generation.adapters.local.upscale_video import VideoUpscaleAdapter
from generation.adapters.local.interpolate import InterpolateAdapter
from generation.adapters.local.face_swap import FaceSwapImageAdapter
from generation.adapters.local.face_swap_video import FaceSwapVideoAdapter
from generation.adapters.local.audio_mmaudio import MMAudioAdapter
from generation.adapters.local.voice_clone import VoiceCloneAdapter
from generation.adapters.local.lipsync import LipSyncAdapter
from generation.adapters.local.inpaint import InpaintAdapter
from generation.adapters.local.caption_image import ImageCaptionAdapter
from generation.adapters.local.caption_video import VideoCaptionAdapter


# ── Helpers ──────────────────────────────────────────────────


def _make_req(**kwargs) -> GenerationRequest:
    defaults = {"operation": Operation.GENERATE, "target_type": MediaType.IMAGE, "prompt": "test"}
    defaults.update(kwargs)
    return GenerationRequest(**defaults)


def _mock_comfyui():
    client = MagicMock()
    client.queue_prompt.return_value = "prompt-1234"
    return lambda: client


def _mock_face_service():
    svc = MagicMock()
    svc.swap_faces_to_bytes = AsyncMock(return_value=b"png-data")
    svc.swap_faces_in_video = AsyncMock(return_value="/tmp/swapped.mp4")
    return lambda: svc


def _mock_guardian():
    guardian = MagicMock()
    guardian.caption_image = AsyncMock(return_value="A cat sitting on a mat.")
    guardian.caption_video = AsyncMock(return_value="A dog running in a park.")
    return lambda: guardian


# ── Lightning I2V ────────────────────────────────────────────


class TestWan22LightningI2V(unittest.TestCase):
    def test_identity(self):
        a = Wan22LocalI2VLightningAdapter()
        assert a.name == "wan22-local-i2v-lightning"
        assert a.model_family == "wan2.2"
        assert a.lora_format == LoraFormat.DUAL_STAGE
        assert a.output_type == MediaType.VIDEO

    def test_constraints(self):
        a = Wan22LocalI2VLightningAdapter()
        c = a.constraints()
        assert c.max_frames == 41
        assert c.default_steps == 4
        assert c.default_cfg == 1.0

    def test_quant_config(self):
        a = Wan22LocalI2VLightningAdapter()
        qc = a._get_quant_config()
        assert qc.builder_method == "build_enhanced_workflow"
        assert qc.name == "Q4KM_lightning"

    def test_cost(self):
        a = Wan22LocalI2VLightningAdapter()
        req = _make_req(operation=Operation.GENERATE, target_type=MediaType.VIDEO, frames=41)
        assert a.cost(req) == 5

    def test_build_workflow(self):
        mock_comfyui = MagicMock()
        mock_comfyui.build_enhanced_workflow.return_value = {"nodes": []}
        a = Wan22LocalI2VLightningAdapter(comfyui_client_fn=lambda: mock_comfyui)
        req = _make_req(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            input_images=["test.png"],
            frames=41,
        )
        wf = a.build_workflow(req)
        mock_comfyui.build_enhanced_workflow.assert_called_once()


# ── I2I Transform ────────────────────────────────────────────


class TestI2ITransform(unittest.TestCase):
    def test_identity(self):
        a = I2ITransformAdapter()
        assert a.name == "local-i2i-transform"
        assert a.model_family == "sdxl"
        assert Operation.TRANSFORM in a.supported_ops
        assert MediaType.IMAGE in a.input_types
        assert a.output_type == MediaType.IMAGE

    def test_constraints(self):
        c = I2ITransformAdapter().constraints()
        assert c.default_steps == 25
        assert c.default_cfg == 7.5
        assert "dpmpp_2m" in c.supported_samplers

    def test_cost_base(self):
        a = I2ITransformAdapter()
        req = _make_req(operation=Operation.TRANSFORM)
        assert a.cost(req) == 2

    def test_cost_with_face_features(self):
        a = I2ITransformAdapter()
        req = _make_req(
            operation=Operation.TRANSFORM,
            face_id=True,
            face_detailer=True,
            face_restore=True,
        )
        assert a.cost(req) == 8  # 2 + 3 + 2 + 1

    def test_cost_partial_face(self):
        a = I2ITransformAdapter()
        req = _make_req(operation=Operation.TRANSFORM, face_id=True)
        assert a.cost(req) == 5  # 2 + 3


# ── V2V ──────────────────────────────────────────────────────


class TestV2V(unittest.TestCase):
    def test_identity(self):
        a = V2VStyleTransferAdapter()
        assert a.name == "local-v2v"
        assert a.model_family == "wan2.2"
        assert Operation.TRANSFORM in a.supported_ops
        assert MediaType.VIDEO in a.input_types
        assert a.output_type == MediaType.VIDEO

    def test_constraints(self):
        c = V2VStyleTransferAdapter().constraints()
        assert c.max_frames == 161
        assert 16 in c.allowed_fps

    def test_cost_short(self):
        a = V2VStyleTransferAdapter()
        req = _make_req(operation=Operation.TRANSFORM, target_type=MediaType.VIDEO, frames=41)
        assert a.cost(req) == 5

    def test_cost_medium(self):
        a = V2VStyleTransferAdapter()
        req = _make_req(operation=Operation.TRANSFORM, target_type=MediaType.VIDEO, frames=121)
        assert a.cost(req) == 8


# ── Image Upscale ────────────────────────────────────────────


class TestImageUpscale(unittest.TestCase):
    def test_identity(self):
        a = ImageUpscaleAdapter()
        assert a.name == "local-upscale-image"
        assert Operation.UPSCALE in a.supported_ops
        assert a.output_type == MediaType.IMAGE

    def test_cost(self):
        assert ImageUpscaleAdapter().cost(_make_req(operation=Operation.UPSCALE)) == 2

    def test_build_workflow(self):
        a = ImageUpscaleAdapter()
        req = _make_req(
            operation=Operation.UPSCALE,
            input_images=["test.png"],
        )
        wf = a.build_workflow(req)
        assert wf["2"]["inputs"]["model_name"] == "RealESRGAN_x4plus.pth"

    def test_build_workflow_custom_model(self):
        a = ImageUpscaleAdapter()
        req = _make_req(
            operation=Operation.UPSCALE,
            input_images=["test.png"],
            upscale_model="SwinIR_x4.pth",
        )
        wf = a.build_workflow(req)
        assert wf["2"]["inputs"]["model_name"] == "SwinIR_x4.pth"


# ── Video Upscale ────────────────────────────────────────────


class TestVideoUpscale(unittest.TestCase):
    def test_identity(self):
        a = VideoUpscaleAdapter()
        assert a.name == "local-upscale-video"
        assert a.output_type == MediaType.VIDEO

    def test_cost_lanczos(self):
        a = VideoUpscaleAdapter()
        req = _make_req(operation=Operation.UPSCALE, target_type=MediaType.VIDEO, upscale_preset="fast")
        assert a.cost(req) == 2

    def test_cost_seedvr2(self):
        a = VideoUpscaleAdapter()
        req = _make_req(operation=Operation.UPSCALE, target_type=MediaType.VIDEO, upscale_preset="quality")
        assert a.cost(req) == 10

    def test_resolve_model_preset(self):
        a = VideoUpscaleAdapter()
        req = _make_req(upscale_preset="fast")
        assert a._resolve_model(req) == "lanczos"
        req2 = _make_req(upscale_preset="quality")
        assert a._resolve_model(req2) == "seedvr2"
        req3 = _make_req(upscale_preset="balanced")
        assert a._resolve_model(req3) == "realesrgan"


# ── Interpolate ──────────────────────────────────────────────


class TestInterpolate(unittest.TestCase):
    def test_identity(self):
        a = InterpolateAdapter()
        assert a.name == "local-interpolate"
        assert Operation.INTERPOLATE in a.supported_ops
        assert a.output_type == MediaType.VIDEO

    def test_cost(self):
        assert InterpolateAdapter().cost(_make_req(operation=Operation.INTERPOLATE)) == 3

    def test_constraints_fps(self):
        c = InterpolateAdapter().constraints()
        assert 60 in c.allowed_fps
        assert 120 in c.allowed_fps


# ── Face Swap Image ──────────────────────────────────────────


class TestFaceSwapImage(unittest.TestCase):
    def test_identity(self):
        a = FaceSwapImageAdapter()
        assert a.name == "local-faceswap"
        assert Operation.SWAP in a.supported_ops
        assert a.output_type == MediaType.IMAGE

    def test_constraints(self):
        c = FaceSwapImageAdapter().constraints()
        assert c.max_input_images == 2

    def test_cost(self):
        assert FaceSwapImageAdapter().cost(_make_req(operation=Operation.SWAP)) == 1


# ── Face Swap Video ──────────────────────────────────────────


class TestFaceSwapVideo(unittest.TestCase):
    def test_identity(self):
        a = FaceSwapVideoAdapter()
        assert a.name == "local-faceswap-video"
        assert Operation.SWAP in a.supported_ops
        assert MediaType.VIDEO in a.input_types
        assert a.output_type == MediaType.VIDEO

    def test_cost(self):
        assert FaceSwapVideoAdapter().cost(_make_req(operation=Operation.SWAP)) == 5


# ── MMAudio ──────────────────────────────────────────────────


class TestMMAudio(unittest.TestCase):
    def test_identity(self):
        a = MMAudioAdapter()
        assert a.name == "local-mmaudio"
        assert Operation.GENERATE in a.supported_ops
        assert MediaType.TEXT in a.input_types
        assert a.output_type == MediaType.AUDIO

    def test_cost_short(self):
        a = MMAudioAdapter()
        req = _make_req(duration=5.0)
        assert a.cost(req) == 3

    def test_cost_long(self):
        a = MMAudioAdapter()
        req = _make_req(duration=30.0)
        assert a.cost(req) == 5

    def test_constraints(self):
        c = MMAudioAdapter().constraints()
        assert c.max_duration_seconds == 60.0
        assert c.supports_negative_prompt is False


# ── Voice Clone ──────────────────────────────────────────────


class TestVoiceClone(unittest.TestCase):
    def test_identity(self):
        a = VoiceCloneAdapter()
        assert a.name == "local-voice-clone"
        assert a.output_type == MediaType.AUDIO
        assert MediaType.AUDIO in a.input_types

    def test_cost(self):
        assert VoiceCloneAdapter().cost(_make_req()) == 20

    def test_build_workflow(self):
        a = VoiceCloneAdapter()
        req = _make_req(
            prompt="Hello world",
            checkpoint="F5v1",
            voice_sample_path="/tmp/voice.wav",
        )
        wf = a.build_workflow(req)
        assert wf["1"]["class_type"] == "F5TTSAudio"
        assert wf["1"]["inputs"]["model_type"] == "F5v1"


# ── LipSync ──────────────────────────────────────────────────


class TestLipSync(unittest.TestCase):
    def test_identity(self):
        a = LipSyncAdapter()
        assert a.name == "local-lipsync"
        assert Operation.LIPSYNC in a.supported_ops
        assert a.output_type == MediaType.VIDEO

    def test_constraints(self):
        c = LipSyncAdapter().constraints()
        assert c.allowed_fps == [25]
        assert c.default_steps == 20

    def test_cost(self):
        assert LipSyncAdapter().cost(_make_req(operation=Operation.LIPSYNC)) == 5

    def test_build_workflow(self):
        a = LipSyncAdapter()
        req = _make_req(
            operation=Operation.LIPSYNC,
            target_type=MediaType.VIDEO,
            input_video="video.mp4",
            input_audio="audio.wav",
            lips_expression=2.0,
        )
        wf = a.build_workflow(req)
        assert wf["3"]["class_type"] == "LatentSyncNode"
        assert wf["3"]["inputs"]["lips_expression"] == 2.0
        assert wf["4"]["inputs"]["frame_rate"] == 25


# ── Inpaint ──────────────────────────────────────────────────


class TestInpaint(unittest.TestCase):
    def test_identity(self):
        a = InpaintAdapter()
        assert a.name == "local-inpaint"
        assert Operation.INPAINT in a.supported_ops
        assert a.model_family == "sdxl"
        assert a.output_type == MediaType.IMAGE

    def test_constraints(self):
        c = InpaintAdapter().constraints()
        assert c.default_steps == 20
        assert c.default_cfg == 7.0

    def test_cost(self):
        assert InpaintAdapter().cost(_make_req(operation=Operation.INPAINT)) == 2

    def test_build_workflow(self):
        a = InpaintAdapter()
        req = _make_req(
            operation=Operation.INPAINT,
            input_images=["image.png"],
            input_mask="mask.png",
            prompt="fill with flowers",
            negative_prompt="ugly",
            steps=25,
            feathering=24,
        )
        wf = a.build_workflow(req)
        assert wf["1"]["class_type"] == "CheckpointLoaderSimple"
        assert wf["5"]["inputs"]["expand"] == 24
        assert wf["10"]["inputs"]["steps"] == 25

    def test_build_workflow_defaults(self):
        a = InpaintAdapter()
        req = _make_req(
            operation=Operation.INPAINT,
            input_images=["img.png"],
            input_mask="m.png",
        )
        wf = a.build_workflow(req)
        assert wf["10"]["inputs"]["denoise"] == 0.85
        assert wf["10"]["inputs"]["cfg"] == 7.0


# ── Image Caption ────────────────────────────────────────────


class TestImageCaption(unittest.TestCase):
    def test_identity(self):
        a = ImageCaptionAdapter()
        assert a.name == "local-caption-image"
        assert Operation.CAPTION in a.supported_ops
        assert MediaType.IMAGE in a.input_types
        assert a.output_type == MediaType.TEXT

    def test_cost(self):
        assert ImageCaptionAdapter().cost(_make_req(operation=Operation.CAPTION)) == 1

    def test_constraints(self):
        c = ImageCaptionAdapter().constraints()
        assert c.supports_negative_prompt is False


# ── Video Caption ────────────────────────────────────────────


class TestVideoCaption(unittest.TestCase):
    def test_identity(self):
        a = VideoCaptionAdapter()
        assert a.name == "local-caption-video"
        assert Operation.CAPTION in a.supported_ops
        assert MediaType.VIDEO in a.input_types
        assert a.output_type == MediaType.TEXT

    def test_cost(self):
        assert VideoCaptionAdapter().cost(_make_req(operation=Operation.CAPTION)) == 2


# ── Async execute tests ─────────────────────────────────────


class TestAsyncExecute(unittest.IsolatedAsyncioTestCase):
    async def test_lightning_i2v_execute(self):
        mock = MagicMock()
        mock.queue_prompt.return_value = "p-1"
        mock.build_enhanced_workflow.return_value = {"nodes": []}
        a = Wan22LocalI2VLightningAdapter(comfyui_client_fn=lambda: mock)
        req = _make_req(
            operation=Operation.GENERATE,
            target_type=MediaType.VIDEO,
            input_images=["img.png"],
            frames=41,
        )
        result = await a.execute(req)
        assert result.prompt_id == "p-1"
        assert result.status == "queued_local"

    async def test_i2i_execute_no_image_raises(self):
        a = I2ITransformAdapter(comfyui_client_fn=_mock_comfyui())
        req = _make_req(operation=Operation.TRANSFORM)
        with self.assertRaises(ValueError, msg="requires an input image"):
            await a.execute(req)

    async def test_v2v_execute_no_video_raises(self):
        a = V2VStyleTransferAdapter(comfyui_client_fn=_mock_comfyui())
        req = _make_req(operation=Operation.TRANSFORM, target_type=MediaType.VIDEO)
        with self.assertRaises(ValueError, msg="requires an input video"):
            await a.execute(req)

    async def test_upscale_image_execute(self):
        a = ImageUpscaleAdapter(comfyui_client_fn=_mock_comfyui())
        req = _make_req(
            operation=Operation.UPSCALE,
            input_images=["test.png"],
        )
        result = await a.execute(req)
        assert result.prompt_id == "prompt-1234"
        assert result.adapter_name == "local-upscale-image"

    async def test_upscale_image_no_input_raises(self):
        a = ImageUpscaleAdapter(comfyui_client_fn=_mock_comfyui())
        req = _make_req(operation=Operation.UPSCALE)
        with self.assertRaises(ValueError, msg="requires an input image"):
            await a.execute(req)

    async def test_upscale_video_no_input_raises(self):
        a = VideoUpscaleAdapter(comfyui_client_fn=_mock_comfyui())
        req = _make_req(operation=Operation.UPSCALE, target_type=MediaType.VIDEO)
        with self.assertRaises(ValueError, msg="requires an input video"):
            await a.execute(req)

    async def test_interpolate_execute(self):
        a = InterpolateAdapter(comfyui_client_fn=_mock_comfyui())
        req = _make_req(
            operation=Operation.INTERPOLATE,
            target_type=MediaType.VIDEO,
            input_video="vid.mp4",
        )
        result = await a.execute(req)
        assert result.adapter_name == "local-interpolate"

    async def test_interpolate_no_video_raises(self):
        a = InterpolateAdapter(comfyui_client_fn=_mock_comfyui())
        req = _make_req(operation=Operation.INTERPOLATE, target_type=MediaType.VIDEO)
        with self.assertRaises(ValueError, msg="requires an input video"):
            await a.execute(req)

    async def test_faceswap_image_execute(self):
        a = FaceSwapImageAdapter(face_service_fn=_mock_face_service())
        req = _make_req(
            operation=Operation.SWAP,
            input_images=["target.png", "source.png"],
        )
        result = await a.execute(req)
        assert result.status == "completed"

    async def test_faceswap_image_insufficient_images_raises(self):
        a = FaceSwapImageAdapter(face_service_fn=_mock_face_service())
        req = _make_req(operation=Operation.SWAP, input_images=["only_one.png"])
        with self.assertRaises(ValueError, msg="requires 2 input images"):
            await a.execute(req)

    async def test_faceswap_video_execute(self):
        a = FaceSwapVideoAdapter(face_service_fn=_mock_face_service())
        req = _make_req(
            operation=Operation.SWAP,
            target_type=MediaType.VIDEO,
            input_video="video.mp4",
            input_images=["source.png"],
        )
        result = await a.execute(req)
        assert result.status == "completed"

    async def test_faceswap_video_no_video_raises(self):
        a = FaceSwapVideoAdapter(face_service_fn=_mock_face_service())
        req = _make_req(operation=Operation.SWAP, target_type=MediaType.VIDEO, input_images=["s.png"])
        with self.assertRaises(ValueError, msg="requires an input video"):
            await a.execute(req)

    async def test_audio_execute(self):
        a = MMAudioAdapter(comfyui_client_fn=_mock_comfyui())
        req = _make_req(prompt="jazz music", audio_mode="music", duration=10.0)
        result = await a.execute(req)
        assert result.adapter_name == "local-mmaudio"

    async def test_voice_clone_execute(self):
        a = VoiceCloneAdapter(comfyui_client_fn=_mock_comfyui())
        req = _make_req(
            prompt="Hello world",
            voice_sample_path="/tmp/voice.wav",
        )
        result = await a.execute(req)
        assert result.adapter_name == "local-voice-clone"

    async def test_lipsync_execute(self):
        a = LipSyncAdapter(comfyui_client_fn=_mock_comfyui())
        req = _make_req(
            operation=Operation.LIPSYNC,
            target_type=MediaType.VIDEO,
            input_video="vid.mp4",
            input_audio="audio.wav",
        )
        result = await a.execute(req)
        assert result.adapter_name == "local-lipsync"

    async def test_lipsync_no_video_raises(self):
        a = LipSyncAdapter(comfyui_client_fn=_mock_comfyui())
        req = _make_req(operation=Operation.LIPSYNC, input_audio="a.wav")
        with self.assertRaises(ValueError, msg="requires an input video"):
            await a.execute(req)

    async def test_lipsync_no_audio_raises(self):
        a = LipSyncAdapter(comfyui_client_fn=_mock_comfyui())
        req = _make_req(operation=Operation.LIPSYNC, input_video="v.mp4")
        with self.assertRaises(ValueError, msg="requires input audio"):
            await a.execute(req)

    async def test_inpaint_execute(self):
        a = InpaintAdapter(comfyui_client_fn=_mock_comfyui())
        req = _make_req(
            operation=Operation.INPAINT,
            input_images=["img.png"],
            input_mask="mask.png",
            prompt="fill with flowers",
        )
        result = await a.execute(req)
        assert result.adapter_name == "local-inpaint"

    async def test_inpaint_no_image_raises(self):
        a = InpaintAdapter(comfyui_client_fn=_mock_comfyui())
        req = _make_req(operation=Operation.INPAINT, input_mask="m.png")
        with self.assertRaises(ValueError, msg="requires an input image"):
            await a.execute(req)

    async def test_inpaint_no_mask_raises(self):
        a = InpaintAdapter(comfyui_client_fn=_mock_comfyui())
        req = _make_req(operation=Operation.INPAINT, input_images=["i.png"])
        with self.assertRaises(ValueError, msg="requires a mask"):
            await a.execute(req)

    async def test_caption_image_execute(self):
        a = ImageCaptionAdapter(guardian_client_fn=_mock_guardian())
        req = _make_req(
            operation=Operation.CAPTION,
            target_type=MediaType.TEXT,
            input_images=["img.png"],
        )
        result = await a.execute(req)
        assert result.status == "completed"
        assert "caption" in result.meta

    async def test_caption_image_no_image_raises(self):
        a = ImageCaptionAdapter(guardian_client_fn=_mock_guardian())
        req = _make_req(operation=Operation.CAPTION, target_type=MediaType.TEXT)
        with self.assertRaises(ValueError, msg="requires an input image"):
            await a.execute(req)

    async def test_caption_video_execute(self):
        a = VideoCaptionAdapter(guardian_client_fn=_mock_guardian())
        req = _make_req(
            operation=Operation.CAPTION,
            target_type=MediaType.TEXT,
            input_video="vid.mp4",
        )
        result = await a.execute(req)
        assert result.status == "completed"

    async def test_caption_video_no_video_raises(self):
        a = VideoCaptionAdapter(guardian_client_fn=_mock_guardian())
        req = _make_req(operation=Operation.CAPTION, target_type=MediaType.TEXT)
        with self.assertRaises(ValueError, msg="requires an input video"):
            await a.execute(req)

    async def test_no_service_raises_runtime_error(self):
        """All adapters without service fn should raise RuntimeError."""
        adapters_and_reqs = [
            (I2ITransformAdapter(), _make_req(operation=Operation.TRANSFORM, input_images=["x"])),
            (V2VStyleTransferAdapter(), _make_req(operation=Operation.TRANSFORM, input_video="v")),
            (ImageUpscaleAdapter(), _make_req(operation=Operation.UPSCALE, input_images=["x"])),
            (VideoUpscaleAdapter(), _make_req(operation=Operation.UPSCALE, input_video="v")),
            (InterpolateAdapter(), _make_req(operation=Operation.INTERPOLATE, input_video="v")),
            (FaceSwapImageAdapter(), _make_req(operation=Operation.SWAP, input_images=["t", "s"])),
            (FaceSwapVideoAdapter(), _make_req(operation=Operation.SWAP, input_video="v", input_images=["s"])),
            (MMAudioAdapter(), _make_req(prompt="music")),
            (VoiceCloneAdapter(), _make_req(prompt="text")),
            (LipSyncAdapter(), _make_req(operation=Operation.LIPSYNC, input_video="v", input_audio="a")),
            (InpaintAdapter(), _make_req(operation=Operation.INPAINT, input_images=["i"], input_mask="m")),
            (ImageCaptionAdapter(), _make_req(operation=Operation.CAPTION, input_images=["i"])),
            (VideoCaptionAdapter(), _make_req(operation=Operation.CAPTION, input_video="v")),
        ]
        for adapter, req in adapters_and_reqs:
            with self.assertRaises(RuntimeError, msg=f"{adapter.name} should raise RuntimeError"):
                await adapter.execute(req)


# ── Registry integration ─────────────────────────────────────


class TestAllAdaptersRegistrable(unittest.TestCase):
    """Verify all new adapters produce valid to_dict() output."""

    def test_all_to_dict(self):
        adapters = [
            Wan22LocalI2VLightningAdapter(),
            I2ITransformAdapter(),
            V2VStyleTransferAdapter(),
            ImageUpscaleAdapter(),
            VideoUpscaleAdapter(),
            InterpolateAdapter(),
            FaceSwapImageAdapter(),
            FaceSwapVideoAdapter(),
            MMAudioAdapter(),
            VoiceCloneAdapter(),
            LipSyncAdapter(),
            InpaintAdapter(),
            ImageCaptionAdapter(),
            VideoCaptionAdapter(),
        ]
        names = set()
        for a in adapters:
            d = a.to_dict()
            assert "name" in d
            assert "constraints" in d
            assert d["name"] not in names, f"Duplicate adapter name: {d['name']}"
            names.add(d["name"])
        assert len(names) == 14


if __name__ == "__main__":
    unittest.main()
