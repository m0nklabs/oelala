"""
Adapter factory — creates and registers all adapters with real dependencies.

Called once at app startup. Injects production dependencies
(ComfyUI client, RunPod client, face service, Guardian base URL)
into each adapter.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .registry import AdapterRegistry

logger = logging.getLogger(__name__)


def create_registry(
    *,
    comfyui_client_fn: Any = None,
    submit_to_runpod_fn: Any = None,
    face_service_mod: Any = None,
    guardian_base_url: Optional[str] = None,
    runpod_endpoint_wan22: Optional[str] = None,
    runpod_endpoint_ltx23: Optional[str] = None,
    runpod_endpoint_i2i: Optional[str] = None,
) -> AdapterRegistry:
    """
    Build an AdapterRegistry with all adapters wired to real dependencies.

    Each adapter receives only the dependencies it needs via constructor
    injection (callables / function references).

    Args:
        comfyui_client_fn: Callable returning ComfyUIClient singleton.
        submit_to_runpod_fn: Async function for RunPod job submission.
        face_service_mod: The face_service module (InsightFace).
        guardian_base_url: Guardian LLM base URL for captioning.
        runpod_endpoint_wan22: RunPod endpoint ID for Wan2.2 jobs.
        runpod_endpoint_ltx23: RunPod endpoint ID for LTX-2.3 jobs.
        runpod_endpoint_i2i: RunPod endpoint ID for I2I edit jobs.

    Returns:
        Fully populated AdapterRegistry.
    """
    registry = AdapterRegistry()
    registered = 0
    skipped = 0

    def _register(adapter_cls, **kwargs):
        nonlocal registered, skipped
        try:
            adapter = adapter_cls(**kwargs)
            registry.register(adapter)
            registered += 1
        except Exception as e:
            logger.warning(f"⚠️ Skipped {adapter_cls.__name__}: {e}")
            skipped += 1

    # ── Cloud adapters ──────────────────────────────────────────
    if submit_to_runpod_fn:
        from .adapters.cloud.wan22_i2v import Wan22CloudI2VAdapter
        from .adapters.cloud.wan22_t2v import Wan22CloudT2VAdapter
        from .adapters.cloud.ltx23_i2v import LTX23CloudI2VAdapter
        from .adapters.cloud.ltx23_t2v import LTX23CloudT2VAdapter
        from .adapters.cloud.cloud_i2i import (
            I2IEditCloudAdapter,
            CloudI2ITransformAdapter,
        )

        _register(
            Wan22CloudI2VAdapter,
            submit_to_runpod_fn=submit_to_runpod_fn,
            comfyui_client_fn=comfyui_client_fn,
            endpoint_id=runpod_endpoint_wan22,
        )
        _register(
            Wan22CloudT2VAdapter,
            submit_to_runpod_fn=submit_to_runpod_fn,
            comfyui_client_fn=comfyui_client_fn,
            endpoint_id=runpod_endpoint_wan22,
        )
        _register(
            LTX23CloudI2VAdapter,
            submit_to_runpod_fn=submit_to_runpod_fn,
            comfyui_client_fn=comfyui_client_fn,
        )
        _register(
            LTX23CloudT2VAdapter,
            submit_to_runpod_fn=submit_to_runpod_fn,
            comfyui_client_fn=comfyui_client_fn,
        )
        _register(
            I2IEditCloudAdapter,
            submit_to_runpod_fn=submit_to_runpod_fn,
        )
        _register(
            CloudI2ITransformAdapter,
            submit_to_runpod_fn=submit_to_runpod_fn,
        )
    else:
        logger.info("☁️ RunPod not available — skipping cloud adapters")

    # ── Local T2I adapters ──────────────────────────────────────
    if comfyui_client_fn:
        from .adapters.local.t2i_sdxl import SDXLLocalT2IAdapter
        from .adapters.local.t2i_flux import FluxLocalT2IAdapter
        from .adapters.local.t2i_sd15 import SD15LocalT2IAdapter
        from .adapters.local.t2i_wan22 import Wan22LocalT2IAdapter
        from .adapters.local.t2i_ernie import ErnieLocalT2IAdapter

        _register(SDXLLocalT2IAdapter, comfyui_client_fn=comfyui_client_fn)
        _register(FluxLocalT2IAdapter, comfyui_client_fn=comfyui_client_fn)
        _register(SD15LocalT2IAdapter, comfyui_client_fn=comfyui_client_fn)
        _register(Wan22LocalT2IAdapter, comfyui_client_fn=comfyui_client_fn)
        _register(ErnieLocalT2IAdapter, comfyui_client_fn=comfyui_client_fn)

        # ── Local I2V adapters ──────────────────────────────────
        from .adapters.local.i2v_wan22 import (
            Wan22LocalI2VQ6Adapter,
            Wan22LocalI2VDisTorch2Adapter,
        )
        from .adapters.local.i2v_wan22_lightning import Wan22LocalI2VLightningAdapter

        _register(Wan22LocalI2VQ6Adapter, comfyui_client_fn=comfyui_client_fn)
        _register(Wan22LocalI2VDisTorch2Adapter, comfyui_client_fn=comfyui_client_fn)
        _register(Wan22LocalI2VLightningAdapter, comfyui_client_fn=comfyui_client_fn)

        # ── Local T2V adapter ──────────────────────────────────
        from .adapters.local.t2v_wan22 import Wan22LocalT2VQ6Adapter

        _register(Wan22LocalT2VQ6Adapter, comfyui_client_fn=comfyui_client_fn)

        # ── Utility adapters ───────────────────────────────────
        from .adapters.local.i2i_transform import I2ITransformAdapter
        from .adapters.local.v2v import V2VStyleTransferAdapter
        from .adapters.local.upscale_image import ImageUpscaleAdapter
        from .adapters.local.upscale_video import VideoUpscaleAdapter
        from .adapters.local.interpolate import InterpolateAdapter
        from .adapters.local.inpaint import InpaintAdapter
        from .adapters.local.lipsync import LipSyncAdapter
        from .adapters.local.audio_mmaudio import MMAudioAdapter
        from .adapters.local.voice_clone import VoiceCloneAdapter

        _register(I2ITransformAdapter, comfyui_client_fn=comfyui_client_fn)
        _register(V2VStyleTransferAdapter, comfyui_client_fn=comfyui_client_fn)
        _register(ImageUpscaleAdapter, comfyui_client_fn=comfyui_client_fn)
        _register(VideoUpscaleAdapter, comfyui_client_fn=comfyui_client_fn)
        _register(InterpolateAdapter, comfyui_client_fn=comfyui_client_fn)
        _register(InpaintAdapter, comfyui_client_fn=comfyui_client_fn)
        _register(LipSyncAdapter, comfyui_client_fn=comfyui_client_fn)
        _register(MMAudioAdapter, comfyui_client_fn=comfyui_client_fn)
        _register(VoiceCloneAdapter, comfyui_client_fn=comfyui_client_fn)
    else:
        logger.warning("⚠️ ComfyUI not available — skipping all local adapters")

    # ── Face adapters (optional dep) ────────────────────────────
    if face_service_mod:
        from .adapters.local.face_swap import FaceSwapImageAdapter
        from .adapters.local.face_swap_video import FaceSwapVideoAdapter

        _register(FaceSwapImageAdapter, face_service_fn=face_service_mod)
        _register(FaceSwapVideoAdapter, face_service_fn=face_service_mod)
    else:
        logger.info("👤 face_service not available — skipping face swap adapters")

    # ── Caption adapters (Guardian LLM) ─────────────────────────
    guardian_url = guardian_base_url or os.getenv(
        "GUARDIAN_BASE_URL",
        os.getenv("GUARDIAN_BASE", os.getenv("OLLAMA_BASE", "http://localhost:11434")),
    )
    from .adapters.local.caption_image import ImageCaptionAdapter
    from .adapters.local.caption_video import VideoCaptionAdapter

    _register(ImageCaptionAdapter, guardian_client_fn=guardian_url)
    _register(VideoCaptionAdapter, guardian_client_fn=guardian_url)

    logger.info(
        f"🏭 Adapter factory complete: {registered} registered, {skipped} skipped"
    )
    return registry
