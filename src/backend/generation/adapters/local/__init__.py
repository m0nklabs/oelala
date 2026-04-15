"""Local ComfyUI-based generation adapters."""

from generation.adapters.local.t2i_sdxl import SDXLLocalT2IAdapter
from generation.adapters.local.t2i_flux import FluxLocalT2IAdapter
from generation.adapters.local.t2i_sd15 import SD15LocalT2IAdapter
from generation.adapters.local.t2i_wan22 import Wan22LocalT2IAdapter
from generation.adapters.local.i2v_wan22 import (
    Wan22LocalI2VQ6Adapter,
    Wan22LocalI2VDisTorch2Adapter,
    Wan22LocalI2VBlockSwapAdapter,
    Wan22LocalI2VUltraAdapter,
)
from generation.adapters.local.i2v_wan22_lightning import Wan22LocalI2VLightningAdapter
from generation.adapters.local.t2v_wan22 import Wan22LocalT2VQ6Adapter
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

__all__ = [
    "SDXLLocalT2IAdapter",
    "FluxLocalT2IAdapter",
    "SD15LocalT2IAdapter",
    "Wan22LocalT2IAdapter",
    "Wan22LocalI2VQ6Adapter",
    "Wan22LocalI2VDisTorch2Adapter",
    "Wan22LocalI2VBlockSwapAdapter",
    "Wan22LocalI2VUltraAdapter",
    "Wan22LocalI2VLightningAdapter",
    "Wan22LocalT2VQ6Adapter",
    "I2ITransformAdapter",
    "V2VStyleTransferAdapter",
    "ImageUpscaleAdapter",
    "VideoUpscaleAdapter",
    "InterpolateAdapter",
    "FaceSwapImageAdapter",
    "FaceSwapVideoAdapter",
    "MMAudioAdapter",
    "VoiceCloneAdapter",
    "LipSyncAdapter",
    "InpaintAdapter",
    "ImageCaptionAdapter",
    "VideoCaptionAdapter",
]
