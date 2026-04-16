"""Local ComfyUI-based generation adapters."""

from .t2i_sdxl import SDXLLocalT2IAdapter
from .t2i_flux import FluxLocalT2IAdapter
from .t2i_sd15 import SD15LocalT2IAdapter
from .t2i_wan22 import Wan22LocalT2IAdapter
from .i2v_wan22 import (
    Wan22LocalI2VQ6Adapter,
    Wan22LocalI2VDisTorch2Adapter,
    Wan22LocalI2VBlockSwapAdapter,
    Wan22LocalI2VUltraAdapter,
)
from .i2v_wan22_lightning import Wan22LocalI2VLightningAdapter
from .t2v_wan22 import Wan22LocalT2VQ6Adapter
from .i2i_transform import I2ITransformAdapter
from .v2v import V2VStyleTransferAdapter
from .upscale_image import ImageUpscaleAdapter
from .upscale_video import VideoUpscaleAdapter
from .interpolate import InterpolateAdapter
from .face_swap import FaceSwapImageAdapter
from .face_swap_video import FaceSwapVideoAdapter
from .audio_mmaudio import MMAudioAdapter
from .voice_clone import VoiceCloneAdapter
from .lipsync import LipSyncAdapter
from .inpaint import InpaintAdapter
from .caption_image import ImageCaptionAdapter
from .caption_video import VideoCaptionAdapter

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
