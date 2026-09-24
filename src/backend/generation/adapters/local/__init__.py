"""Local ComfyUI-based generation adapters."""

from .audio_mmaudio import MMAudioAdapter
from .caption_image import ImageCaptionAdapter
from .caption_video import VideoCaptionAdapter
from .face_swap import FaceSwapImageAdapter
from .face_swap_video import FaceSwapVideoAdapter
from .i2i_transform import I2ITransformAdapter
from .i2v_wan22 import (
    Wan22LocalI2VBlockSwapAdapter,
    Wan22LocalI2VDisTorch2Adapter,
    Wan22LocalI2VQ6Adapter,
    Wan22LocalI2VUltraAdapter,
)
from .i2v_wan22_lightning import Wan22LocalI2VLightningAdapter
from .inpaint import InpaintAdapter
from .interpolate import InterpolateAdapter
from .lipsync import LipSyncAdapter
from .minimax_h3_i2v import MiniMaxH3LocalI2VAdapter
from .minimax_h3_t2v import MiniMaxH3LocalT2VAdapter
from .t2i_flux import FluxLocalT2IAdapter
from .t2i_flux2 import Flux2LocalT2IAdapter
from .t2i_krea2 import Krea2LocalT2IAdapter
from .t2i_sdxl import SDXLLocalT2IAdapter
from .t2v_wan22 import Wan22LocalT2VQ6Adapter
from .upscale_image import ImageUpscaleAdapter
from .upscale_video import VideoUpscaleAdapter
from .v2v import V2VStyleTransferAdapter
from .voice_clone import VoiceCloneAdapter

__all__ = [
    "FaceSwapImageAdapter",
    "FaceSwapVideoAdapter",
    "Flux2LocalT2IAdapter",
    "FluxLocalT2IAdapter",
    "I2ITransformAdapter",
    "ImageCaptionAdapter",
    "ImageUpscaleAdapter",
    "InpaintAdapter",
    "InterpolateAdapter",
    "Krea2LocalT2IAdapter",
    "LipSyncAdapter",
    "MMAudioAdapter",
    "MiniMaxH3LocalI2VAdapter",
    "MiniMaxH3LocalT2VAdapter",
    "SDXLLocalT2IAdapter",
    "V2VStyleTransferAdapter",
    "VideoCaptionAdapter",
    "VideoUpscaleAdapter",
    "VoiceCloneAdapter",
    "Wan22LocalI2VBlockSwapAdapter",
    "Wan22LocalI2VDisTorch2Adapter",
    "Wan22LocalI2VLightningAdapter",
    "Wan22LocalI2VQ6Adapter",
    "Wan22LocalI2VUltraAdapter",
    "Wan22LocalT2VQ6Adapter",
]
