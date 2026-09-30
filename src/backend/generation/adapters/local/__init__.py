"""Local ComfyUI-based generation adapters."""

from .t2i_sdxl import SDXLLocalT2IAdapter
from .t2i_flux import FluxLocalT2IAdapter
from .t2i_krea2 import Krea2LocalT2IAdapter
from .t2i_flux2 import Flux2LocalT2IAdapter
from .minimax_h3_t2v import MiniMaxH3LocalT2VAdapter
from .minimax_h3_i2v import MiniMaxH3LocalI2VAdapter
from .i2i_transform import I2ITransformAdapter
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
    "Krea2LocalT2IAdapter",
    "Flux2LocalT2IAdapter",
    "MiniMaxH3LocalT2VAdapter",
    "MiniMaxH3LocalI2VAdapter",
    "I2ITransformAdapter",
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
