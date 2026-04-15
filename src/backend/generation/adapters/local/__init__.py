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
from generation.adapters.local.t2v_wan22 import Wan22LocalT2VQ6Adapter

__all__ = [
    "SDXLLocalT2IAdapter",
    "FluxLocalT2IAdapter",
    "SD15LocalT2IAdapter",
    "Wan22LocalT2IAdapter",
    "Wan22LocalI2VQ6Adapter",
    "Wan22LocalI2VDisTorch2Adapter",
    "Wan22LocalI2VBlockSwapAdapter",
    "Wan22LocalI2VUltraAdapter",
    "Wan22LocalT2VQ6Adapter",
]
