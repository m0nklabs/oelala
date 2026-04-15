"""Local ComfyUI-based generation adapters."""

from generation.adapters.local.t2i_sdxl import SDXLLocalT2IAdapter
from generation.adapters.local.t2i_flux import FluxLocalT2IAdapter
from generation.adapters.local.t2i_sd15 import SD15LocalT2IAdapter
from generation.adapters.local.t2i_wan22 import Wan22LocalT2IAdapter

__all__ = [
    "SDXLLocalT2IAdapter",
    "FluxLocalT2IAdapter",
    "SD15LocalT2IAdapter",
    "Wan22LocalT2IAdapter",
]
