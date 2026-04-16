"""Cloud RunPod-based generation adapters."""

from .qwen_edit import QwenEditCloudAdapter
from .wan22_i2v import Wan22CloudI2VAdapter
from .wan22_t2v import Wan22CloudT2VAdapter
from .ltx23_i2v import LTX23CloudI2VAdapter
from .ltx23_t2v import LTX23CloudT2VAdapter

__all__ = [
    "QwenEditCloudAdapter",
    "Wan22CloudI2VAdapter",
    "Wan22CloudT2VAdapter",
    "LTX23CloudI2VAdapter",
    "LTX23CloudT2VAdapter",
]
