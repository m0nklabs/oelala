"""Cloud RunPod-based generation adapters."""

from generation.adapters.cloud.qwen_edit import QwenEditCloudAdapter
from generation.adapters.cloud.wan22_i2v import Wan22CloudI2VAdapter
from generation.adapters.cloud.wan22_t2v import Wan22CloudT2VAdapter
from generation.adapters.cloud.ltx23_i2v import LTX23CloudI2VAdapter
from generation.adapters.cloud.ltx23_t2v import LTX23CloudT2VAdapter

__all__ = [
    "QwenEditCloudAdapter",
    "Wan22CloudI2VAdapter",
    "Wan22CloudT2VAdapter",
    "LTX23CloudI2VAdapter",
    "LTX23CloudT2VAdapter",
]
