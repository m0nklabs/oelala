"""
Unified Generation Core — adapter-based architecture for all generation tools.

This package provides:
- types: MediaType, Operation, ComputeTarget, LoraFormat, AdapterConstraints,
         GenerationRequest, GenerationResult, LoraStackItem
- adapter: GenerationAdapter ABC
- registry: AdapterRegistry (register/find/list adapters)
- router: GenerationRouter (dispatch requests to adapters)
- lora_utils: LoRA resolution, filtering, sanitization helpers
"""

from .adapter import GenerationAdapter
from .factory import create_registry
from .registry import AdapterRegistry
from .router import GenerationRouter, normalize_frame_count, resolve_resolution
from .types import (
    AdapterConstraints,
    ComputeTarget,
    GenerationRequest,
    GenerationResult,
    LoraFormat,
    LoraStackItem,
    MediaType,
    Operation,
)
from .v1_compat import (
    dispatch_v1,
    form_to_generation_request,
    generation_result_to_v1_response,
    init_v1_compat,
)

__all__ = [
    "AdapterConstraints",
    "AdapterRegistry",
    "ComputeTarget",
    "GenerationAdapter",
    "GenerationRequest",
    "GenerationResult",
    "GenerationRouter",
    "LoraFormat",
    "LoraStackItem",
    "MediaType",
    "Operation",
    "create_registry",
    "dispatch_v1",
    "form_to_generation_request",
    "generation_result_to_v1_response",
    "init_v1_compat",
    "normalize_frame_count",
    "resolve_resolution",
]
