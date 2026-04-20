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

from .types import (
    MediaType,
    Operation,
    ComputeTarget,
    LoraFormat,
    AdapterConstraints,
    GenerationRequest,
    GenerationResult,
    LoraStackItem,
)
from .adapter import GenerationAdapter
from .registry import AdapterRegistry
from .router import GenerationRouter, resolve_resolution, normalize_frame_count
from .factory import create_registry
from .v1_compat import (
    form_to_generation_request,
    generation_result_to_v1_response,
    dispatch_v1,
    init_v1_compat,
)

__all__ = [
    "MediaType",
    "Operation",
    "ComputeTarget",
    "LoraFormat",
    "AdapterConstraints",
    "GenerationRequest",
    "GenerationResult",
    "LoraStackItem",
    "GenerationAdapter",
    "AdapterRegistry",
    "GenerationRouter",
    "resolve_resolution",
    "normalize_frame_count",
    "create_registry",
    "form_to_generation_request",
    "generation_result_to_v1_response",
    "dispatch_v1",
    "init_v1_compat",
]
