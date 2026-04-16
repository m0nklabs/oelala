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
from .router import GenerationRouter
from .factory import create_registry

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
    "create_registry",
]
