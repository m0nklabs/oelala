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

from generation.types import (
    MediaType,
    Operation,
    ComputeTarget,
    LoraFormat,
    AdapterConstraints,
    GenerationRequest,
    GenerationResult,
    LoraStackItem,
)
from generation.adapter import GenerationAdapter
from generation.registry import AdapterRegistry
from generation.router import GenerationRouter

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
]
