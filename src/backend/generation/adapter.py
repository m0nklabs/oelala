"""
GenerationAdapter — abstract base class for all generation adapters.

Each adapter represents one model/compute/operation combination.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Callable, Awaitable

from generation.types import (
    AdapterConstraints,
    ComputeTarget,
    GenerationRequest,
    GenerationResult,
    LoraFormat,
    MediaType,
    Operation,
)

# Type alias for optional progress callback wired by the router
ProgressCallback = Optional[Callable[[str, float, str], Awaitable[None]]]


class GenerationAdapter(ABC):
    """
    One adapter per model/compute/operation combination.

    Subclasses MUST set the class-level attributes and implement
    the four abstract methods.
    """

    # ── Identity (set in subclass) ──────────────────────────────
    name: str  # e.g. "wan22-distorch2-q6-i2v"
    model_family: str  # e.g. "wan2.2" — used for LoRA compat filtering
    supported_ops: set[Operation]
    input_types: set[MediaType]
    output_type: MediaType
    compute: ComputeTarget
    lora_format: LoraFormat

    # ── Abstract interface ──────────────────────────────────────

    @abstractmethod
    def constraints(self) -> AdapterConstraints:
        """Declares what this adapter supports — used by router AND frontend."""
        ...

    @abstractmethod
    def build_workflow(self, req: GenerationRequest) -> dict:
        """Build ComfyUI API-format workflow JSON (or RunPod payload)."""
        ...

    @abstractmethod
    def cost(self, req: GenerationRequest) -> int:
        """Calculate credit cost for this specific request."""
        ...

    @abstractmethod
    async def execute(
        self,
        req: GenerationRequest,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        """
        Queue locally or submit to RunPod.

        Returns a GenerationResult with tracking info.
        The router handles credit checks/deductions and media upload.
        """
        ...

    # ── Convenience helpers ─────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize adapter metadata for the /v2/adapters endpoint."""
        return {
            "name": self.name,
            "model_family": self.model_family,
            "supported_ops": sorted(op.value for op in self.supported_ops),
            "input_types": sorted(t.value for t in self.input_types),
            "output_type": self.output_type.value,
            "compute": self.compute.value,
            "lora_format": self.lora_format.value,
            "constraints": self.constraints().model_dump(),
        }
