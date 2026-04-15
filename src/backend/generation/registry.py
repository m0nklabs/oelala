"""
AdapterRegistry — singleton registry of all generation adapters.

Supports registration, lookup by criteria, and listing all adapters.
"""

from __future__ import annotations

import logging
from typing import Optional

from generation.adapter import GenerationAdapter
from generation.types import ComputeTarget, MediaType, Operation

logger = logging.getLogger(__name__)


class AdapterRegistry:
    """Thread-safe registry for generation adapters."""

    def __init__(self) -> None:
        self._adapters: dict[str, GenerationAdapter] = {}

    def register(self, adapter: GenerationAdapter) -> None:
        """Register an adapter. Raises ValueError on duplicate name."""
        if adapter.name in self._adapters:
            raise ValueError(f"Adapter '{adapter.name}' already registered")
        self._adapters[adapter.name] = adapter
        logger.info(f"📦 Registered adapter: {adapter.name}")

    def get(self, name: str) -> Optional[GenerationAdapter]:
        """Look up an adapter by exact name."""
        return self._adapters.get(name)

    def find(
        self,
        operation: Optional[Operation] = None,
        input_type: Optional[MediaType] = None,
        target_type: Optional[MediaType] = None,
        compute: Optional[ComputeTarget] = None,
    ) -> list[GenerationAdapter]:
        """
        Find adapters matching the given criteria.

        All parameters are optional filters — only non-None values are checked.
        Returns a list of matching adapters (may be empty).
        """
        results: list[GenerationAdapter] = []
        for adapter in self._adapters.values():
            if operation is not None and operation not in adapter.supported_ops:
                continue
            if input_type is not None and input_type not in adapter.input_types:
                continue
            if target_type is not None and adapter.output_type != target_type:
                continue
            if compute is not None and compute != ComputeTarget.AUTO:
                if adapter.compute != compute:
                    continue
            results.append(adapter)
        return results

    def list_all(self) -> list[GenerationAdapter]:
        """Return all registered adapters."""
        return list(self._adapters.values())

    def __len__(self) -> int:
        return len(self._adapters)

    def __contains__(self, name: str) -> bool:
        return name in self._adapters
