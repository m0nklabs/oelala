"""
GenerationRouter — dispatches GenerationRequests to the correct adapter.

Handles:
1. Adapter resolution (by hint or auto-match)
2. Control validation against adapter.constraints()
3. LoRA filtering by model compatibility
4. Credit check + deduction
5. Adapter execution
6. Job tracking
"""

from __future__ import annotations

import logging
import random
from typing import Any

from generation.adapter import GenerationAdapter, ProgressCallback
from generation.registry import AdapterRegistry
from generation.types import (
    ComputeTarget,
    GenerationRequest,
    GenerationResult,
    LoraFormat,
    MediaType,
)
from generation import lora_utils

logger = logging.getLogger(__name__)


class GenerationRouter:
    """
    Central dispatch point for all generation requests.

    The router is stateless — all state lives in the registry,
    the credit system, and the job queue.
    """

    def __init__(self, registry: AdapterRegistry) -> None:
        self.registry = registry

    def resolve_adapter(self, req: GenerationRequest) -> GenerationAdapter:
        """
        Find the best adapter for a request.

        Priority:
        1. adapter_hint (exact name match)
        2. Auto-match by (operation, input_types, target_type)
        """
        # 1. Explicit adapter hint
        if req.adapter_hint:
            adapter = self.registry.get(req.adapter_hint)
            if adapter is None:
                raise ValueError(
                    f"Requested adapter '{req.adapter_hint}' not found. "
                    f"Available: {[a.name for a in self.registry.list_all()]}"
                )
            return adapter

        # 2. Auto-match
        # Determine input type from request content
        if req.input_images:
            input_type = MediaType.IMAGE
        elif req.input_video:
            input_type = MediaType.VIDEO
        elif req.input_audio:
            input_type = MediaType.AUDIO
        else:
            input_type = MediaType.TEXT

        candidates = self.registry.find(
            operation=req.operation,
            input_type=input_type,
            target_type=req.target_type,
        )

        if not candidates:
            raise ValueError(
                f"No adapter found for operation={req.operation.value}, "
                f"input={input_type.value if input_type else 'none'}, "
                f"target={req.target_type.value}"
            )

        # If multiple candidates, prefer local over cloud (unless cloud requested)
        if len(candidates) > 1:
            local = [a for a in candidates if a.compute == ComputeTarget.LOCAL]
            cloud = [a for a in candidates if a.compute == ComputeTarget.CLOUD]
            # Return first match (local preferred)
            return local[0] if local else cloud[0]

        return candidates[0]

    def validate_controls(
        self, req: GenerationRequest, adapter: GenerationAdapter
    ) -> GenerationRequest:
        """
        Validate and clamp request controls against adapter constraints.

        Returns a (possibly modified) request with defaults applied.
        """
        c = adapter.constraints()

        # Apply default steps/cfg if not specified
        if req.steps is None:
            req = req.model_copy(update={"steps": c.default_steps})
        if req.cfg is None:
            req = req.model_copy(update={"cfg": c.default_cfg})

        # Clamp resolution to adapter limits and step
        updates: dict[str, Any] = {}
        if req.width is not None:
            w = max(c.min_width, min(c.max_width, req.width))
            w = (w // c.resolution_step) * c.resolution_step
            updates["width"] = w
        if req.height is not None:
            h = max(c.min_height, min(c.max_height, req.height))
            h = (h // c.resolution_step) * c.resolution_step
            updates["height"] = h

        # Clamp steps
        if req.steps is not None:
            updates["steps"] = max(c.min_steps, min(c.max_steps, req.steps))

        # Generate random seed if -1
        if req.seed == -1:
            updates["seed"] = random.randint(0, 2**32 - 1)

        if updates:
            req = req.model_copy(update=updates)

        return req

    def filter_loras(
        self, req: GenerationRequest, adapter: GenerationAdapter
    ) -> GenerationRequest:
        """
        Filter LoRAs for model compatibility and enforce max count.

        Delegates to lora_utils for the actual filtering logic.
        """
        if not req.loras:
            return req

        # Convert LoraStackItem models to dicts for existing helper functions
        lora_dicts = [lr.model_dump(exclude_none=True) for lr in req.loras]

        # For single-stage adapters, sanitize dual-stage configs
        if adapter.lora_format == LoraFormat.SINGLE_STAGE:
            lora_dicts = lora_utils.sanitize_lora_configs_for_single_stage(lora_dicts)

        # Filter by model compatibility
        lora_dicts = lora_utils.filter_loras_by_model_compat(
            lora_dicts, adapter.model_family
        )

        # Enforce max LoRAs
        c = adapter.constraints()
        if len(lora_dicts) > c.max_loras:
            logger.warning(
                f"⚠️ Trimming LoRA stack from {len(lora_dicts)} to {c.max_loras}"
            )
            lora_dicts = lora_dicts[: c.max_loras]

        # Convert back to LoraStackItem
        from generation.types import LoraStackItem

        filtered_loras = [LoraStackItem(**d) for d in lora_dicts]
        return req.model_copy(update={"loras": filtered_loras})

    async def dispatch(
        self,
        req: GenerationRequest,
        user: Any,
        *,
        check_credits_fn: Any = None,
        deduct_credits_fn: Any = None,
        progress_callback: ProgressCallback = None,
    ) -> GenerationResult:
        """
        Full dispatch pipeline:
        1. Resolve adapter
        2. Validate controls
        3. Filter LoRAs
        4. Calculate + check credits
        5. Execute adapter
        6. Deduct credits
        7. Return result

        Args:
            req: The generation request
            user: Authenticated user object (from auth.User)
            check_credits_fn: async fn(user, amount) — raises on insufficient
            deduct_credits_fn: async fn(user, amount, job_id, description) -> bool
            progress_callback: Optional WebSocket progress callback
        """
        # 1. Resolve adapter
        adapter = self.resolve_adapter(req)
        logger.info(f"🎯 Resolved adapter: {adapter.name}")

        # 2. Validate controls
        req = self.validate_controls(req, adapter)

        # 3. Filter LoRAs
        req = self.filter_loras(req, adapter)

        # 4. Calculate + check credits
        credits_required = adapter.cost(req)
        if check_credits_fn:
            await check_credits_fn(user, credits_required)

        # 5. Execute
        result = await adapter.execute(req, progress_callback=progress_callback)
        result = result.model_copy(
            update={"credits_used": credits_required, "adapter_name": adapter.name}
        )

        # 6. Deduct credits
        if deduct_credits_fn:
            await deduct_credits_fn(
                user, credits_required, result.prompt_id, adapter.name
            )

        logger.info(
            f"✅ Dispatched via {adapter.name}: prompt_id={result.prompt_id}, "
            f"credits={credits_required}"
        )

        return result
