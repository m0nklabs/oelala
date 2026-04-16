"""
V2 Generation API — unified endpoints for all generation tools.

Endpoints:
  POST /v2/generate   — dispatch a GenerationRequest to the correct adapter
  GET  /v2/adapters   — list all registered adapters with constraints
  POST /v2/estimate   — estimate credit cost for a request (without executing)
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from generation.types import GenerationRequest, GenerationResult
from generation.registry import AdapterRegistry
from generation.router import GenerationRouter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v2", tags=["v2-generation"])

# ── Module-level state (set by init_v2_api) ─────────────────────────
_registry: AdapterRegistry | None = None
_router: GenerationRouter | None = None
_check_credits_fn: Any = None
_deduct_credits_fn: Any = None
_get_current_user_fn: Any = None


def init_v2_api(
    registry: AdapterRegistry,
    gen_router: GenerationRouter,
    get_current_user: Any,
    check_credits: Any = None,
    deduct_credits: Any = None,
) -> None:
    """
    Initialize the v2 API with dependencies from app.py.

    Called once at startup after all adapters are registered.
    """
    global \
        _registry, \
        _router, \
        _check_credits_fn, \
        _deduct_credits_fn, \
        _get_current_user_fn
    _registry = registry
    _router = gen_router
    _check_credits_fn = check_credits
    _deduct_credits_fn = deduct_credits
    _get_current_user_fn = get_current_user
    logger.info(f"🚀 V2 Generation API initialized ({len(registry)} adapters)")


async def _resolve_user():
    """Resolve the current user via the injected auth dependency."""
    if _get_current_user_fn is not None:
        try:
            return await _get_current_user_fn()
        except Exception:
            return None
    return None


@router.post("/generate", response_model=GenerationResult)
async def v2_generate(
    req: GenerationRequest,
    user: Any = Depends(_resolve_user),
):
    """
    Unified generation endpoint.

    Accepts a GenerationRequest JSON body, resolves the adapter,
    validates controls, checks credits, and dispatches.
    """
    if _router is None:
        raise HTTPException(status_code=503, detail="V2 API not initialized")

    try:
        result = await _router.dispatch(
            req,
            user,
            check_credits_fn=_check_credits_fn,
            deduct_credits_fn=_deduct_credits_fn,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ V2 generate error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/adapters")
async def v2_list_adapters():
    """
    List all registered adapters with their constraints.

    Returns metadata about each adapter including supported operations,
    input/output types, compute target, LoRA format, and constraints.
    Used by the frontend for dynamic UI rendering.
    """
    if _registry is None:
        raise HTTPException(status_code=503, detail="V2 API not initialized")

    adapters = _registry.list_all()
    return {
        "adapters": [a.to_dict() for a in adapters],
        "count": len(adapters),
    }


@router.post("/estimate")
async def v2_estimate(req: GenerationRequest):
    """
    Estimate credit cost for a request without executing it.

    Returns the resolved adapter and credit cost.
    """
    if _router is None:
        raise HTTPException(status_code=503, detail="V2 API not initialized")

    try:
        adapter = _router.resolve_adapter(req)
        validated_req = _router.validate_controls(req, adapter)
        cost = adapter.cost(validated_req)
        return {
            "adapter": adapter.name,
            "credits_required": cost,
            "constraints": adapter.constraints().model_dump(),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
