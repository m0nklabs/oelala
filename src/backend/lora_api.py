"""
LoRA Browser API — Endpoints for browsing and searching LoRA models.

Provides:
- GET /api/loras — List all LoRAs (with optional search/filter)
- GET /api/loras/categories — Get available categories
- GET /api/loras/tags — Get available tags
- GET /api/loras/{lora_id} — Get LoRA details
- POST /api/loras/refresh — Force re-scan of LoRA directories
"""

import logging
import os
from typing import List, Optional

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from lora_scanner import lora_cache

logger = logging.getLogger("lora_api")

DEBUG = os.getenv("DEBUG", "").lower() in ("1", "true", "yes")


def debug_log(msg: str):
    if DEBUG:
        logger.info(f"🎨 [LoRA API] {msg}")


router = APIRouter(prefix="/api/loras", tags=["loras"])


# =============================================================================
# Response Models
# =============================================================================


class LoRAItem(BaseModel):
    id: str
    filename: str
    name: str
    path: str
    size_mb: float
    modified: float
    category: str
    tags: List[str]
    base_model: str
    noise_level: str
    format: str
    rank: str


class LoRAListResponse(BaseModel):
    items: List[LoRAItem]
    total: int
    categories: List[dict]
    tags: List[dict]


class LoRADetailResponse(LoRAItem):
    full_path: str
    size_bytes: int


# =============================================================================
# Endpoints
# =============================================================================


@router.get("", response_model=LoRAListResponse)
async def list_loras(
    q: Optional[str] = Query(None, description="Search query"),
    category: Optional[str] = Query(None, description="Filter by category"),
    tag: Optional[str] = Query(None, description="Filter by tag"),
    base_model: Optional[str] = Query(None, description="Filter by base model"),
    noise: Optional[str] = Query(None, description="Filter by noise level (high/low)"),
    sort: str = Query("name", description="Sort by: name, size, modified"),
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
):
    """List all available LoRA models with optional search and filters."""
    debug_log(f"List LoRAs: q={q}, category={category}, tag={tag}, sort={sort}")

    # Get LoRAs (with optional search)
    loras = lora_cache.search(q) if q else lora_cache.get_all()

    # Apply filters
    if category:
        loras = [l for l in loras if l.category == category]
    if tag:
        loras = [l for l in loras if tag in l.tags]
    if base_model:
        loras = [l for l in loras if l.base_model == base_model]
    if noise:
        loras = [l for l in loras if l.noise_level == noise]

    # Sort
    if sort == "size":
        loras.sort(key=lambda x: x.size_bytes, reverse=True)
    elif sort == "modified":
        loras.sort(key=lambda x: x.modified, reverse=True)
    else:
        loras.sort(key=lambda x: x.name.lower())

    total = len(loras)

    # Paginate
    start = (page - 1) * per_page
    end = start + per_page
    page_items = loras[start:end]

    return LoRAListResponse(
        items=[
            LoRAItem(
                id=l.id,
                filename=l.filename,
                name=l.name,
                path=l.path,
                size_mb=l.size_mb,
                modified=l.modified,
                category=l.category,
                tags=l.tags,
                base_model=l.base_model,
                noise_level=l.noise_level,
                format=l.format,
                rank=l.rank,
            )
            for l in page_items
        ],
        total=total,
        categories=lora_cache.get_categories(),
        tags=lora_cache.get_tags(),
    )


@router.get("/categories")
async def get_categories():
    """Get available LoRA categories with counts."""
    return lora_cache.get_categories()


@router.get("/tags")
async def get_tags():
    """Get available LoRA tags with counts."""
    return lora_cache.get_tags()


@router.post("/refresh")
async def refresh_loras():
    """Force re-scan of LoRA directories."""
    debug_log("Force refreshing LoRA cache")
    loras = lora_cache.get_all(force_refresh=True)
    return {
        "success": True,
        "total": len(loras),
        "categories": lora_cache.get_categories(),
    }


@router.get("/{lora_id}", response_model=LoRADetailResponse)
async def get_lora_detail(lora_id: str):
    """Get detailed information about a specific LoRA."""
    lora = lora_cache.get_by_id(lora_id)
    if not lora:
        raise HTTPException(status_code=404, detail="LoRA not found")

    return LoRADetailResponse(
        id=lora.id,
        filename=lora.filename,
        name=lora.name,
        path=lora.path,
        full_path=lora.full_path,
        size_bytes=lora.size_bytes,
        size_mb=lora.size_mb,
        modified=lora.modified,
        category=lora.category,
        tags=lora.tags,
        base_model=lora.base_model,
        noise_level=lora.noise_level,
        format=lora.format,
        rank=lora.rank,
    )
