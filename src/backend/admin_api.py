"""
Oelala Admin API Routes
FastAPI endpoints for admin user management.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from cachetools import TTLCache
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field, validator
import httpx

from auth import get_current_user, User
from storage_client import get_client as get_storage_client

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/admin", tags=["admin"])

# Debug flag
DEBUG = os.getenv("OELALA_DEBUG", "0") == "1"

# Admin bypass - ONLY enable explicitly for local development
# Default is OFF (secure) - set OELALA_ADMIN_BYPASS=1 to enable
ADMIN_BYPASS = os.getenv("OELALA_ADMIN_BYPASS", "0") == "1"

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://nsbjwhxdkxnyggtuxjjp.supabase.co")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")


def debug_log(msg: str):
    if DEBUG:
        logger.info(f"👑 ADMIN-API: {msg}")


# =============================================================================
# Pydantic Models
# =============================================================================


class UserInfo(BaseModel):
    """User information for admin panel."""

    user_id: str
    email: Optional[str]
    created_at: datetime
    balance: int
    tier: str
    is_vip: bool
    is_admin: bool
    is_suspended: bool = False
    suspended_at: Optional[datetime] = None
    suspension_reason: Optional[str] = None
    lifetime_purchased: int
    lifetime_used: int


class UserListResponse(BaseModel):
    """Response for user list with pagination."""

    users: List[UserInfo]
    total: int
    page: int
    per_page: int


class CreditAdjustment(BaseModel):
    """Request to adjust user credits."""

    user_id: str
    amount: int  # Positive to add, negative to subtract
    reason: str = Field(
        min_length=3, description="Reason for credit adjustment (min 3 characters)"
    )

    @validator("amount")
    def validate_amount(cls, value):
        if abs(value) > 100000:
            raise ValueError("Amount must be between -100,000 and 100,000")
        if value == 0:
            raise ValueError("Amount cannot be zero")
        return value


class TierUpdate(BaseModel):
    """Request to update user tier."""

    user_id: str
    tier: str  # 'free', 'pro', 'vip'


class StatusToggle(BaseModel):
    """Request to toggle admin/VIP status."""

    user_id: str
    is_admin: Optional[bool] = None
    is_vip: Optional[bool] = None


class SuspensionToggle(BaseModel):
    """Request to suspend/unsuspend a user."""

    user_id: str
    is_suspended: bool
    reason: Optional[str] = Field(
        None,
        max_length=500,
        description="Reason for suspension (optional for unsuspend)",
    )


class TransactionInfo(BaseModel):
    """Transaction information."""

    id: str
    user_id: str
    amount: int
    type: str
    description: Optional[str]
    reference_id: Optional[str]
    created_at: datetime


class AdminStats(BaseModel):
    """System-wide admin statistics."""

    total_users: int
    total_credits_issued: int
    total_credits_used: int
    total_admins: int
    total_vips: int
    tier_counts: dict


# =============================================================================
# Helper Functions
# =============================================================================

# TTL cache for admin status (60 seconds, max 128 users)
_admin_cache: TTLCache = TTLCache(maxsize=128, ttl=60)

# Shared httpx client for Supabase requests (connection pooling)
_admin_http_client: Optional[httpx.AsyncClient] = None


def _get_admin_client() -> httpx.AsyncClient:
    """Get or create shared httpx client for admin API."""
    global _admin_http_client
    if _admin_http_client is None or _admin_http_client.is_closed:
        _admin_http_client = httpx.AsyncClient(
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
    return _admin_http_client


async def check_admin(user: User) -> bool:
    """Check if user is an admin. Results cached for 60s."""
    # TEMPORARY BYPASS: All authenticated users are admin until DB is set up
    if ADMIN_BYPASS:
        debug_log(f"ADMIN_BYPASS enabled - user {user.id} granted admin access")
        return True

    if not SUPABASE_SERVICE_KEY:
        debug_log("SUPABASE_SERVICE_KEY not configured")
        return False

    # Check cache first
    cached = _admin_cache.get(user.id)
    if cached is not None:
        return cached

    client = _get_admin_client()
    response = await client.get(
        f"{SUPABASE_URL}/rest/v1/user_credits",
        params={"user_id": f"eq.{user.id}", "select": "is_admin"},
    )

    is_admin = False
    if response.status_code == 200 and response.json():
        data = response.json()[0]
        is_admin = data.get("is_admin", False)

    _admin_cache[user.id] = is_admin
    return is_admin


async def get_admin_user(user: User = Depends(get_current_user)) -> User:
    """
    Dependency to ensure user is an admin.
    Raises HTTPException 403 if not admin.
    """
    is_admin = await check_admin(user)
    if not is_admin:
        debug_log(f"User {user.id} is not an admin")
        raise HTTPException(status_code=403, detail="Admin access required")

    debug_log(f"Admin user {user.id} authenticated")
    return user


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/check", response_model=dict)
async def check_admin_status(user: User = Depends(get_current_user)):
    """
    Check if current user is an admin.
    Returns {"is_admin": true/false}
    """
    is_admin = await check_admin(user)
    return {"is_admin": is_admin}


@router.get("/users", response_model=UserListResponse)
async def list_users(
    page: int = Query(1, ge=1),
    per_page: int = Query(30, ge=1, le=100),
    search: Optional[str] = None,
    tier: Optional[str] = None,
    admin: User = Depends(get_admin_user),
):
    """
    List all users with pagination and filtering.
    Admin only.
    """
    debug_log(f"Listing users: page={page}, per_page={per_page}, search={search}")

    client = _get_admin_client()

    # Build query params — include count in the same request
    params = {
        "select": "user_id,balance,tier,is_vip,is_admin,lifetime_purchased,lifetime_used,created_at,is_suspended,suspended_at,suspension_reason",
        "order": "created_at.desc",
        "limit": per_page,
        "offset": (page - 1) * per_page,
    }

    if tier:
        params["tier"] = f"eq.{tier}"

    # Single request with count=exact header (eliminates separate count query)
    response = await client.get(
        f"{SUPABASE_URL}/rest/v1/user_credits",
        headers={
            **client.headers,
            "Prefer": "count=exact",
        },
        params=params,
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch users")

    credits_data = response.json()

    # Parse total from Content-Range header
    total = 0
    content_range = response.headers.get("Content-Range", "")
    if "/" in content_range:
        try:
            total = int(content_range.split("/")[1])
        except (ValueError, IndexError):
            total = len(credits_data)

    # Get auth.users data for emails — only for users on this page
    user_ids = [u["user_id"] for u in credits_data]
    email_map = {}
    if user_ids:
        # Fetch individual users in parallel (instead of fetching ALL users)
        async def fetch_user_email(uid: str):
            try:
                resp = await client.get(
                    f"{SUPABASE_URL}/auth/v1/admin/users/{uid}",
                )
                if resp.status_code == 200:
                    return uid, resp.json().get("email")
            except Exception:
                pass
            return uid, None

        results = await asyncio.gather(
            *[fetch_user_email(uid) for uid in user_ids],
            return_exceptions=True,
        )
        for result in results:
            if not isinstance(result, Exception) and result:
                email_map[result[0]] = result[1]

    # Combine data
    users = [
        UserInfo(
            user_id=u["user_id"],
            email=email_map.get(u["user_id"]),
            created_at=u["created_at"],
            balance=u["balance"],
            tier=u["tier"],
            is_vip=u["is_vip"],
            is_admin=u["is_admin"],
            is_suspended=u.get("is_suspended", False),
            suspended_at=u.get("suspended_at"),
            suspension_reason=u.get("suspension_reason"),
            lifetime_purchased=u["lifetime_purchased"],
            lifetime_used=u["lifetime_used"],
        )
        for u in credits_data
    ]

    return UserListResponse(
        users=users,
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/users/{user_id}", response_model=UserInfo)
async def get_user(user_id: str, admin: User = Depends(get_admin_user)):
    """
    Get detailed information about a specific user.
    Admin only.
    """
    debug_log(f"Fetching user {user_id}")

    client = _get_admin_client()

    # Parallel fetch: user_credits + auth email
    credits_task = client.get(
        f"{SUPABASE_URL}/rest/v1/user_credits",
        params={"user_id": f"eq.{user_id}", "select": "*"},
    )
    auth_task = client.get(
        f"{SUPABASE_URL}/auth/v1/admin/users/{user_id}",
    )
    response, auth_response = await asyncio.gather(credits_task, auth_task)

    if response.status_code != 200 or not response.json():
        raise HTTPException(status_code=404, detail="User not found")

    data = response.json()[0]
    email = (
        auth_response.json().get("email") if auth_response.status_code == 200 else None
    )

    return UserInfo(
        user_id=data["user_id"],
        email=email,
        created_at=data["created_at"],
        balance=data["balance"],
        tier=data["tier"],
        is_vip=data["is_vip"],
        is_admin=data["is_admin"],
        is_suspended=data.get("is_suspended", False),
        suspended_at=data.get("suspended_at"),
        suspension_reason=data.get("suspension_reason"),
        lifetime_purchased=data["lifetime_purchased"],
        lifetime_used=data["lifetime_used"],
    )


@router.post("/credits/adjust")
async def adjust_credits(
    adjustment: CreditAdjustment, admin: User = Depends(get_admin_user)
):
    """
    Adjust user credits (add or subtract).
    Admin only.
    """
    debug_log(
        f"Adjusting credits for {adjustment.user_id}: {adjustment.amount} ({adjustment.reason})"
    )

    client = _get_admin_client()

    # Call admin_grant_credits function
    response = await client.post(
        f"{SUPABASE_URL}/rest/v1/rpc/admin_grant_credits",
        json={
            "p_user_id": adjustment.user_id,
            "p_amount": adjustment.amount,
            "p_description": adjustment.reason,
            "p_admin_id": admin.id,
        },
    )

    if response.status_code != 200:
        logger.error(f"Failed to adjust credits: {response.text}")
        raise HTTPException(status_code=500, detail="Failed to adjust credits")

    result = response.json()
    if isinstance(result, list) and result:
        result = result[0]

    if not result.get("success"):
        raise HTTPException(
            status_code=400, detail=result.get("error", "Failed to adjust credits")
        )

    return {
        "success": True,
        "new_balance": result.get("new_balance"),
        "message": f"Credits adjusted by {adjustment.amount}",
    }


@router.post("/tier/update")
async def update_tier(tier_update: TierUpdate, admin: User = Depends(get_admin_user)):
    """
    Update user tier (free/pro/vip).
    Admin only.
    """
    debug_log(f"Updating tier for {tier_update.user_id} to {tier_update.tier}")

    if tier_update.tier not in ["free", "pro", "vip"]:
        raise HTTPException(status_code=400, detail="Invalid tier value")

    client = _get_admin_client()

    response = await client.post(
        f"{SUPABASE_URL}/rest/v1/rpc/admin_update_tier",
        json={
            "p_user_id": tier_update.user_id,
            "p_tier": tier_update.tier,
            "p_admin_id": admin.id,
        },
    )

    if response.status_code != 200:
        logger.error(f"Failed to update tier: {response.text}")
        raise HTTPException(status_code=500, detail="Failed to update tier")

    result = response.json()
    if isinstance(result, list) and result:
        result = result[0]

    if not result.get("success"):
        raise HTTPException(
            status_code=400, detail=result.get("error", "Failed to update tier")
        )

    return {"success": True, "message": f"Tier updated to {tier_update.tier}"}


@router.post("/status/toggle")
async def toggle_status(status: StatusToggle, admin: User = Depends(get_admin_user)):
    """
    Toggle admin or VIP status for a user.
    Admin only (service role for admin status).
    """
    debug_log(
        f"Toggling status for {status.user_id}: admin={status.is_admin}, vip={status.is_vip}"
    )

    client = _get_admin_client()

    response = await client.post(
        f"{SUPABASE_URL}/rest/v1/rpc/admin_toggle_status",
        json={
            "p_user_id": status.user_id,
            "p_is_admin": status.is_admin,
            "p_is_vip": status.is_vip,
        },
    )

    if response.status_code != 200:
        logger.error(f"Failed to toggle status: {response.text}")
        raise HTTPException(status_code=500, detail="Failed to toggle status")

    result = response.json()
    if isinstance(result, list) and result:
        result = result[0]

    if not result.get("success"):
        raise HTTPException(
            status_code=400, detail=result.get("error", "Failed to toggle status")
        )

    return {"success": True, "message": "Status updated successfully"}


@router.post("/suspension/toggle")
async def toggle_suspension(
    suspension: SuspensionToggle, admin: User = Depends(get_admin_user)
):
    """
    Suspend or unsuspend a user.
    Suspended users cannot generate content but can still view their existing media.
    Admin only.
    """
    debug_log(
        f"Toggling suspension for {suspension.user_id}: suspended={suspension.is_suspended}, reason={suspension.reason}"
    )

    client = _get_admin_client()

    response = await client.post(
        f"{SUPABASE_URL}/rest/v1/rpc/admin_toggle_suspension",
        json={
            "p_user_id": suspension.user_id,
            "p_is_suspended": suspension.is_suspended,
            "p_reason": suspension.reason,
        },
    )

    if response.status_code != 200:
        logger.error(f"Failed to toggle suspension: {response.text}")
        raise HTTPException(status_code=500, detail="Failed to toggle suspension")

    result = response.json()
    if isinstance(result, list) and result:
        result = result[0]

    if not result.get("success"):
        raise HTTPException(
            status_code=400,
            detail=result.get("error", "Failed to toggle suspension"),
        )

    action = "suspended" if suspension.is_suspended else "unsuspended"
    return {"success": True, "message": f"User {action} successfully"}


@router.get("/transactions/{user_id}", response_model=List[TransactionInfo])
async def get_user_transactions(
    user_id: str,
    limit: int = Query(50, ge=1, le=200),
    admin: User = Depends(get_admin_user),
):
    """
    Get credit transaction history for a user.
    Admin only.
    """
    debug_log(f"Fetching transactions for {user_id}")

    client = _get_admin_client()

    response = await client.get(
        f"{SUPABASE_URL}/rest/v1/credit_transactions",
        params={
            "user_id": f"eq.{user_id}",
            "select": "id,user_id,amount,type,description,reference_id,created_at",
            "order": "created_at.desc",
            "limit": limit,
        },
    )

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch transactions")

    return [
        TransactionInfo(
            id=t["id"],
            user_id=t["user_id"],
            amount=t["amount"],
            type=t["type"],
            description=t.get("description"),
            reference_id=t.get("reference_id"),
            created_at=t["created_at"],
        )
        for t in response.json()
    ]


@router.get("/stats", response_model=AdminStats)
async def get_admin_stats(admin: User = Depends(get_admin_user)):
    """
    Get system-wide statistics using database aggregation.
    Admin only.
    """
    debug_log("Fetching admin stats")

    client = _get_admin_client()

    # Use PostgreSQL aggregation via Supabase RPC for efficiency
    response = await client.post(
        f"{SUPABASE_URL}/rest/v1/rpc/get_admin_stats",
        json={},
    )

    # Fallback to simple query if RPC doesn't exist
    if response.status_code != 200:
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/user_credits",
            params={
                "select": "tier,is_admin,is_vip,lifetime_purchased,lifetime_used",
            },
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch stats")

        users = response.json()

        tier_counts = {"free": 0, "pro": 0, "vip": 0}
        total_purchased = 0
        total_used = 0
        admin_count = 0
        vip_count = 0

        for user in users:
            tier = user.get("tier", "free")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            total_purchased += user.get("lifetime_purchased", 0)
            total_used += user.get("lifetime_used", 0)
            if user.get("is_admin"):
                admin_count += 1
            if user.get("is_vip"):
                vip_count += 1

        return AdminStats(
            total_users=len(users),
            total_credits_issued=total_purchased,
            total_credits_used=total_used,
            total_admins=admin_count,
            total_vips=vip_count,
            tier_counts=tier_counts,
        )

    # Use RPC result if available
    result = response.json()
    if isinstance(result, list) and result:
        result = result[0]

    return AdminStats(
        total_users=result.get("total_users", 0),
        total_credits_issued=result.get("total_purchased", 0),
        total_credits_used=result.get("total_used", 0),
        total_admins=result.get("admin_count", 0),
        total_vips=result.get("vip_count", 0),
        tier_counts={
            "free": result.get("free_count", 0),
            "pro": result.get("pro_count", 0),
            "vip": result.get("vip_tier_count", 0),
        },
    )


# =============================================================================
# Admin Generated Media Access (Transition Phase)
# =============================================================================

# Fallback directories (used when storage service is unavailable)
MEDIA_GENERATED_DIR = Path("/home/flip/oelala/media/generated")
COMFYUI_OUTPUT_DIR = Path("/home/flip/oelala/ComfyUI/output")


def check_file_has_metadata(file_path: Path) -> bool:
    """
    Quick check if a media file has embedded ComfyUI workflow metadata.
    Returns True if metadata exists, False otherwise.
    """
    import subprocess
    import json

    ext = file_path.suffix.lower()

    try:
        if ext in [".mp4", ".webm", ".mov"]:
            # Check video metadata using ffprobe
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    str(file_path),
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                probe_data = json.loads(result.stdout)
                comment = (
                    probe_data.get("format", {}).get("tags", {}).get("comment", "")
                )
                return bool(comment and comment.startswith("{"))

        elif ext == ".png":
            # Check PNG metadata
            from PIL import Image

            img = Image.open(str(file_path))
            if hasattr(img, "text"):
                return "prompt" in img.text or "workflow" in img.text
            img.close()
    except Exception:
        pass

    return False


@router.get("/generated-media")
async def list_generated_media(
    admin: User = Depends(get_admin_user),
    type: str = Query("all", description="Filter: 'all', 'video', 'image'"),
    limit: int = Query(100, ge=1, le=500),
):
    """
    List all media files from MinIO storage buckets (admin only).
    Reads from 'generated' and 'comfyui-local' buckets.
    """
    media = []
    video_exts = {".mp4", ".webm"}
    image_exts = {".png", ".jpg", ".jpeg", ".webp"}
    all_exts = video_exts | image_exts

    try:
        storage = get_storage_client()

        # List generated bucket
        for obj in storage.list("generated"):
            key = obj.get("key", "")
            ext = Path(key).suffix.lower()
            if ext not in all_exts:
                continue
            item_type = "video" if ext in video_exts else "image"
            if type != "all" and item_type != type:
                continue
            media.append(
                {
                    "name": Path(key).name,
                    "type": item_type,
                    "url": f"/media/generated/{key}",
                    "source": "generated",
                    "size": obj.get("size", 0),
                    "modified": obj.get("modified_at", ""),
                    "mtime": 0,
                    "has_metadata": False,
                }
            )

        # List comfyui-local bucket
        for obj in storage.list("comfyui-local"):
            key = obj.get("key", "")
            ext = Path(key).suffix.lower()
            if ext not in all_exts:
                continue
            item_type = "video" if ext in video_exts else "image"
            if type != "all" and item_type != type:
                continue
            media.append(
                {
                    "name": Path(key).name,
                    "type": item_type,
                    "url": f"/comfyui/output/{Path(key).name}",
                    "source": "comfyui-local",
                    "size": obj.get("size", 0),
                    "modified": obj.get("modified_at", ""),
                    "mtime": 0,
                    "has_metadata": False,
                }
            )
    except Exception as e:
        logger.warning(f"⚠️ Storage list failed, falling back to local scan: {e}")
        # Fallback to local scan
        for src_dir, url_prefix, source in [
            (MEDIA_GENERATED_DIR, "/media/generated", "media/generated"),
            (COMFYUI_OUTPUT_DIR, "/comfyui/output", "ComfyUI/output"),
        ]:
            if src_dir.exists():
                for ext_pat in [
                    "*.mp4",
                    "*.webm",
                    "*.png",
                    "*.jpg",
                    "*.jpeg",
                    "*.webp",
                ]:
                    for fp in src_dir.glob(ext_pat):
                        is_video = fp.suffix.lower() in video_exts
                        item_type = "video" if is_video else "image"
                        if type != "all" and item_type != type:
                            continue
                        stat = fp.stat()
                        media.append(
                            {
                                "name": fp.name,
                                "type": item_type,
                                "url": f"{url_prefix}/{fp.name}",
                                "source": source,
                                "size": stat.st_size,
                                "modified": datetime.fromtimestamp(
                                    stat.st_mtime
                                ).isoformat(),
                                "mtime": stat.st_mtime,
                                "has_metadata": False,
                            }
                        )

    # Sort by modified (newest first)
    media.sort(key=lambda m: m.get("modified", ""), reverse=True)

    return {
        "media": media[:limit],
        "total": len(media),
        "sources": ["generated", "comfyui-local"],
    }


@router.get("/generated-media/file/{filename}")
async def get_generated_file(
    filename: str,
    admin: User = Depends(get_admin_user),
):
    """Serve a file from generated storage bucket (admin only)."""
    try:
        storage = get_storage_client()
        data, content_type, _ = storage.get_with_metadata("generated", filename)
        return Response(
            content=data,
            media_type=content_type,
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "Cache-Control": "public, max-age=3600",
            },
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="File not found")
        raise HTTPException(status_code=502, detail="Storage error")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Storage unavailable")


@router.get("/generated-media/comfyui/{filename}")
async def get_comfyui_file(
    filename: str,
    admin: User = Depends(get_admin_user),
):
    """Serve a file from comfyui-local storage bucket (admin only)."""
    try:
        storage = get_storage_client()
        data, content_type, _ = storage.get_with_metadata("comfyui-local", filename)
        return Response(
            content=data,
            media_type=content_type,
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "Cache-Control": "public, max-age=3600",
            },
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="File not found")
        raise HTTPException(status_code=502, detail="Storage error")
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail="Storage unavailable")


# =============================================================================
# System Monitoring Endpoints (Issue #58)
# =============================================================================


@router.get("/system/gpu")
async def get_gpu_status(admin: User = Depends(get_admin_user)):
    """
    Get GPU utilization via nvidia-smi.
    Returns VRAM usage, GPU utilization, temperature for each GPU.
    """
    import subprocess

    try:
        # Run nvidia-smi with CSV output for easy parsing
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode != 0:
            raise HTTPException(status_code=500, detail="nvidia-smi failed")

        gpus = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 7:
                # Parse values
                idx = int(parts[0])
                name = parts[1]
                mem_total = int(parts[2])
                mem_used = int(parts[3])
                mem_free = int(parts[4])
                util = int(parts[5]) if parts[5] != "[N/A]" else 0
                temp = int(parts[6]) if parts[6] != "[N/A]" else 0

                gpus.append(
                    {
                        "index": idx,
                        "name": name,
                        "memory_total_mb": mem_total,
                        "memory_used_mb": mem_used,
                        "memory_free_mb": mem_free,
                        "memory_percent": round(mem_used / mem_total * 100, 1)
                        if mem_total > 0
                        else 0,
                        "utilization_percent": util,
                        "temperature_c": temp,
                    }
                )

        return {
            "gpus": gpus,
            "total_gpus": len(gpus),
            "timestamp": datetime.now().isoformat(),
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="nvidia-smi timed out")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="nvidia-smi not found")
    except Exception as e:
        logger.error(f"GPU status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/queue")
async def get_queue_status(admin: User = Depends(get_admin_user)):
    """
    Get ComfyUI queue status - running and pending jobs.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:8188/queue", timeout=5.0)

            if response.status_code != 200:
                return {
                    "status": "error",
                    "message": "ComfyUI not responding",
                    "running": [],
                    "pending": [],
                }

            data = response.json()

            # Parse running jobs
            running = []
            for item in data.get("queue_running", []):
                if len(item) >= 2:
                    running.append(
                        {
                            "prompt_id": item[1],
                            "status": "running",
                        }
                    )

            # Parse pending jobs
            pending = []
            for idx, item in enumerate(data.get("queue_pending", [])):
                if len(item) >= 2:
                    pending.append(
                        {
                            "prompt_id": item[1],
                            "status": "pending",
                            "position": idx + 1,
                        }
                    )

            return {
                "status": "ok",
                "running": running,
                "pending": pending,
                "running_count": len(running),
                "pending_count": len(pending),
                "timestamp": datetime.now().isoformat(),
            }

    except httpx.TimeoutException:
        return {
            "status": "timeout",
            "message": "ComfyUI request timed out",
            "running": [],
            "pending": [],
        }
    except httpx.ConnectError:
        return {
            "status": "offline",
            "message": "ComfyUI is offline",
            "running": [],
            "pending": [],
        }
    except Exception as e:
        logger.error(f"Queue status error: {e}")
        return {
            "status": "error",
            "message": str(e),
            "running": [],
            "pending": [],
        }


@router.get("/system/health")
async def get_system_health(admin: User = Depends(get_admin_user)):
    """
    Comprehensive system health check for admin dashboard.
    """
    import shutil

    health = {
        "timestamp": datetime.now().isoformat(),
        "services": {},
        "disk": {},
    }

    # Check ComfyUI
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://localhost:8188/system_stats", timeout=3.0
            )
            health["services"]["comfyui"] = {
                "status": "online" if response.status_code == 200 else "error",
                "port": 8188,
            }
            if response.status_code == 200:
                stats = response.json()
                if "system" in stats:
                    health["services"]["comfyui"]["system"] = stats["system"]
    except Exception:
        health["services"]["comfyui"] = {"status": "offline", "port": 8188}

    # Check MinIO storage
    try:
        storage = get_storage_client()
        storage_health = storage.health()
        health["services"]["storage"] = {
            "status": "online"
            if storage_health.get("status") == "healthy"
            else "error",
            "port": 9000,
            "backend": "minio",
        }
    except Exception:
        health["services"]["storage"] = {
            "status": "offline",
            "port": 9000,
            "backend": "minio",
        }

    # Disk usage
    for name, path in [
        ("root", "/"),
        ("home", "/home/flip"),
        ("ssd", "/mnt/ssd"),
    ]:
        try:
            usage = shutil.disk_usage(path)
            health["disk"][name] = {
                "total_gb": round(usage.total / (1024**3), 1),
                "used_gb": round(usage.used / (1024**3), 1),
                "free_gb": round(usage.free / (1024**3), 1),
                "percent": round(usage.used / usage.total * 100, 1),
            }
        except Exception:
            pass

    return health


@router.get("/system/logs")
async def get_recent_logs(
    service: str = "oelala-backend",
    lines: int = Query(default=50, le=200),
    admin: User = Depends(get_admin_user),
):
    """
    Get recent logs from systemd services.
    Supported services: oelala-backend, comfyui, minio
    """
    import subprocess

    allowed_services = [
        "oelala-backend",
        "comfyui",
        "minio",
        "oelala-frontend",
    ]

    if service not in allowed_services:
        raise HTTPException(
            status_code=400,
            detail=f"Service must be one of: {', '.join(allowed_services)}",
        )

    try:
        result = subprocess.run(
            [
                "journalctl",
                "-u",
                service,
                "-n",
                str(lines),
                "--no-pager",
                "-o",
                "short-iso",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        log_lines = []
        for line in result.stdout.strip().split("\n"):
            if line.strip():
                log_lines.append(line)

        return {
            "service": service,
            "lines": log_lines,
            "count": len(log_lines),
            "timestamp": datetime.now().isoformat(),
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=500, detail="Log fetch timed out")
    except Exception as e:
        logger.error(f"Log fetch error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# AI Settings Management
# =============================================================================

AI_SETTINGS_FILE = Path("/home/flip/oelala/data/ai_settings.json")

DEFAULT_PROMPT_SYSTEM = """You are an expert AI image/video generation prompt engineer.
Your task is to enhance simple descriptions into detailed, high-quality prompts.

Rules:
1. Keep the core subject/concept intact
2. Add visual details: lighting, composition, atmosphere, colors
3. Add quality boosters at the end: masterpiece, best quality, highly detailed
4. For negative prompts: focus on common defects to avoid
5. Be concise but descriptive (max 100 words per prompt)
6. Output ONLY valid JSON, no markdown, no explanation

Output format (strict JSON):
{"prompt": "enhanced prompt here", "negative_prompt": "negative prompt here", "motion_prompt": "motion description if requested"}"""


class AISettingsUpdate(BaseModel):
    """AI settings update request"""

    prompt_system: Optional[str] = None
    llm_model: Optional[str] = None
    ollama_model: Optional[str] = None  # Deprecated alias, use llm_model


@router.get("/ai-settings")
async def get_ai_settings(user: User = Depends(get_current_user)):
    """Get current AI settings (admin only)"""
    if not ADMIN_BYPASS and not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    import json

    _guardian_base = os.getenv(
        "GUARDIAN_BASE_URL",
        os.getenv("GUARDIAN_BASE", os.getenv("OLLAMA_BASE", "http://localhost:11434")),
    ).rstrip("/")
    _default_model = os.getenv("GUARDIAN_MODEL", os.getenv("OLLAMA_MODEL", ""))
    settings = {
        "prompt_system": DEFAULT_PROMPT_SYSTEM,
        "llm_model": _default_model,
    }

    if AI_SETTINGS_FILE.exists():
        try:
            with open(AI_SETTINGS_FILE, "r") as f:
                saved = json.load(f)
                # Migrate legacy key
                if "ollama_model" in saved and "llm_model" not in saved:
                    saved["llm_model"] = saved.pop("ollama_model")
                settings.update(saved)
        except Exception as e:
            logger.warning(f"Failed to load AI settings: {e}")

    # Fetch available Guardian models via OpenAI /v1/models (Bearer token auth)
    _guardian_api_key = os.getenv("GUARDIAN_API_KEY", "")
    _auth_headers = (
        {"Authorization": f"Bearer {_guardian_api_key}"} if _guardian_api_key else {}
    )
    available_models = []
    try:
        async with httpx.AsyncClient(timeout=5.0, headers=_auth_headers) as client:
            res = await client.get(f"{_guardian_base}/v1/models")
            if res.status_code == 200:
                models = res.json().get("data", [])
                available_models = [m.get("id", "") for m in models]
    except Exception:
        pass

    return {
        **settings,
        "available_models": available_models,
        "default_prompt_system": DEFAULT_PROMPT_SYSTEM,
    }


@router.post("/ai-settings")
async def update_ai_settings(
    update: AISettingsUpdate, user: User = Depends(get_current_user)
):
    """Update AI settings (admin only)"""
    if not ADMIN_BYPASS and not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    import json

    # Load existing settings
    _default_model = os.getenv("GUARDIAN_MODEL", os.getenv("OLLAMA_MODEL", ""))
    settings = {
        "prompt_system": DEFAULT_PROMPT_SYSTEM,
        "llm_model": _default_model,
    }

    if AI_SETTINGS_FILE.exists():
        try:
            with open(AI_SETTINGS_FILE, "r") as f:
                saved = json.load(f)
                # Migrate legacy key
                if "ollama_model" in saved and "llm_model" not in saved:
                    saved["llm_model"] = saved.pop("ollama_model")
                settings.update(saved)
        except Exception:
            pass

    # Update with new values
    if update.prompt_system is not None:
        settings["prompt_system"] = update.prompt_system
    # Support both new llm_model and deprecated ollama_model
    new_model = update.llm_model or update.ollama_model
    if new_model is not None:
        settings["llm_model"] = new_model

    # Save
    try:
        AI_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(AI_SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)

        logger.info(f"AI settings updated by admin {user.id}")
        return {"success": True, "settings": settings}
    except Exception as e:
        logger.error(f"Failed to save AI settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save settings: {e}")


@router.post("/ai-settings/reset")
async def reset_ai_settings(user: User = Depends(get_current_user)):
    """Reset AI settings to defaults (admin only)"""
    if not ADMIN_BYPASS and not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    try:
        if AI_SETTINGS_FILE.exists():
            AI_SETTINGS_FILE.unlink()

        logger.info(f"AI settings reset to defaults by admin {user.id}")
        return {
            "success": True,
            "settings": {
                "prompt_system": DEFAULT_PROMPT_SYSTEM,
                "llm_model": os.getenv("GUARDIAN_MODEL", os.getenv("OLLAMA_MODEL", "")),
            },
        }
    except Exception as e:
        logger.error(f"Failed to reset AI settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset settings: {e}")
