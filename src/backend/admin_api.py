"""
Oelala Admin API Routes
FastAPI endpoints for admin user management.
"""

import os
import logging
from pathlib import Path
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator
import httpx

from auth import get_current_user, User

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


async def check_admin(user: User) -> bool:
    """Check if user is an admin by querying user_credits table."""
    # TEMPORARY BYPASS: All authenticated users are admin until DB is set up
    if ADMIN_BYPASS:
        debug_log(f"ADMIN_BYPASS enabled - user {user.id} granted admin access")
        return True

    if not SUPABASE_SERVICE_KEY:
        debug_log("SUPABASE_SERVICE_KEY not configured")
        return False

    async with httpx.AsyncClient() as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        }

        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/user_credits",
            headers=headers,
            params={"user_id": f"eq.{user.id}", "select": "is_admin"},
        )

        if response.status_code == 200 and response.json():
            data = response.json()[0]
            return data.get("is_admin", False)

    return False


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

    async with httpx.AsyncClient() as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        }

        # Build query params
        params = {
            "select": "user_id,balance,tier,is_vip,is_admin,lifetime_purchased,lifetime_used,created_at",
            "order": "created_at.desc",
            "limit": per_page,
            "offset": (page - 1) * per_page,
        }

        if tier:
            params["tier"] = f"eq.{tier}"

        # Get user_credits data
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/user_credits",
            headers=headers,
            params=params,
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch users")

        credits_data = response.json()

        # Get auth.users data for emails
        user_ids = [u["user_id"] for u in credits_data]

        # Fetch emails from auth.users
        email_map = {}
        if user_ids:
            auth_response = await client.get(
                f"{SUPABASE_URL}/auth/v1/admin/users",
                headers={
                    "apikey": SUPABASE_SERVICE_KEY,
                    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                },
            )

            if auth_response.status_code == 200:
                auth_users = auth_response.json().get("users", [])
                email_map = {u["id"]: u.get("email") for u in auth_users}

        # Get total count
        count_response = await client.get(
            f"{SUPABASE_URL}/rest/v1/user_credits",
            headers={**headers, "Prefer": "count=exact"},
            params={"select": "user_id"},
        )

        total = 0
        if count_response.status_code == 200:
            content_range = count_response.headers.get("Content-Range", "")
            if "/" in content_range:
                total = int(content_range.split("/")[1])

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

    async with httpx.AsyncClient() as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        }

        # Get user_credits
        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/user_credits",
            headers=headers,
            params={"user_id": f"eq.{user_id}", "select": "*"},
        )

        if response.status_code != 200 or not response.json():
            raise HTTPException(status_code=404, detail="User not found")

        data = response.json()[0]

        # Get email from auth.users
        auth_response = await client.get(
            f"{SUPABASE_URL}/auth/v1/admin/users/{user_id}",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            },
        )

        email = None
        if auth_response.status_code == 200:
            email = auth_response.json().get("email")

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

    async with httpx.AsyncClient() as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
        }

        # Call admin_grant_credits function
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/admin_grant_credits",
            headers=headers,
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

    async with httpx.AsyncClient() as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
        }

        # Call admin_update_tier function
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/admin_update_tier",
            headers=headers,
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

    async with httpx.AsyncClient() as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
        }

        # Call admin_toggle_status function
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/admin_toggle_status",
            headers=headers,
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

    async with httpx.AsyncClient() as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
            "Content-Type": "application/json",
        }

        # Call admin_toggle_suspension function
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/admin_toggle_suspension",
            headers=headers,
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

    async with httpx.AsyncClient() as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        }

        response = await client.get(
            f"{SUPABASE_URL}/rest/v1/credit_transactions",
            headers=headers,
            params={
                "user_id": f"eq.{user_id}",
                "select": "*",
                "order": "created_at.desc",
                "limit": limit,
            },
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to fetch transactions")

        transactions = [
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

        return transactions


@router.get("/stats", response_model=AdminStats)
async def get_admin_stats(admin: User = Depends(get_admin_user)):
    """
    Get system-wide statistics using database aggregation.
    Admin only.
    """
    debug_log("Fetching admin stats")

    async with httpx.AsyncClient() as client:
        headers = {
            "apikey": SUPABASE_SERVICE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        }

        # Use PostgreSQL aggregation via Supabase RPC for efficiency
        response = await client.post(
            f"{SUPABASE_URL}/rest/v1/rpc/get_admin_stats",
            headers=headers,
            json={},
        )

        # Fallback to simple query if RPC doesn't exist
        if response.status_code != 200:
            # Get aggregated data directly
            response = await client.get(
                f"{SUPABASE_URL}/rest/v1/user_credits",
                headers=headers,
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

# Media directories for admin access
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
    List all media files from generated directories (admin only).

    This is a transition endpoint while migrating to user-scoped storage.
    Returns videos/images from media/generated/ and ComfyUI/output/.
    """
    media = []

    # Scan media/generated
    if MEDIA_GENERATED_DIR.exists():
        for ext in ["*.mp4", "*.webm", "*.png", "*.jpg", "*.jpeg", "*.webp"]:
            for file_path in MEDIA_GENERATED_DIR.glob(ext):
                stat = file_path.stat()
                is_video = file_path.suffix.lower() in [".mp4", ".webm"]
                item_type = "video" if is_video else "image"

                if type != "all" and item_type != type:
                    continue

                # Check if file has embedded metadata (workflow)
                has_metadata = check_file_has_metadata(file_path)

                media.append(
                    {
                        "name": file_path.name,
                        "type": item_type,
                        "url": f"/media/generated/{file_path.name}",
                        "source": "media/generated",
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "mtime": stat.st_mtime,
                        "has_metadata": has_metadata,
                    }
                )

    # Scan ComfyUI output
    if COMFYUI_OUTPUT_DIR.exists():
        for ext in ["*.mp4", "*.webm", "*.png", "*.jpg", "*.jpeg", "*.webp"]:
            for file_path in COMFYUI_OUTPUT_DIR.glob(ext):
                stat = file_path.stat()
                is_video = file_path.suffix.lower() in [".mp4", ".webm"]
                item_type = "video" if is_video else "image"

                if type != "all" and item_type != type:
                    continue

                # Check if file has embedded metadata (workflow)
                has_metadata = check_file_has_metadata(file_path)

                media.append(
                    {
                        "name": file_path.name,
                        "type": item_type,
                        "url": f"/comfyui/output/{file_path.name}",
                        "source": "ComfyUI/output",
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "mtime": stat.st_mtime,
                        "has_metadata": has_metadata,
                    }
                )

    # Sort by mtime (newest first)
    media.sort(key=lambda m: m.get("mtime", 0), reverse=True)

    return {
        "media": media[:limit],
        "total": len(media),
        "sources": ["media/generated", "ComfyUI/output"],
    }


@router.get("/generated-media/file/{filename}")
async def get_generated_file(
    filename: str,
    admin: User = Depends(get_admin_user),
):
    """Serve a file from media/generated/ (admin only)."""
    file_path = MEDIA_GENERATED_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine content type
    suffix = file_path.suffix.lower()
    content_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    content_type = content_types.get(suffix, "application/octet-stream")

    return FileResponse(path=file_path, media_type=content_type, filename=filename)


@router.get("/generated-media/comfyui/{filename}")
async def get_comfyui_file(
    filename: str,
    admin: User = Depends(get_admin_user),
):
    """Serve a file from ComfyUI/output/ (admin only)."""
    file_path = COMFYUI_OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    # Determine content type
    suffix = file_path.suffix.lower()
    content_types = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }
    content_type = content_types.get(suffix, "application/octet-stream")

    return FileResponse(path=file_path, media_type=content_type, filename=filename)


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

    # Check oelala-storage
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://localhost:7990/health", timeout=3.0)
            health["services"]["storage"] = {
                "status": "online" if response.status_code == 200 else "error",
                "port": 7990,
            }
    except Exception:
        health["services"]["storage"] = {"status": "offline", "port": 7990}

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
    Supported services: oelala-backend, comfyui, oelala-storage
    """
    import subprocess

    allowed_services = [
        "oelala-backend",
        "comfyui",
        "oelala-storage",
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
    ollama_model: Optional[str] = None


@router.get("/ai-settings")
async def get_ai_settings(user: User = Depends(get_current_user)):
    """Get current AI settings (admin only)"""
    if not ADMIN_BYPASS and not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")

    import json

    settings = {
        "prompt_system": DEFAULT_PROMPT_SYSTEM,
        "ollama_model": os.getenv("OLLAMA_MODEL", "gemma2:9b"),
    }

    if AI_SETTINGS_FILE.exists():
        try:
            with open(AI_SETTINGS_FILE, "r") as f:
                saved = json.load(f)
                settings.update(saved)
        except Exception as e:
            logger.warning(f"Failed to load AI settings: {e}")

    # Also get available Ollama models
    available_models = []
    try:
        async with httpx.AsyncClient(
            timeout=5.0, auth=("oelala-backend", "")
        ) as client:
            res = await client.get("http://localhost:11434/api/tags")
            if res.status_code == 200:
                models = res.json().get("models", [])
                available_models = [m.get("name", "") for m in models]
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
    settings = {
        "prompt_system": DEFAULT_PROMPT_SYSTEM,
        "ollama_model": os.getenv("OLLAMA_MODEL", "gemma2:9b"),
    }

    if AI_SETTINGS_FILE.exists():
        try:
            with open(AI_SETTINGS_FILE, "r") as f:
                settings.update(json.load(f))
        except Exception:
            pass

    # Update with new values
    if update.prompt_system is not None:
        settings["prompt_system"] = update.prompt_system
    if update.ollama_model is not None:
        settings["ollama_model"] = update.ollama_model

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
                "ollama_model": os.getenv("OLLAMA_MODEL", "gemma2:9b"),
            },
        }
    except Exception as e:
        logger.error(f"Failed to reset AI settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset settings: {e}")
