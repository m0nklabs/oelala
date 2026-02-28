"""
Oelala Content Moderation API
Endpoints for content reporting and admin moderation queue.
"""

import os
import logging
from typing import Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
import httpx

from auth import get_current_user, User

logger = logging.getLogger(__name__)
DEBUG = os.getenv("OELALA_DEBUG", "0") == "1"

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://nsbjwhxdkxnyggtuxjjp.supabase.co")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

# Valid report reasons
REPORT_REASONS = ["inappropriate", "copyright", "spam", "harassment", "underage", "other"]

# Valid moderation actions
MODERATION_ACTIONS = ["approve", "reject", "hide", "unhide", "warn_user", "dismiss_report"]


def debug_log(msg: str):
    if DEBUG:
        logger.info(f"🛡️ MODERATION: {msg}")


# =============================================================================
# Shared HTTP client
# =============================================================================

_mod_http_client: Optional[httpx.AsyncClient] = None


def _get_client() -> httpx.AsyncClient:
    """Get or create shared httpx client for Supabase requests."""
    global _mod_http_client
    if _mod_http_client is None or _mod_http_client.is_closed:
        _mod_http_client = httpx.AsyncClient(
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            },
            timeout=30.0,
        )
    return _mod_http_client


# =============================================================================
# Pydantic Models
# =============================================================================


class ReportRequest(BaseModel):
    """User content report request."""
    media_id: str = Field(..., description="UUID of the published media item")
    reason: str = Field(..., description="Report reason category")
    description: Optional[str] = Field(None, max_length=500, description="Additional details")


class ReportResponse(BaseModel):
    """Response after creating a report."""
    id: str
    media_id: str
    reason: str
    status: str
    created_at: str


class ModerationActionRequest(BaseModel):
    """Admin moderation action request."""
    action: str = Field(..., description="Moderation action to take")
    reason: Optional[str] = Field(None, max_length=500, description="Reason for action")
    report_id: Optional[str] = Field(None, description="Associated report ID to resolve")


class BulkActionRequest(BaseModel):
    """Bulk moderation action."""
    media_ids: List[str] = Field(..., min_length=1, max_length=50)
    action: str = Field(..., description="Action to apply to all items")
    reason: Optional[str] = None


class QueueItem(BaseModel):
    """A moderation queue item with report details."""
    media_id: str
    title: str
    media_type: str
    storage_path: str
    is_nsfw: bool
    moderation_status: str
    creator_id: str
    creator_email: Optional[str] = None
    report_count: int
    reports: list
    created_at: str


class ModerationStats(BaseModel):
    """Moderation statistics."""
    pending_reports: int
    reviewed_today: int
    total_hidden: int
    total_rejected: int
    total_reports: int


# =============================================================================
# Admin dependency (reuse from admin_api)
# =============================================================================

async def _check_admin(user: User) -> bool:
    """Check if user is admin via user_credits table."""
    from admin_api import check_admin
    return await check_admin(user)


async def get_admin_user(user: User = Depends(get_current_user)) -> User:
    """Dependency: require admin access."""
    is_admin = await _check_admin(user)
    if not is_admin:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# =============================================================================
# Routers
# =============================================================================

# Public router — for user reports
public_router = APIRouter(prefix="/api/report", tags=["moderation"])

# Admin router — for moderation actions
admin_router = APIRouter(prefix="/api/admin/moderation", tags=["admin-moderation"])


# =============================================================================
# User Endpoints
# =============================================================================


@public_router.get("/reasons")
async def get_report_reasons():
    """List valid report reasons."""
    return {
        "reasons": [
            {"value": "inappropriate", "label": "Inappropriate content"},
            {"value": "copyright", "label": "Copyright violation"},
            {"value": "spam", "label": "Spam or misleading"},
            {"value": "harassment", "label": "Harassment or bullying"},
            {"value": "underage", "label": "Underage content (urgent)"},
            {"value": "other", "label": "Other"},
        ]
    }


@public_router.post("", response_model=ReportResponse)
async def report_content(
    report: ReportRequest,
    user: User = Depends(get_current_user),
):
    """Report a piece of published content."""
    debug_log(f"User {user.id} reporting media {report.media_id} for: {report.reason}")

    if report.reason not in REPORT_REASONS:
        raise HTTPException(status_code=400, detail=f"Invalid reason. Must be one of: {REPORT_REASONS}")

    client = _get_client()

    # Verify media exists
    media_resp = await client.get(
        f"{SUPABASE_URL}/rest/v1/published_media",
        params={"id": f"eq.{report.media_id}", "select": "id,user_id"},
    )
    if media_resp.status_code != 200 or not media_resp.json():
        raise HTTPException(status_code=404, detail="Media item not found")

    media_data = media_resp.json()[0]

    # Can't report your own content
    if media_data["user_id"] == user.id:
        raise HTTPException(status_code=400, detail="Cannot report your own content")

    # Check for existing pending report from this user
    existing_resp = await client.get(
        f"{SUPABASE_URL}/rest/v1/content_reports",
        params={
            "media_id": f"eq.{report.media_id}",
            "reporter_id": f"eq.{user.id}",
            "status": "eq.pending",
            "select": "id",
        },
    )
    if existing_resp.status_code == 200 and existing_resp.json():
        raise HTTPException(status_code=409, detail="You have already reported this content")

    # Create the report
    report_data = {
        "media_id": report.media_id,
        "reporter_id": user.id,
        "reason": report.reason,
        "description": report.description,
    }

    resp = await client.post(
        f"{SUPABASE_URL}/rest/v1/content_reports",
        json=report_data,
    )

    if resp.status_code not in (200, 201):
        logger.error(f"Failed to create report: {resp.status_code} {resp.text}")
        raise HTTPException(status_code=500, detail="Failed to submit report")

    result = resp.json()
    created = result[0] if isinstance(result, list) else result
    debug_log(f"Report created: {created['id']}")

    # If media gets 3+ pending reports, auto-flag it as pending review
    count_resp = await client.get(
        f"{SUPABASE_URL}/rest/v1/content_reports",
        params={
            "media_id": f"eq.{report.media_id}",
            "status": "eq.pending",
            "select": "id",
        },
        headers={**client.headers, "Prefer": "count=exact"},
    )
    report_count = int(count_resp.headers.get("content-range", "0-0/0").split("/")[-1])

    if report_count >= 3:
        debug_log(f"Media {report.media_id} auto-flagged: {report_count} reports")
        await client.patch(
            f"{SUPABASE_URL}/rest/v1/published_media",
            params={"id": f"eq.{report.media_id}"},
            json={"moderation_status": "pending"},
        )

    # Urgent: underage reports immediately flag content
    if report.reason == "underage":
        debug_log(f"URGENT: Underage report on {report.media_id} — auto-hiding")
        await client.patch(
            f"{SUPABASE_URL}/rest/v1/published_media",
            params={"id": f"eq.{report.media_id}"},
            json={"moderation_status": "hidden"},
        )

    return ReportResponse(
        id=created["id"],
        media_id=created["media_id"],
        reason=created["reason"],
        status=created["status"],
        created_at=created["created_at"],
    )


# =============================================================================
# Admin Endpoints
# =============================================================================


@admin_router.get("/queue")
async def get_moderation_queue(
    status: str = Query("pending", description="Filter by report status"),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    admin: User = Depends(get_admin_user),
):
    """Get the moderation queue — items with pending reports grouped by media."""
    debug_log(f"Admin {admin.id} viewing moderation queue (status={status})")
    client = _get_client()

    offset = (page - 1) * per_page

    # Get reports with media info, ordered by newest first
    reports_resp = await client.get(
        f"{SUPABASE_URL}/rest/v1/content_reports",
        params={
            "status": f"eq.{status}",
            "select": "id,media_id,reporter_id,reason,description,status,created_at",
            "order": "created_at.desc",
        },
    )

    if reports_resp.status_code != 200:
        logger.error(f"Failed to fetch reports: {reports_resp.status_code}")
        raise HTTPException(status_code=500, detail="Failed to fetch moderation queue")

    all_reports = reports_resp.json()

    # Group reports by media_id
    media_reports = {}
    for r in all_reports:
        mid = r["media_id"]
        if mid not in media_reports:
            media_reports[mid] = []
        media_reports[mid].append(r)

    # Sort by report count (most reported first), then by earliest report
    sorted_media_ids = sorted(
        media_reports.keys(),
        key=lambda mid: (-len(media_reports[mid]), media_reports[mid][0]["created_at"]),
    )

    # Paginate
    total = len(sorted_media_ids)
    page_ids = sorted_media_ids[offset : offset + per_page]

    if not page_ids:
        return {"items": [], "total": total, "page": page, "per_page": per_page}

    # Fetch media details for this page
    # Build OR filter for multiple IDs
    media_filter = ",".join(page_ids)
    media_resp = await client.get(
        f"{SUPABASE_URL}/rest/v1/published_media",
        params={
            "id": f"in.({media_filter})",
            "select": "id,title,media_type,storage_path,is_nsfw,moderation_status,user_id,created_at",
        },
    )

    media_by_id = {}
    if media_resp.status_code == 200:
        for m in media_resp.json():
            media_by_id[m["id"]] = m

    # Build queue items
    items = []
    for mid in page_ids:
        media = media_by_id.get(mid, {})
        reports = media_reports[mid]
        items.append({
            "media_id": mid,
            "title": media.get("title", "Unknown"),
            "media_type": media.get("media_type", "unknown"),
            "storage_path": media.get("storage_path", ""),
            "is_nsfw": media.get("is_nsfw", False),
            "moderation_status": media.get("moderation_status", "approved"),
            "creator_id": media.get("user_id", ""),
            "report_count": len(reports),
            "reports": reports,
            "created_at": media.get("created_at", ""),
        })

    return {"items": items, "total": total, "page": page, "per_page": per_page}


@admin_router.get("/stats")
async def get_moderation_stats(admin: User = Depends(get_admin_user)):
    """Get moderation statistics."""
    debug_log(f"Admin {admin.id} fetching moderation stats")
    client = _get_client()

    # Pending reports count
    pending_resp = await client.get(
        f"{SUPABASE_URL}/rest/v1/content_reports",
        params={"status": "eq.pending", "select": "id"},
        headers={**client.headers, "Prefer": "count=exact"},
    )
    pending = int(pending_resp.headers.get("content-range", "0-0/0").split("/")[-1])

    # Total reports
    total_resp = await client.get(
        f"{SUPABASE_URL}/rest/v1/content_reports",
        params={"select": "id"},
        headers={**client.headers, "Prefer": "count=exact"},
    )
    total_reports = int(total_resp.headers.get("content-range", "0-0/0").split("/")[-1])

    # Today's reviewed
    today = datetime.utcnow().strftime("%Y-%m-%dT00:00:00")
    reviewed_resp = await client.get(
        f"{SUPABASE_URL}/rest/v1/moderation_actions",
        params={"created_at": f"gte.{today}", "select": "id"},
        headers={**client.headers, "Prefer": "count=exact"},
    )
    reviewed_today = int(reviewed_resp.headers.get("content-range", "0-0/0").split("/")[-1])

    # Hidden media count
    hidden_resp = await client.get(
        f"{SUPABASE_URL}/rest/v1/published_media",
        params={"moderation_status": "eq.hidden", "select": "id"},
        headers={**client.headers, "Prefer": "count=exact"},
    )
    total_hidden = int(hidden_resp.headers.get("content-range", "0-0/0").split("/")[-1])

    # Rejected media count
    rejected_resp = await client.get(
        f"{SUPABASE_URL}/rest/v1/published_media",
        params={"moderation_status": "eq.rejected", "select": "id"},
        headers={**client.headers, "Prefer": "count=exact"},
    )
    total_rejected = int(rejected_resp.headers.get("content-range", "0-0/0").split("/")[-1])

    return {
        "pending_reports": pending,
        "reviewed_today": reviewed_today,
        "total_hidden": total_hidden,
        "total_rejected": total_rejected,
        "total_reports": total_reports,
    }


@admin_router.get("/{media_id}")
async def get_media_moderation_detail(
    media_id: str,
    admin: User = Depends(get_admin_user),
):
    """Get detailed moderation info for a specific media item."""
    debug_log(f"Admin {admin.id} viewing moderation detail for {media_id}")
    client = _get_client()

    # Get media info
    media_resp = await client.get(
        f"{SUPABASE_URL}/rest/v1/published_media",
        params={
            "id": f"eq.{media_id}",
            "select": "id,title,description,media_type,storage_path,is_nsfw,moderation_status,user_id,created_at,metadata",
        },
    )
    if media_resp.status_code != 200 or not media_resp.json():
        raise HTTPException(status_code=404, detail="Media not found")

    media = media_resp.json()[0]

    # Get all reports for this media
    reports_resp = await client.get(
        f"{SUPABASE_URL}/rest/v1/content_reports",
        params={
            "media_id": f"eq.{media_id}",
            "select": "id,reporter_id,reason,description,status,created_at,reviewed_at",
            "order": "created_at.desc",
        },
    )
    reports = reports_resp.json() if reports_resp.status_code == 200 else []

    # Get moderation action history
    actions_resp = await client.get(
        f"{SUPABASE_URL}/rest/v1/moderation_actions",
        params={
            "media_id": f"eq.{media_id}",
            "select": "id,moderator_id,action,reason,created_at",
            "order": "created_at.desc",
        },
    )
    actions = actions_resp.json() if actions_resp.status_code == 200 else []

    return {
        "media": media,
        "reports": reports,
        "moderation_history": actions,
    }


@admin_router.post("/{media_id}/action")
async def take_moderation_action(
    media_id: str,
    req: ModerationActionRequest,
    admin: User = Depends(get_admin_user),
):
    """Take a moderation action on a media item."""
    debug_log(f"Admin {admin.id} action '{req.action}' on {media_id}")

    if req.action not in MODERATION_ACTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid action. Must be one of: {MODERATION_ACTIONS}")

    client = _get_client()

    # Verify media exists
    media_resp = await client.get(
        f"{SUPABASE_URL}/rest/v1/published_media",
        params={"id": f"eq.{media_id}", "select": "id,moderation_status"},
    )
    if media_resp.status_code != 200 or not media_resp.json():
        raise HTTPException(status_code=404, detail="Media not found")

    # Map action to moderation_status
    status_map = {
        "approve": "approved",
        "reject": "rejected",
        "hide": "hidden",
        "unhide": "approved",
    }

    # Update media moderation_status if applicable
    new_status = status_map.get(req.action)
    if new_status:
        patch_resp = await client.patch(
            f"{SUPABASE_URL}/rest/v1/published_media",
            params={"id": f"eq.{media_id}"},
            json={"moderation_status": new_status},
        )
        if patch_resp.status_code not in (200, 204):
            logger.error(f"Failed to update media status: {patch_resp.status_code}")
            raise HTTPException(status_code=500, detail="Failed to update media status")

    # If dismissing a specific report or action relates to reports, update them
    if req.report_id:
        await client.patch(
            f"{SUPABASE_URL}/rest/v1/content_reports",
            params={"id": f"eq.{req.report_id}"},
            json={
                "status": "reviewed",
                "reviewed_by": admin.id,
                "reviewed_at": datetime.utcnow().isoformat(),
            },
        )
    elif req.action in ("approve", "reject", "hide"):
        # Resolve all pending reports for this media
        await client.patch(
            f"{SUPABASE_URL}/rest/v1/content_reports",
            params={"media_id": f"eq.{media_id}", "status": "eq.pending"},
            json={
                "status": "reviewed",
                "reviewed_by": admin.id,
                "reviewed_at": datetime.utcnow().isoformat(),
            },
        )

    # Log the action
    action_data = {
        "media_id": media_id,
        "moderator_id": admin.id,
        "action": req.action,
        "reason": req.reason,
        "report_id": req.report_id,
    }
    await client.post(
        f"{SUPABASE_URL}/rest/v1/moderation_actions",
        json=action_data,
    )

    debug_log(f"Action '{req.action}' completed on {media_id}")
    return {"success": True, "action": req.action, "media_id": media_id, "new_status": new_status}


@admin_router.post("/bulk-action")
async def bulk_moderation_action(
    req: BulkActionRequest,
    admin: User = Depends(get_admin_user),
):
    """Apply a moderation action to multiple media items at once."""
    debug_log(f"Admin {admin.id} bulk action '{req.action}' on {len(req.media_ids)} items")

    if req.action not in MODERATION_ACTIONS:
        raise HTTPException(status_code=400, detail=f"Invalid action. Must be one of: {MODERATION_ACTIONS}")

    client = _get_client()
    results = {"success": 0, "failed": 0, "errors": []}

    status_map = {
        "approve": "approved",
        "reject": "rejected",
        "hide": "hidden",
        "unhide": "approved",
    }
    new_status = status_map.get(req.action)

    for media_id in req.media_ids:
        try:
            # Update media status
            if new_status:
                await client.patch(
                    f"{SUPABASE_URL}/rest/v1/published_media",
                    params={"id": f"eq.{media_id}"},
                    json={"moderation_status": new_status},
                )

            # Resolve pending reports
            if req.action in ("approve", "reject", "hide"):
                await client.patch(
                    f"{SUPABASE_URL}/rest/v1/content_reports",
                    params={"media_id": f"eq.{media_id}", "status": "eq.pending"},
                    json={
                        "status": "reviewed",
                        "reviewed_by": admin.id,
                        "reviewed_at": datetime.utcnow().isoformat(),
                    },
                )

            # Log action
            await client.post(
                f"{SUPABASE_URL}/rest/v1/moderation_actions",
                json={
                    "media_id": media_id,
                    "moderator_id": admin.id,
                    "action": req.action,
                    "reason": req.reason,
                },
            )
            results["success"] += 1
        except Exception as e:
            results["failed"] += 1
            results["errors"].append({"media_id": media_id, "error": str(e)})

    return results


@admin_router.get("/log/actions")
async def get_moderation_log(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=100),
    admin: User = Depends(get_admin_user),
):
    """Get the moderation audit log."""
    debug_log(f"Admin {admin.id} viewing moderation log")
    client = _get_client()

    offset = (page - 1) * per_page
    resp = await client.get(
        f"{SUPABASE_URL}/rest/v1/moderation_actions",
        params={
            "select": "id,media_id,moderator_id,action,reason,report_id,created_at",
            "order": "created_at.desc",
            "offset": str(offset),
            "limit": str(per_page),
        },
        headers={**client.headers, "Prefer": "count=exact"},
    )

    if resp.status_code != 200:
        raise HTTPException(status_code=500, detail="Failed to fetch moderation log")

    total = int(resp.headers.get("content-range", "0-0/0").split("/")[-1])
    return {"items": resp.json(), "total": total, "page": page, "per_page": per_page}
