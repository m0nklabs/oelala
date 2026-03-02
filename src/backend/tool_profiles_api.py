#!/usr/bin/env python3
"""
Tool Profiles API for Oelala
Per-user, per-tool settings persistence with auto-save support.

Stores all user-adjustable settings (prompt, resolution, model, sliders, etc.)
as JSONB profiles that auto-save on every change.
"""

import os
import logging
from contextlib import asynccontextmanager
from typing import Optional, List
import httpx
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from auth import get_current_user, User

logger = logging.getLogger(__name__)
DEBUG = os.getenv("OELALA_DEBUG", "0") == "1"

# Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://nsbjwhxdkxnyggtuxjjp.supabase.co")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")

# Valid tool names (extensible)
VALID_TOOLS = {
    "image_to_video",
    "text_to_video",
    "text_to_image",
    "image_upscale",
    "video_extend",
}

# =============================================================================
# Factory Presets — OOTB best-tested settings per tool+mode
# These are hardcoded, not stored in DB. Users can load one as a starting point.
# =============================================================================

FACTORY_PRESETS: dict[str, list[dict]] = {
    "image_to_video": [
        {
            "name": "Wan2.2 Q6 — Best Quality (480p, 8s)",
            "description": "Production-safe Q6_K. 6 steps, ~20GB VRAM, ~6 min gen time.",
            "mode": "wan2.2",
            "settings": {
                "modelMode": "wan2.2",
                "modelVersion": "v2",
                "resolution": "480p",
                "aspectRatio": "9:16",
                "duration": 8,
                "fps": 16,
                "steps": 6,
                "cfg": 1.0,
                "seed": -1,
                "cameraMotion": "",
                "unetHighNoise": "wan2.2_i2v_high_noise_14B_Q6_K.gguf",
                "unetLowNoise": "wan2.2_i2v_low_noise_14B_Q6_K.gguf",
                "loraConfigs": [],
                "extendMode": False,
                "clipCount": 1,
                "postUpscale": False,
                "postUpscaleScale": 2,
                "postInterpolate": False,
                "postInterpolateFps": 60,
                "enhanceModel": "GLM-4.7-Flash-Claude-Opus-Reasoning",
            },
        },
        {
            "name": "Wan2.2 Q6 — Max Length (480p, 20s)",
            "description": "Maximum safe duration at 480p. ~26GB VRAM, ~23 min gen time.",
            "mode": "wan2.2",
            "settings": {
                "modelMode": "wan2.2",
                "modelVersion": "v2",
                "resolution": "480p",
                "aspectRatio": "9:16",
                "duration": 20,
                "fps": 16,
                "steps": 6,
                "cfg": 1.0,
                "seed": -1,
                "cameraMotion": "",
                "unetHighNoise": "wan2.2_i2v_high_noise_14B_Q6_K.gguf",
                "unetLowNoise": "wan2.2_i2v_low_noise_14B_Q6_K.gguf",
                "loraConfigs": [],
                "extendMode": False,
                "clipCount": 1,
                "postUpscale": False,
                "postUpscaleScale": 2,
                "postInterpolate": False,
                "postInterpolateFps": 60,
                "enhanceModel": "GLM-4.7-Flash-Claude-Opus-Reasoning",
            },
        },
        {
            "name": "Wan2.2 Q6 — High Res (576p, 5s)",
            "description": "Higher resolution, shorter clip. ~24GB VRAM.",
            "mode": "wan2.2",
            "settings": {
                "modelMode": "wan2.2",
                "modelVersion": "v2",
                "resolution": "576p",
                "aspectRatio": "9:16",
                "duration": 5,
                "fps": 16,
                "steps": 6,
                "cfg": 1.0,
                "seed": -1,
                "cameraMotion": "",
                "unetHighNoise": "wan2.2_i2v_high_noise_14B_Q6_K.gguf",
                "unetLowNoise": "wan2.2_i2v_low_noise_14B_Q6_K.gguf",
                "loraConfigs": [],
                "extendMode": False,
                "clipCount": 1,
                "postUpscale": False,
                "postUpscaleScale": 2,
                "postInterpolate": False,
                "postInterpolateFps": 60,
                "enhanceModel": "GLM-4.7-Flash-Claude-Opus-Reasoning",
            },
        },
        {
            "name": "Wan2.2 Lightning — Fast (480p, 8s)",
            "description": "Lightning UNETs for faster generation. Same VRAM.",
            "mode": "wan2.2",
            "settings": {
                "modelMode": "wan2.2",
                "modelVersion": "v2",
                "resolution": "480p",
                "aspectRatio": "9:16",
                "duration": 8,
                "fps": 16,
                "steps": 6,
                "cfg": 1.0,
                "seed": -1,
                "cameraMotion": "",
                "unetHighNoise": "Wan22-I2V_A14B-Lightning-H-Q6_K.gguf",
                "unetLowNoise": "Wan22-I2V_A14B-Lightning-L-Q6_K.gguf",
                "loraConfigs": [],
                "extendMode": False,
                "clipCount": 1,
                "postUpscale": False,
                "postUpscaleScale": 2,
                "postInterpolate": False,
                "postInterpolateFps": 60,
                "enhanceModel": "GLM-4.7-Flash-Claude-Opus-Reasoning",
            },
        },
        {
            "name": "BlockSwap Q8 — Max Quality (480p, 10s)",
            "description": "Q8_0 precision + NAG + EnhanceAVideo. Best visual quality.",
            "mode": "blockswap_q8",
            "settings": {
                "modelMode": "blockswap_q8",
                "resolution": "480p",
                "aspectRatio": "9:16",
                "duration": 10,
                "fps": 16,
                "steps": 8,
                "cfg": 1.0,
                "seed": -1,
                "cameraMotion": "",
                "bsShift": 9.0,
                "bsNagScale": 11.0,
                "bsEnableFlorence2": True,
                "bsEnableUpscale": False,
                "bsEnableInterpolation": False,
                "bsHighNoiseSteps": 4,
                "loraConfigs": [],
                "extendMode": False,
                "clipCount": 1,
                "postUpscale": False,
                "postUpscaleScale": 2,
                "postInterpolate": False,
                "postInterpolateFps": 60,
                "enhanceModel": "GLM-4.7-Flash-Claude-Opus-Reasoning",
            },
        },
        {
            "name": "BlockSwap Q8 — Full Pipeline (720p upscaled)",
            "description": "Q8 + Florence2 auto-caption + 4x upscale + 2x interpolation.",
            "mode": "blockswap_q8",
            "settings": {
                "modelMode": "blockswap_q8",
                "resolution": "480p",
                "aspectRatio": "9:16",
                "duration": 8,
                "fps": 16,
                "steps": 8,
                "cfg": 1.0,
                "seed": -1,
                "cameraMotion": "",
                "bsShift": 9.0,
                "bsNagScale": 11.0,
                "bsEnableFlorence2": True,
                "bsEnableUpscale": True,
                "bsEnableInterpolation": True,
                "bsHighNoiseSteps": 4,
                "loraConfigs": [],
                "extendMode": False,
                "clipCount": 1,
                "postUpscale": False,
                "postUpscaleScale": 2,
                "postInterpolate": False,
                "postInterpolateFps": 60,
                "enhanceModel": "GLM-4.7-Flash-Claude-Opus-Reasoning",
            },
        },
        {
            "name": "DisTorch2 Q8 — Best Quality (480p, 10s)",
            "description": "Q8_0 + DisTorch2 multi-GPU. Same quality, better VRAM mgmt.",
            "mode": "distorch2_q8",
            "settings": {
                "modelMode": "distorch2_q8",
                "resolution": "480p",
                "aspectRatio": "9:16",
                "duration": 10,
                "fps": 16,
                "steps": 8,
                "cfg": 1.0,
                "seed": -1,
                "cameraMotion": "",
                "bsShift": 9.0,
                "bsNagScale": 11.0,
                "bsEnableFlorence2": True,
                "bsEnableUpscale": False,
                "bsEnableInterpolation": False,
                "bsHighNoiseSteps": 4,
                "loraConfigs": [],
                "extendMode": False,
                "clipCount": 1,
                "postUpscale": False,
                "postUpscaleScale": 2,
                "postInterpolate": False,
                "postInterpolateFps": 60,
                "enhanceModel": "GLM-4.7-Flash-Claude-Opus-Reasoning",
            },
        },
        {
            "name": "LTX-2 — Quick Preview (576p, 5s)",
            "description": "LTX-2 19B distilled. Fast previews, lower quality. ~18GB VRAM.",
            "mode": "ltx2",
            "settings": {
                "modelMode": "ltx2",
                "resolution": "576p",
                "aspectRatio": "9:16",
                "duration": 5,
                "fps": 25,
                "steps": 20,
                "cfg": 3.0,
                "seed": -1,
                "cameraMotion": "",
                "loraConfigs": [],
                "extendMode": False,
                "clipCount": 1,
                "postUpscale": False,
                "postUpscaleScale": 2,
                "postInterpolate": False,
                "postInterpolateFps": 60,
                "enhanceModel": "GLM-4.7-Flash-Claude-Opus-Reasoning",
            },
        },
    ],
    "text_to_video": [],  # TODO: add when T2V tool is ready
    "text_to_image": [],  # TODO: add when T2I presets are defined
    "image_upscale": [],  # TODO: add when upscale presets are defined
    "video_extend": [],  # TODO: add when extend presets are defined
}


# Max profiles per user per tool
MAX_PROFILES_PER_TOOL = 20
# Max settings payload size (100KB)
MAX_SETTINGS_SIZE = 102_400


def debug_log(msg: str):
    if DEBUG:
        logger.info(f"⚙️ TOOL_PROFILE: {msg}")


# =============================================================================
# Pydantic Models
# =============================================================================


class ToolProfileSettings(BaseModel):
    """Auto-save settings payload (any JSON object)"""

    settings: dict = Field(
        default_factory=dict, description="Tool settings as JSON object"
    )


class ToolProfileCreate(BaseModel):
    """Create a named profile snapshot"""

    profile_name: str = Field(
        ...,
        min_length=1,
        max_length=50,
        description="Profile name (e.g., 'my_best_settings', 'test_run_3')",
    )
    settings: dict = Field(
        default_factory=dict, description="Tool settings as JSON object"
    )


class ToolProfileResponse(BaseModel):
    """Single profile response"""

    id: str
    tool_name: str
    profile_name: str
    settings: dict
    is_active: bool
    created_at: str
    updated_at: str


class ToolProfileListResponse(BaseModel):
    """List of profiles for a tool"""

    profiles: List[ToolProfileResponse]
    active_profile: Optional[str] = None


class FactoryPreset(BaseModel):
    """A built-in preset with best-tested settings"""

    name: str
    description: str
    mode: str
    settings: dict


class FactoryPresetsResponse(BaseModel):
    """List of factory presets for a tool"""

    presets: List[FactoryPreset]


# =============================================================================
# Router
# =============================================================================

router = APIRouter(prefix="/api/settings", tags=["tool-profiles"])


# =============================================================================
# Supabase Client (shared singleton, same pattern as profile_api.py)
# =============================================================================

_supabase_client: Optional[httpx.AsyncClient] = None


@asynccontextmanager
async def get_supabase_client():
    """Get shared Supabase REST API client (singleton with connection pooling)."""
    global _supabase_client
    if _supabase_client is None or _supabase_client.is_closed:
        _supabase_client = httpx.AsyncClient(
            base_url=f"{SUPABASE_URL}/rest/v1",
            headers={
                "apikey": SUPABASE_SERVICE_KEY,
                "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            },
            timeout=30.0,
        )
    yield _supabase_client


def _validate_tool_name(tool_name: str) -> str:
    """Validate tool name against allowed list."""
    if tool_name not in VALID_TOOLS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid tool name '{tool_name}'. Valid tools: {sorted(VALID_TOOLS)}",
        )
    return tool_name


def _validate_settings_size(settings: dict):
    """Prevent oversized settings payloads."""
    import json

    size = len(json.dumps(settings))
    if size > MAX_SETTINGS_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Settings payload too large ({size} bytes, max {MAX_SETTINGS_SIZE})",
        )


# =============================================================================
# Endpoints
# =============================================================================


@router.get("/{tool_name}", response_model=Optional[ToolProfileResponse])
async def get_active_profile(tool_name: str, user: User = Depends(get_current_user)):
    """
    Get the active profile for a tool.
    Returns the active profile, or the 'default' profile,
    or null if no profile exists yet.
    """
    _validate_tool_name(tool_name)

    async with get_supabase_client() as client:
        # Try active profile first
        response = await client.get(
            "/tool_profiles",
            params={
                "user_id": f"eq.{user.id}",
                "tool_name": f"eq.{tool_name}",
                "is_active": "eq.true",
                "select": "*",
                "limit": "1",
            },
        )

        if response.status_code == 200 and response.json():
            profile = response.json()[0]
            debug_log(
                f"Active profile for {user.id}/{tool_name}: {profile['profile_name']}"
            )
            return ToolProfileResponse(**profile)

        # Fallback to 'default' profile
        response = await client.get(
            "/tool_profiles",
            params={
                "user_id": f"eq.{user.id}",
                "tool_name": f"eq.{tool_name}",
                "profile_name": "eq.default",
                "select": "*",
                "limit": "1",
            },
        )

        if response.status_code == 200 and response.json():
            profile = response.json()[0]
            debug_log(f"Default profile for {user.id}/{tool_name}")
            return ToolProfileResponse(**profile)

        debug_log(f"No profile found for {user.id}/{tool_name}")
        return None


@router.get("/{tool_name}/presets", response_model=FactoryPresetsResponse)
async def get_factory_presets(tool_name: str):
    """
    Get built-in factory presets for a tool.
    No auth required — these are public best-tested settings.
    """
    _validate_tool_name(tool_name)
    presets = FACTORY_PRESETS.get(tool_name, [])
    debug_log(f"Factory presets for {tool_name}: {len(presets)} available")
    return FactoryPresetsResponse(presets=[FactoryPreset(**p) for p in presets])


@router.put("/{tool_name}", response_model=ToolProfileResponse)
async def save_active_profile(
    tool_name: str,
    body: ToolProfileSettings,
    user: User = Depends(get_current_user),
):
    """
    Auto-save the active profile for a tool.
    Creates 'default' profile if none exists.
    Called on every settings change (frontend debounces).
    """
    _validate_tool_name(tool_name)
    _validate_settings_size(body.settings)

    async with get_supabase_client() as client:
        # Find active profile (or default)
        response = await client.get(
            "/tool_profiles",
            params={
                "user_id": f"eq.{user.id}",
                "tool_name": f"eq.{tool_name}",
                "or": "(is_active.eq.true,profile_name.eq.default)",
                "order": "is_active.desc",
                "select": "id,profile_name,is_active",
                "limit": "1",
            },
        )

        if response.status_code == 200 and response.json():
            # Update existing profile
            existing = response.json()[0]
            update_resp = await client.patch(
                "/tool_profiles",
                params={
                    "id": f"eq.{existing['id']}",
                },
                json={"settings": body.settings},
            )

            if update_resp.status_code in (200, 204) and update_resp.json():
                profile = update_resp.json()[0]
                debug_log(f"Updated profile {existing['profile_name']} for {tool_name}")
                return ToolProfileResponse(**profile)
            else:
                logger.error(
                    f"Failed to update profile: {update_resp.status_code} {update_resp.text}"
                )
                raise HTTPException(status_code=500, detail="Failed to update profile")
        else:
            # Create default profile
            create_resp = await client.post(
                "/tool_profiles",
                json={
                    "user_id": user.id,
                    "tool_name": tool_name,
                    "profile_name": "default",
                    "settings": body.settings,
                    "is_active": True,
                },
            )

            if create_resp.status_code in (200, 201) and create_resp.json():
                profile = create_resp.json()[0]
                debug_log(f"Created default profile for {tool_name}")
                return ToolProfileResponse(**profile)
            else:
                logger.error(
                    f"Failed to create profile: {create_resp.status_code} {create_resp.text}"
                )
                raise HTTPException(status_code=500, detail="Failed to create profile")


@router.get("/{tool_name}/profiles", response_model=ToolProfileListResponse)
async def list_profiles(tool_name: str, user: User = Depends(get_current_user)):
    """List all profiles for a tool."""
    _validate_tool_name(tool_name)

    async with get_supabase_client() as client:
        response = await client.get(
            "/tool_profiles",
            params={
                "user_id": f"eq.{user.id}",
                "tool_name": f"eq.{tool_name}",
                "select": "*",
                "order": "created_at.asc",
            },
        )

        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Failed to list profiles")

        profiles = response.json()
        active = next((p["profile_name"] for p in profiles if p["is_active"]), None)

        return ToolProfileListResponse(
            profiles=[ToolProfileResponse(**p) for p in profiles],
            active_profile=active,
        )


@router.post("/{tool_name}/profiles", response_model=ToolProfileResponse)
async def create_profile(
    tool_name: str,
    body: ToolProfileCreate,
    user: User = Depends(get_current_user),
):
    """Create a named profile snapshot (save current settings under a name)."""
    _validate_tool_name(tool_name)
    _validate_settings_size(body.settings)

    if body.profile_name.lower() == "default":
        raise HTTPException(
            status_code=400,
            detail="Cannot create profile named 'default'. Use PUT /{tool_name} instead.",
        )

    async with get_supabase_client() as client:
        # Check profile count
        count_resp = await client.get(
            "/tool_profiles",
            params={
                "user_id": f"eq.{user.id}",
                "tool_name": f"eq.{tool_name}",
                "select": "id",
            },
            headers={"Prefer": "count=exact"},
        )
        # Parse count from content-range header
        content_range = count_resp.headers.get("content-range", "")
        total = (
            int(content_range.split("/")[-1])
            if "/" in content_range
            else len(count_resp.json())
        )

        if total >= MAX_PROFILES_PER_TOOL:
            raise HTTPException(
                status_code=400,
                detail=f"Maximum {MAX_PROFILES_PER_TOOL} profiles per tool reached",
            )

        # Upsert profile (Supabase ON CONFLICT)
        create_resp = await client.post(
            "/tool_profiles",
            json={
                "user_id": user.id,
                "tool_name": tool_name,
                "profile_name": body.profile_name,
                "settings": body.settings,
                "is_active": False,
            },
            headers={
                "Prefer": "return=representation,resolution=merge-duplicates",
            },
        )

        if create_resp.status_code in (200, 201) and create_resp.json():
            profile = create_resp.json()[0]
            debug_log(f"Created profile '{body.profile_name}' for {tool_name}")
            return ToolProfileResponse(**profile)
        else:
            logger.error(
                f"Failed to create profile: {create_resp.status_code} {create_resp.text}"
            )
            raise HTTPException(status_code=500, detail="Failed to create profile")


@router.put(
    "/{tool_name}/profiles/{profile_name}/activate", response_model=ToolProfileResponse
)
async def activate_profile(
    tool_name: str,
    profile_name: str,
    user: User = Depends(get_current_user),
):
    """Switch to a different profile (loads its settings)."""
    _validate_tool_name(tool_name)

    async with get_supabase_client() as client:
        # Find the profile to activate
        response = await client.get(
            "/tool_profiles",
            params={
                "user_id": f"eq.{user.id}",
                "tool_name": f"eq.{tool_name}",
                "profile_name": f"eq.{profile_name}",
                "select": "*",
                "limit": "1",
            },
        )

        if response.status_code != 200 or not response.json():
            raise HTTPException(
                status_code=404, detail=f"Profile '{profile_name}' not found"
            )

        profile_id = response.json()[0]["id"]

        # The DB trigger handles deactivating other profiles
        activate_resp = await client.patch(
            "/tool_profiles",
            params={"id": f"eq.{profile_id}"},
            json={"is_active": True},
        )

        if activate_resp.status_code in (200, 204) and activate_resp.json():
            profile = activate_resp.json()[0]
            debug_log(f"Activated profile '{profile_name}' for {tool_name}")
            return ToolProfileResponse(**profile)
        else:
            raise HTTPException(status_code=500, detail="Failed to activate profile")


@router.delete("/{tool_name}/profiles/{profile_name}")
async def delete_profile(
    tool_name: str,
    profile_name: str,
    user: User = Depends(get_current_user),
):
    """Delete a named profile. Cannot delete the 'default' profile."""
    _validate_tool_name(tool_name)

    if profile_name.lower() == "default":
        raise HTTPException(status_code=400, detail="Cannot delete the default profile")

    async with get_supabase_client() as client:
        response = await client.delete(
            "/tool_profiles",
            params={
                "user_id": f"eq.{user.id}",
                "tool_name": f"eq.{tool_name}",
                "profile_name": f"eq.{profile_name}",
            },
        )

        if response.status_code not in (200, 204):
            raise HTTPException(status_code=500, detail="Failed to delete profile")

        debug_log(f"Deleted profile '{profile_name}' for {tool_name}")
        return {"deleted": True, "profile_name": profile_name}
