#!/usr/bin/env python3
"""
Guardian VRAM Management Client

Manages LLM VRAM via the Guardian proxy admin API.
Before ComfyUI generation: unload the model to free VRAM.
After generation: Guardian auto-reloads on the next inference request.

Endpoints (llama_cpp_guardian):
  POST /admin/unload  → free VRAM immediately
  POST /admin/load    → force-reload the pinned model

Config (via environment):
  GUARDIAN_BASE_URL : Base URL (default: http://localhost:11434)
  GUARDIAN_API_KEY  : Bearer token for /admin/* endpoints
"""

import logging
import os

import httpx

logger = logging.getLogger(__name__)

DEBUG_ENABLED = bool(int(os.getenv("OELALA_DEBUG", "0")))


def _debug(msg: str) -> None:
    if DEBUG_ENABLED:
        logger.debug(f"🛡️ [guardian] {msg}")


class GuardianVRAMClient:
    """
    Lightweight async client for Guardian LLM proxy VRAM management.

    Usage (fire-and-forget):
        client = GuardianVRAMClient()
        await client.unload()   # before ComfyUI generation
        # ... generation happens ...
        # no explicit reload needed — Guardian auto-reloads on next inference
    """

    def __init__(
        self,
        base_url: str | None = None,
        admin_token: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("GUARDIAN_BASE_URL", "http://localhost:11434")).rstrip("/")
        self.admin_token = admin_token or os.getenv("GUARDIAN_API_KEY", "")
        self.timeout = timeout

        if not self.admin_token:
            logger.warning("⚠️ [guardian] GUARDIAN_API_KEY not set — VRAM management disabled")

    @property
    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.admin_token}"}

    @property
    def _enabled(self) -> bool:
        return bool(self.admin_token)

    async def unload(self) -> bool:
        """
        Unload the LLM from VRAM immediately.
        Returns True on success, False on failure (non-fatal — generation continues either way).
        """
        if not self._enabled:
            _debug("Skipping unload — no admin token configured")
            return False

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.base_url}/admin/unload", headers=self._headers)

            if resp.status_code == 200:
                data = resp.json()
                logger.info(f"🛡️ Guardian: LLM unloaded — {data.get('message', 'VRAM free')}")
                _debug(f"unload response: {data}")
                return True
            else:
                logger.warning(f"⚠️ [guardian] Unload returned {resp.status_code}: {resp.text}")
                return False

        except httpx.ConnectError:
            logger.warning("⚠️ [guardian] Cannot reach Guardian proxy — skipping unload")
            return False
        except Exception as exc:
            logger.warning(f"⚠️ [guardian] Unload error: {exc}")
            return False

    async def load(self) -> bool:
        """
        Force-reload the pinned model. Usually not needed — Guardian auto-reloads.
        Useful for explicit warm-up after a long generation.
        """
        if not self._enabled:
            return False

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(f"{self.base_url}/admin/load", headers=self._headers)

            if resp.status_code == 200:
                data = resp.json()
                logger.info(f"🛡️ Guardian: LLM loaded — {data.get('model', 'unknown')}")
                _debug(f"load response: {data}")
                return True
            else:
                logger.warning(f"⚠️ [guardian] Load returned {resp.status_code}: {resp.text}")
                return False

        except httpx.ConnectError:
            logger.warning("⚠️ [guardian] Cannot reach Guardian proxy — skipping load")
            return False
        except Exception as exc:
            logger.warning(f"⚠️ [guardian] Load error: {exc}")
            return False

    def unload_sync(self) -> bool:
        """
        Synchronous version of unload() for use in sync/mixed contexts (e.g.
        from comfyui_client.queue_prompt which is called from async FastAPI handlers).
        Uses sync httpx directly to avoid RuntimeError from nested event loops.
        """
        if not self._enabled:
            return False

        try:
            resp = httpx.post(
                f"{self.base_url}/admin/unload",
                headers=self._headers,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                logger.info(f"🛡️ Guardian: LLM unloaded — {data.get('message', 'VRAM free')}")
                return True
            else:
                logger.warning(f"⚠️ [guardian] Unload returned {resp.status_code}: {resp.text}")
                return False
        except httpx.ConnectError:
            logger.warning("⚠️ [guardian] Cannot reach Guardian proxy — skipping unload")
            return False
        except Exception as exc:
            logger.warning(f"⚠️ [guardian] Unload error: {exc}")
            return False


# Module-level singleton — import and use directly
_guardian: GuardianVRAMClient | None = None


def get_guardian() -> GuardianVRAMClient:
    """Get (or create) the module-level Guardian client singleton."""
    global _guardian
    if _guardian is None:
        _guardian = GuardianVRAMClient()
    return _guardian


# ---------------------------------------------------------------------------
# ComfyUI VRAM helpers — free loaded models before LLM inference
# ---------------------------------------------------------------------------


async def is_comfyui_busy(comfyui_url: str | None = None) -> bool:
    """
    Check if ComfyUI has any running or pending jobs.
    Returns True if busy, False if idle.
    """
    url = comfyui_url or os.getenv("COMFYUI_BASE_URL", "http://localhost:8188")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{url}/queue")
            if resp.status_code == 200:
                data = resp.json()
                running = len(data.get("queue_running", []))
                pending = len(data.get("queue_pending", []))
                return running > 0 or pending > 0
    except Exception:
        pass
    return False


async def wait_for_comfyui_idle(
    comfyui_url: str | None = None,
    poll_interval: float = 5.0,
    max_wait: float = 1800.0,  # 30 min max
) -> bool:
    """
    Poll ComfyUI queue until no jobs are running or pending.
    Returns True when idle, False if timed out.
    """
    import asyncio

    url = comfyui_url or os.getenv("COMFYUI_BASE_URL", "http://localhost:8188")
    elapsed = 0.0

    if not await is_comfyui_busy(url):
        return True

    logger.info("⏳ [guardian] ComfyUI is busy — waiting for generation to complete before LLM call...")

    while elapsed < max_wait:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval

        if not await is_comfyui_busy(url):
            logger.info(f"✅ [guardian] ComfyUI idle after {elapsed:.0f}s — proceeding with LLM call")
            return True

        if int(elapsed) % 30 == 0:
            _debug(f"Still waiting for ComfyUI... ({elapsed:.0f}s elapsed)")

    logger.warning(f"⚠️ [guardian] Timed out waiting for ComfyUI after {max_wait:.0f}s")
    return False

async def free_comfyui_vram(comfyui_url: str | None = None) -> bool:
    """
    POST to ComfyUI /free to unload all models from VRAM.

    Call this before any LLM inference so the GPU has headroom.
    ComfyUI will reload models automatically on the next queue submission.
    Returns True on success, False if ComfyUI is unreachable (non-fatal).
    """
    url = comfyui_url or os.getenv("COMFYUI_BASE_URL", "http://localhost:8188")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{url}/free",
                json={"unload_models": True, "free_memory": True},
            )
            if resp.status_code == 200:
                logger.info("🖥️ [guardian] ComfyUI VRAM freed before LLM call")
                return True
            logger.warning(f"⚠️ [guardian] ComfyUI /free returned {resp.status_code}")
            return False
    except httpx.ConnectError:
        logger.warning("⚠️ [guardian] Cannot reach ComfyUI to free VRAM — skipping")
        return False
    except Exception as exc:
        logger.warning(f"⚠️ [guardian] ComfyUI free_vram error: {exc}")
        return False


def free_comfyui_vram_sync(comfyui_url: str | None = None) -> bool:
    """Sync version of free_comfyui_vram for use in non-async contexts."""
    url = comfyui_url or os.getenv("COMFYUI_BASE_URL", "http://localhost:8188")
    try:
        with httpx.Client(timeout=10.0) as client:
            resp = client.post(
                f"{url}/free",
                json={"unload_models": True, "free_memory": True},
            )
            if resp.status_code == 200:
                logger.info("🖥️ [guardian] ComfyUI VRAM freed before LLM call (sync)")
                return True
            logger.warning(f"⚠️ [guardian] ComfyUI /free returned {resp.status_code} (sync)")
            return False
    except httpx.ConnectError:
        logger.warning("⚠️ [guardian] Cannot reach ComfyUI to free VRAM (sync) — skipping")
        return False
    except Exception as exc:
        logger.warning(f"⚠️ [guardian] ComfyUI free_vram_sync error: {exc}")
        return False
