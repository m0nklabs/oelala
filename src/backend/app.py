#!/usr/bin/env python3
"""
Oelala Web Interface Backend
FastAPI application for AI Video Generation Pipeline
"""

import io
import html as html_lib
import ipaddress
import os
import posixpath
import socket
import sys
import time

# Load environment variables from .env file BEFORE any other imports
# This must happen early so other modules get the env vars
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/flip/oelala/.env")

# ── Sentry SDK (must init before FastAPI) ────────────────────────────
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.starlette import StarletteIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

_sentry_dsn = os.getenv("SENTRY_DSN", "")
if _sentry_dsn:
    sentry_sdk.init(
        dsn=_sentry_dsn,
        environment=os.getenv("SENTRY_ENVIRONMENT", "production"),
        traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.2")),
        profiles_sample_rate=float(os.getenv("SENTRY_PROFILES_SAMPLE_RATE", "0.1")),
        send_default_pii=False,
        integrations=[
            FastApiIntegration(transaction_style="endpoint"),
            StarletteIntegration(transaction_style="endpoint"),
            LoggingIntegration(level=None, event_level="ERROR"),
        ],
        # Filter out health check noise
        before_send_transaction=lambda event, hint: (
            None if event.get("transaction") in ("/health", "/health/") else event
        ),
    )
    print(
        f"✅ Sentry initialized (env={os.getenv('SENTRY_ENVIRONMENT', 'production')})"
    )
else:
    print("ℹ️  Sentry disabled (SENTRY_DSN not set)")

import uvicorn
import threading
import asyncio
from fastapi import (
    FastAPI,
    File,
    UploadFile,
    Form,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Depends,
    Request,
    Query,
)
from fastapi.responses import (
    FileResponse,
    JSONResponse,
    StreamingResponse,
    HTMLResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import httpx
import shutil
from pathlib import Path
import logging
from datetime import datetime, timezone
import json
import re
from urllib.parse import quote, unquote, urlparse
from collections import deque
import uuid
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("/home/flip/oelala")  # Add oelala root directory

# Authentication
from auth import (
    get_current_user,
    get_optional_user,
    User,
    decode_jwt_with_secret,
    decode_jwt_with_jwks,
)

# Storage client for user media (MinIO-backed)
from storage_client import get_client as get_storage_client
from storage_utils import parse_range_header, format_last_modified, ALLOWED_ORIGINS
from minio.error import S3Error

# MediaService for MinIO + Supabase integration (async client)
from media_service import MediaService, MediaRecord

# Generation artifact storage (workflow, settings, logs per generation)
from gen_artifacts import (
    save_gen_start_artifacts,
    save_gen_logs,
    format_comfyui_history_log,
)

# Credits system
from credits import calculate_credits
from credits_api import (
    router as credits_router,
    stripe_router,
    check_credits,
    deduct_credits,
)
from credits import get_credit_manager

# Gallery system
from gallery_api import router as gallery_router

# Profile system
from profile_api import router as profile_router

# Admin system
from admin_api import router as admin_router, check_admin

# LoRA browser
from lora_api import router as lora_router

# Webhooks system
from webhooks_api import router as webhooks_router

# Content moderation
from moderation_api import (
    public_router as moderation_public_router,
    admin_router as moderation_admin_router,
)
from webhook_service import webhook_service

# Face swap / face profile service (insightface-based, no ComfyUI)
try:
    import face_service

    print("✅ face_service imported successfully")
except ImportError as e:
    print(f"⚠️ face_service import failed (insightface not available?): {e}")
    face_service = None

try:
    import face_train_service

    print("✅ face_train_service imported successfully")
    # Recover any training jobs orphaned by previous backend restarts
    recovered = face_train_service.recover_stuck_jobs()
    if recovered:
        print(f"🔄 Recovered {recovered} orphaned training job(s)")
except ImportError as e:
    print(f"⚠️ face_train_service import failed: {e}")
    face_train_service = None

# ComfyUI Client for all image/video generation
try:
    from src.backend.comfyui_client import (
        ComfyUIClient,
        get_comfyui_client,
        get_windows_comfyui_client,
    )

    print("✅ ComfyUIClient imported successfully")
except ImportError as e:
    print(f"❌ Failed to import ComfyUIClient: {e}")
    ComfyUIClient = None
    get_comfyui_client = None
    get_windows_comfyui_client = None

# WebSocket progress tracking
try:
    from websocket_handler import ws_manager
    from job_queue import job_queue_manager
    from comfyui_progress_monitor import progress_monitor

    print("✅ WebSocket progress modules imported successfully")
except ImportError as e:
    print(f"❌ Failed to import WebSocket progress modules: {e}")
    ws_manager = None
    job_queue_manager = None
    progress_monitor = None

# LLM prompt enhancement queue
try:
    from llm_queue import llm_queue_manager

    print("✅ LLM queue manager imported successfully")
except ImportError as e:
    print(f"❌ Failed to import LLM queue manager: {e}")
    llm_queue_manager = None

# RunPod Serverless client for cloud GPU offloading
try:
    from runpod_client import get_runpod_client, RunPodJobStatus

    _runpod = get_runpod_client()
    if _runpod.is_available():
        print("✅ RunPod client initialized (cloud GPU available)")
    else:
        print("⚠️ RunPod client: no API key (cloud GPU disabled)")
        _runpod = None
except ImportError as e:
    print(f"⚠️ RunPod client not available: {e}")
    _runpod = None
    RunPodJobStatus = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _strip_think_tags(text: str) -> str:
    """Strip <think>...</think> reasoning blocks from LLM output.
    Handles both closed and unclosed <think> tags.
    If the ENTIRE content is inside think tags, extract it rather than deleting everything."""
    if not text or not text.strip():
        return text
    # First strip closed tags
    stripped = re.sub(r"<think>[\s\S]*?</think>", "", text)
    # Then strip unclosed <think> (tag to end of string)
    stripped = re.sub(r"<think>[\s\S]*$", "", stripped)
    stripped = stripped.strip()
    # If stripping removed everything but original had content, the actual
    # content might have been INSIDE the think tags — extract it
    if not stripped and len(text) > 20:
        # Try extracting JSON from inside the think tags
        inner = re.sub(r"</?think>", "", text).strip()
        if inner:
            return inner
    return stripped


# ─── Wire face_train_service into WebSocket progress broadcasts ──────────────
if face_train_service and ws_manager:
    import asyncio as _asyncio

    def _training_progress_callback(job_id: str, event: str, data: dict):
        """Bridge face training thread events to async WebSocket broadcasts."""
        prompt_id = f"train_{job_id}"
        try:
            loop = _asyncio.get_event_loop()
        except RuntimeError:
            return  # No event loop available

        payload = {"job_id": job_id, "prompt_id": prompt_id, "status": event, **data}
        if event == "started":
            payload["status"] = "running"
        elif event == "progress":
            payload["status"] = "running"
        # "completed" and "failed" map directly

        coro = ws_manager.broadcast_to_all("training_update", payload)
        _asyncio.run_coroutine_threadsafe(coro, loop)

    face_train_service.set_progress_callback(_training_progress_callback)
    print("✅ face_train_service wired to WebSocket broadcasts")

# Log Buffer for UI
log_buffer = deque(maxlen=1000)  # Increased buffer size for shell output
progress_store = {}  # job_id -> {progress, status, message, updated_at}
ticker_store = {}  # job_id -> threading.Event to stop ticker

# WebSocket connections for live log streaming
log_subscribers: set[WebSocket] = set()

# Global debug switch for verbose backend traces
DEBUG_ENABLED = os.getenv("OELALA_DEBUG", "0") == "1"

# WebSocket and polling configuration
WEBSOCKET_AUTH_TIMEOUT = 5.0  # seconds - timeout for WebSocket authentication
QUEUE_POLLING_INTERVAL = 2.0  # seconds - interval for ComfyUI queue polling

# Global MediaService instance (initialized lazily)
_media_service: Optional[MediaService] = None


def get_media_service() -> MediaService:
    """Get or create the global MediaService instance."""
    global _media_service
    if _media_service is None:
        _media_service = MediaService()
        logger.info("🗄️ MediaService initialized")
    return _media_service


def debug_log(message: str):
    """Emit debug logs when DEBUG_ENABLED is true."""
    if DEBUG_ENABLED:
        logger.info(f"🐛 {message}")


def start_progress_ticker(
    job_id: str, step: int = 5, interval: float = 2.0, ceiling: int = 95
):
    """Start a background ticker that nudges progress up to a ceiling while job is running."""
    debug_log(f"🔍 starting progress ticker for job {job_id}")
    stop_event = threading.Event()
    ticker_store[job_id] = stop_event

    def _tick():
        while not stop_event.is_set():
            record = progress_store.get(job_id)
            if not record or record.get("status") != "running":
                break
            pct = min(ceiling, record.get("progress", 0) + step)
            progress_store[job_id] = {
                **record,
                "progress": pct,
                "updated_at": datetime.now().isoformat(),
            }
            stop_event.wait(interval)
        debug_log(f"✅ progress ticker finished for job {job_id}")
        ticker_store.pop(job_id, None)

    threading.Thread(target=_tick, daemon=True).start()


def stop_progress_ticker(job_id: str):
    """Stop and clean up ticker for a job."""
    event = ticker_store.pop(job_id, None)
    if event:
        debug_log(f"⚠️ stopping progress ticker for job {job_id}")
        event.set()
    else:
        debug_log(f"⚠️ no active ticker found for job {job_id}")


def inject_png_workflow_metadata(
    image_path: str, workflow: dict, prompt_params: dict
) -> bool:
    """
    Inject ComfyUI-compatible workflow metadata into a PNG file.
    This allows ComfyUI to read the workflow when opening the image.
    Preserves existing T2I prompt metadata if present.

    Args:
        image_path: Path to the PNG file
        workflow: The ComfyUI API workflow dict
        prompt_params: Additional prompt parameters for reference

    Returns:
        True if successful, False otherwise
    """
    try:
        img = Image.open(image_path)

        # Check for existing metadata (e.g., T2I prompt from original image)
        existing_info = img.info if hasattr(img, "info") else {}

        # Try to extract original T2I prompt from existing workflow
        original_t2i_prompt = None
        if "prompt" in existing_info:
            try:
                existing_wf = json.loads(existing_info["prompt"])
                for node_id, node in existing_wf.items():
                    if isinstance(node, dict):
                        inputs = node.get("inputs", {})
                        # CLIPTextEncode has long T2I prompts
                        if "text" in inputs and isinstance(inputs["text"], str):
                            text = inputs["text"]
                            if len(text) > 50:  # Long prompts are likely T2I
                                original_t2i_prompt = text
                                break
            except json.JSONDecodeError as e:
                logger.debug(
                    f"🐛 Failed to decode existing prompt metadata as JSON: {e}"
                )

        # Create PNG metadata
        metadata = PngInfo()

        # ComfyUI expects 'prompt' to contain the API workflow
        metadata.add_text("prompt", json.dumps(workflow))

        # Add extra info for reference
        metadata.add_text(
            "workflow", json.dumps(workflow)
        )  # Some versions look for this

        # Preserve original T2I prompt if we found one
        if original_t2i_prompt:
            prompt_params = dict(prompt_params)  # Don't modify original
            prompt_params["original_t2i_prompt"] = original_t2i_prompt
            logger.info(
                f"📝 Preserved original T2I prompt ({len(original_t2i_prompt)} chars)"
            )

        metadata.add_text("oelala_params", json.dumps(prompt_params))

        # Save with metadata
        img.save(image_path, pnginfo=metadata)
        logger.info(f"📝 Injected workflow metadata into {image_path}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Failed to inject PNG metadata: {e}")
        return False


async def broadcast_log(log_entry: dict):
    """Broadcast a log entry to all WebSocket subscribers"""
    if not log_subscribers:
        return
    message = json.dumps(log_entry)
    disconnected = set()
    for ws in set(log_subscribers):
        try:
            await ws.send_text(message)
        except Exception:
            disconnected.add(ws)
    log_subscribers.difference_update(disconnected)


class BufferHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "message": msg,
            }
            log_buffer.append(log_entry)
            # Queue broadcast to WebSocket subscribers
            if log_subscribers:
                try:
                    asyncio.get_event_loop().create_task(broadcast_log(log_entry))
                except RuntimeError:
                    pass  # No event loop available (startup phase)
        except Exception:
            self.handleError(record)


# Capture Stdout/Stderr for Shell Output (tqdm, print, etc)
class StreamToBuffer:
    """Proxy stream that mirrors output and stores human-readable lines in log_buffer."""

    def __init__(self, original_stream, level="INFO"):
        self.original_stream = original_stream
        self.level = level

    def write(self, message):
        try:
            # Write to original stream first
            self.original_stream.write(message)
            self.original_stream.flush()

            # Filter out empty newlines or carriage returns that are just moving cursor
            if message and message.strip():
                # Clean up tqdm's carriage returns for the web view
                clean_msg = message.replace("\r", "").strip()
                if clean_msg:
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "level": self.level,
                        "message": clean_msg,
                    }
                    log_buffer.append(log_entry)
                    # Broadcast to WebSocket subscribers
                    if log_subscribers:
                        try:
                            asyncio.get_event_loop().create_task(
                                broadcast_log(log_entry)
                            )
                        except RuntimeError as exc:
                            # 🐛 Debug: ignore missing event loop when broadcasting logs
                            logging.debug(
                                "🐛 Failed to schedule log broadcast task: %s", exc
                            )
        except Exception:
            # If writing to buffer fails, don't crash the app
            pass

    def flush(self):
        try:
            self.original_stream.flush()
        except Exception:
            # Ignore flush errors to avoid crashing on non-critical I/O issues
            pass

    # Proxy common file-like attributes used by uvicorn/print/tqdm
    def isatty(self):
        return getattr(self.original_stream, "isatty", lambda: False)()

    @property
    def encoding(self):
        return getattr(self.original_stream, "encoding", "utf-8")

    def fileno(self):
        return getattr(self.original_stream, "fileno", lambda: -1)()

    def writelines(self, lines):
        for line in lines:
            self.write(line)


# Redirect sys.stdout and sys.stderr
sys.stdout = StreamToBuffer(sys.stdout, "INFO")
sys.stderr = StreamToBuffer(
    sys.stderr, "SHELL"
)  # Use SHELL level for stderr (tqdm usually goes here)

# Add buffer handler to root logger only (module loggers propagate by default)
buffer_handler = BufferHandler()
buffer_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)
logging.getLogger().addHandler(buffer_handler)

# Attach to common libraries used by the generators so their INFO logs appear
for noisy_logger in ["diffusers", "transformers", "accelerate"]:
    logging.getLogger(noisy_logger).setLevel(logging.INFO)
    logging.getLogger(noisy_logger).addHandler(buffer_handler)

# Create FastAPI app
app = FastAPI(
    title="Oelala AI Video Generator",
    description="AI-powered video generation from images using Wan2.2",
    version="1.0.0",
)

# CRITICAL: Add CORS middleware FIRST, before any mounts or routes
# NOTE: allow_origins=["*"] + allow_credentials=True is INVALID per CORS spec.
# Starlette returns 'Access-Control-Allow-Origin: *' on non-preflight requests,
# which browsers reject when credentials/Authorization headers are sent.
# Must list specific origins for credentialed requests to work.
# ALLOWED_ORIGINS is imported from storage_utils (single source of truth).
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


# Cache control middleware — sets appropriate headers based on path
@app.middleware("http")
async def cache_control_middleware(request, call_next):
    response = await call_next(request)
    path = request.url.path

    # Skip if Cache-Control already set (e.g. by endpoint)
    if "cache-control" not in response.headers:
        if path.startswith("/avatars/"):
            # Avatars can change — cache 1 day
            response.headers["Cache-Control"] = "public, max-age=86400"
        elif path.startswith("/comfyui-output/"):
            # Immutable generated content
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif path.startswith("/static/assets/"):
            # Vite hashed assets — immutable
            response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        elif path.startswith("/static/"):
            # Non-hashed static (index.html etc) — revalidate
            response.headers["Cache-Control"] = "no-cache"
        elif path.startswith("/api/loras") and request.method == "GET":
            # LoRA list changes rarely
            response.headers["Cache-Control"] = (
                "public, s-maxage=300, stale-while-revalidate=600"
            )
        elif path.startswith("/api/gallery") and request.method == "GET":
            # Gallery listings — short CDN cache
            response.headers["Cache-Control"] = (
                "public, s-maxage=60, stale-while-revalidate=300"
            )
        elif request.method in ("POST", "PUT", "DELETE", "PATCH"):
            response.headers["Cache-Control"] = "private, no-store"

    # Ensure Vary header for compressed content
    if "Vary" not in response.headers:
        response.headers["Vary"] = "Accept-Encoding"

    return response


# ── Request timing & metrics middleware ──────────────────────────────
# Tracks request counts, latencies, and error rates per endpoint.
# Stored in-memory; exposed via /api/admin/metrics.
from collections import Counter

_request_metrics = {
    "total_requests": 0,
    "total_errors": 0,  # 5xx responses
    "status_counts": Counter(),  # status code → count
    "endpoint_latencies": {},  # path → list of recent ms values (capped)
    "started_at": datetime.now().isoformat(),
}
_LATENCY_CAP = 200  # keep last N measurements per path


@app.middleware("http")
async def request_metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        _request_metrics["total_requests"] += 1
        _request_metrics["total_errors"] += 1
        _request_metrics["status_counts"]["500"] += 1
        if _sentry_dsn:
            sentry_sdk.capture_exception(exc)
        raise
    elapsed_ms = round((time.perf_counter() - start) * 1000, 1)

    status = response.status_code
    _request_metrics["total_requests"] += 1
    _request_metrics["status_counts"][str(status)] += 1
    if status >= 500:
        _request_metrics["total_errors"] += 1

    # Track latency for API endpoints only (skip static files)
    path = request.url.path
    if path.startswith("/api/"):
        bucket = _request_metrics["endpoint_latencies"]
        if path not in bucket:
            bucket[path] = deque(maxlen=_LATENCY_CAP)
        bucket[path].append(elapsed_ms)

    # Header for debugging
    response.headers["X-Response-Time"] = f"{elapsed_ms}ms"
    return response


# API v1 router (programmatic access)
from api_v1 import router as api_v1_router

# API keys management router
from api_keys_management import router as api_keys_router

# Include API routers
app.include_router(api_v1_router)  # REST API v1 at /api/v1/*
app.include_router(api_keys_router)  # API key management at /api/keys/*
app.include_router(credits_router)
app.include_router(stripe_router)  # Stripe webhook at /api/stripe/webhook
app.include_router(gallery_router)
app.include_router(profile_router)  # User profiles at /api/profile/*
app.include_router(admin_router)  # Admin panel at /api/admin/*
app.include_router(lora_router)  # LoRA browser at /api/loras/*
app.include_router(webhooks_router)  # Webhooks at /webhooks/*
app.include_router(moderation_public_router)  # Content reports at /api/report/*
app.include_router(
    moderation_admin_router
)  # Admin moderation at /api/admin/moderation/*

# Tool profiles (per-user settings persistence)
from tool_profiles_api import router as tool_profiles_router

app.include_router(tool_profiles_router)  # Tool settings at /api/settings/*

# V2 Unified Generation API
from src.backend.generation.v2_api import router as v2_generation_router

app.include_router(v2_generation_router)  # V2 generation at /v2/*

# Create directories
UPLOAD_DIR = Path("/tmp/oelala_uploads")
OUTPUT_DIR = Path("/tmp/oelala_generated")
FRONTEND_DIR = Path("/home/flip/oelala/src/frontend")
COMFYUI_OUTPUT_DIR = Path("/home/flip/oelala/ComfyUI/output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

SAFE_PATH_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._@()+,= -]{0,254}$")
SAFE_EXTERNAL_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
MAX_STORAGE_KEY_LENGTH = 1024
MAX_METADATA_IMAGE_BYTES = 20 * 1024 * 1024
BLOCKED_METADATA_HOSTS = {"localhost", "localhost.localdomain"}
ALLOWED_USER_MEDIA_TYPES = {"images", "videos", "audio", "uploads"}
ALLOWED_YOUTUBE_HOSTS = {
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtube-nocookie.com",
    "www.youtube-nocookie.com",
    "youtu.be",
}


def _normalize_storage_key(key: str, *, allow_subdirectories: bool = True) -> str:
    """Normalize a MinIO object key and reject traversal or unsafe path segments."""
    raw_key = str(key or "")
    if (
        not raw_key
        or len(raw_key) > MAX_STORAGE_KEY_LENGTH
        or "\x00" in raw_key
        or "\\" in raw_key
        or raw_key.startswith("/")
    ):
        raise HTTPException(status_code=400, detail="Invalid file path")

    normalized = posixpath.normpath(raw_key)
    if normalized in ("", ".", "..") or normalized.startswith("../"):
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not allow_subdirectories and "/" in normalized:
        raise HTTPException(status_code=400, detail="Invalid filename")

    segments = normalized.split("/")
    if any(segment in ("", ".", "..") for segment in segments):
        raise HTTPException(status_code=400, detail="Invalid file path")
    if any(not SAFE_PATH_SEGMENT_RE.fullmatch(segment) for segment in segments):
        raise HTTPException(status_code=400, detail="Invalid file path")

    return normalized


def _safe_filename(filename: str, *, default: str = "file") -> str:
    """Return a validated basename for user-facing file routes."""
    raw_name = str(filename or default).replace("\\", "/")
    basename = posixpath.basename(raw_name)
    return _normalize_storage_key(basename, allow_subdirectories=False)


def _safe_child_path(base_dir: Path, filename: str) -> Path:
    """Resolve a validated filename under a fixed directory."""
    safe_name = _safe_filename(filename)
    base = base_dir.resolve()
    candidate = (base / safe_name).resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid file path") from exc
    return candidate


def _find_existing_media_path(media_ref: str) -> Optional[str]:
    """Resolve an existing media reference under known local media roots."""
    ref = str(media_ref or "").strip()
    if not ref:
        return None

    parsed = urlparse(ref)
    ref_path = unquote(parsed.path if parsed.scheme in {"http", "https"} else ref)
    if ref_path.startswith(("/media/", "/generated/", "/uploads/", "/ComfyUI/output/")):
        ref_path = ref_path.lstrip("/")
    workspace_root = Path("/home/flip/oelala").resolve()
    allowed_roots = (
        OUTPUT_DIR.resolve(),
        COMFYUI_OUTPUT_DIR.resolve(),
        UPLOAD_DIR.resolve(),
        (workspace_root / "media").resolve(),
        (workspace_root / "generated").resolve(),
    )

    if Path(ref_path).is_absolute():
        candidate = Path(ref_path).resolve()
        for root in allowed_roots:
            try:
                candidate.relative_to(root)
                return str(candidate) if candidate.exists() else None
            except ValueError:
                continue
        raise HTTPException(status_code=400, detail="Invalid media path")

    normalized = _normalize_storage_key(ref_path)
    for prefix in ("media/", "generated/", "uploads/", "ComfyUI/output/"):
        if normalized.startswith(prefix):
            candidate = (workspace_root / normalized).resolve()
            for root in allowed_roots:
                try:
                    candidate.relative_to(root)
                    return str(candidate) if candidate.exists() else None
                except ValueError:
                    continue
            raise HTTPException(status_code=400, detail="Invalid media path")

    safe_filename = _safe_filename(normalized)
    for search_dir in (OUTPUT_DIR, COMFYUI_OUTPUT_DIR, UPLOAD_DIR):
        candidate = _safe_child_path(search_dir, safe_filename)
        if candidate.exists():
            return str(candidate)
    return None


def _safe_user_media_type(media_type: str) -> str:
    """Validate a user media type path segment."""
    safe_type = _normalize_storage_key(media_type, allow_subdirectories=False)
    if safe_type not in ALLOWED_USER_MEDIA_TYPES:
        raise HTTPException(status_code=400, detail="Invalid media type")
    return safe_type


def _safe_external_id(value: str, label: str) -> str:
    """Validate an external service ID before using it as a URL path segment."""
    safe_value = str(value or "").strip()
    if not SAFE_EXTERNAL_ID_RE.fullmatch(safe_value):
        raise HTTPException(status_code=400, detail=f"Invalid {label}")
    return safe_value


def _validate_public_image_url(image_url: str) -> str:
    """Validate a user-provided image URL before server-side fetching."""
    url = str(image_url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(status_code=400, detail="Invalid image URL")
    if parsed.username or parsed.password:
        raise HTTPException(status_code=400, detail="Invalid image URL")

    hostname = parsed.hostname.rstrip(".").lower()
    if hostname in BLOCKED_METADATA_HOSTS or hostname.endswith(".local"):
        raise HTTPException(status_code=400, detail="Image URL host is not allowed")

    try:
        addresses = socket.getaddrinfo(
            hostname,
            parsed.port or (443 if parsed.scheme == "https" else 80),
            type=socket.SOCK_STREAM,
        )
    except socket.gaierror as exc:
        raise HTTPException(
            status_code=400, detail="Image URL host is invalid"
        ) from exc

    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            raise HTTPException(status_code=400, detail="Image URL host is not allowed")

    return url


def _validate_youtube_url(video_url: str) -> str:
    """Validate a YouTube URL before passing it to yt-dlp."""
    url = str(video_url or "").strip()
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    if parsed.username or parsed.password:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    hostname = parsed.hostname.rstrip(".").lower()
    if hostname not in ALLOWED_YOUTUBE_HOSTS:
        raise HTTPException(status_code=400, detail="Only YouTube URLs are supported")
    return url


def _js_string_literal(value: str) -> str:
    """Render a safe JavaScript string literal for server-generated HTML."""
    return json.dumps(value)


async def _save_upload(file: UploadFile, dest: Path) -> bytes:
    """Save an UploadFile to disk, returning the raw bytes.

    Seeks to 0 first (guards against consumed streams) and rejects
    empty uploads with a 400 so RunPod / ComfyUI never receives 0-byte files.
    """
    await file.seek(0)
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty (0 bytes). Re-select or re-upload the file.",
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    logger.info(f"📤 Saved upload: {dest} ({len(content)} bytes)")
    return content


# Mount static files after CORS
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# NOTE: /comfyui-output and /avatars StaticFiles mounts removed.
# These are now served via MinIO-backed storage proxy endpoints:
#   /comfyui/output/{filename}  → storage bucket "comfyui-local"
#   /avatars/{filename}         → storage bucket "avatars" (endpoint below)


@app.get("/avatars/{filename}")
async def get_avatar(filename: str, request: Request):
    """Serve avatar images via MinIO storage proxy."""
    return _storage_proxy_response(
        "avatars",
        filename,
        request,
        cache_control="public, max-age=86400, must-revalidate",
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _storage_proxy_response(
    bucket: str,
    key: str,
    request: Request,
    cache_control: str = "public, max-age=3600, must-revalidate",
) -> Response:
    """
    Fetch a file from MinIO storage and return it as a FastAPI Response.

    Supports HTTP Range requests (partial content) for video seeking,
    ETag / Last-Modified for caching, and CORS headers compatible with
    Cloudflare caching.
    """
    key = _normalize_storage_key(key)
    storage = get_storage_client()

    range_header = request.headers.get("range")

    # For Range requests we need metadata first; for full responses we can
    # skip the stat and use get_with_metadata directly (single round-trip).
    if range_header and range_header.startswith("bytes="):
        try:
            meta = storage.stat(bucket, key)
            if meta is None:
                raise HTTPException(status_code=404, detail="File not found")
        except S3Error as exc:
            if exc.code in ("NoSuchKey", "NoSuchBucket"):
                raise HTTPException(status_code=404, detail="File not found")
            logger.error(f"Storage S3 error during stat({bucket}, {key}): {exc}")
            raise HTTPException(status_code=502, detail="Storage service error")
        except Exception as exc:
            logger.error(f"Storage error during stat({bucket}, {key}): {exc}")
            raise HTTPException(status_code=503, detail="Storage service unavailable")

        total_size = meta["size"]
        content_type = meta["content_type"]
        etag = meta.get("etag")
        last_modified = meta.get("last_modified")

        headers = {
            "Cache-Control": cache_control,
            "Accept-Ranges": "bytes",
            "Vary": "Origin",
        }
        if etag:
            headers["ETag"] = f'"{etag}"'
        if last_modified:
            headers["Last-Modified"] = format_last_modified(last_modified)

        origin = request.headers.get("origin")
        if origin and origin in ALLOWED_ORIGINS:
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Credentials"] = "true"

        try:
            parsed = parse_range_header(range_header, total_size)
            if parsed is None:
                pass  # malformed — fall through to full response below
            else:
                range_start, range_end = parsed
                content_length = range_end - range_start + 1
                content = storage.get_object_range(
                    bucket, key, offset=range_start, length=content_length
                )
                headers["Content-Range"] = (
                    f"bytes {range_start}-{range_end}/{total_size}"
                )
                headers["Content-Length"] = str(content_length)
                return Response(
                    content=content,
                    status_code=206,
                    media_type=content_type,
                    headers=headers,
                )
        except ValueError:
            # Unsatisfiable range → 416
            headers["Content-Range"] = f"bytes */{total_size}"
            return Response(
                status_code=416,
                headers=headers,
                media_type=content_type,
            )

    # Full response (no Range, malformed Range, or fallback)
    try:
        content, content_type, total_size, etag, last_modified = (
            storage.get_with_metadata(bucket, key)
        )
    except S3Error as exc:
        if exc.code in ("NoSuchKey", "NoSuchBucket"):
            raise HTTPException(status_code=404, detail="File not found")
        logger.error(f"Storage S3 error during get({bucket}, {key}): {exc}")
        raise HTTPException(status_code=502, detail="Storage service error")
    except Exception as exc:
        logger.error(f"Storage error during get({bucket}, {key}): {exc}")
        raise HTTPException(status_code=503, detail="Storage service unavailable")

    headers = {
        "Cache-Control": cache_control,
        "Accept-Ranges": "bytes",
        "Vary": "Origin",
        "Content-Length": str(total_size),
    }
    if etag:
        headers["ETag"] = etag
    if last_modified:
        headers["Last-Modified"] = last_modified

    origin = request.headers.get("origin")
    if origin and origin in ALLOWED_ORIGINS:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"

    return Response(content=content, media_type=content_type, headers=headers)


# =============================================================================
# Share Link Endpoint  (must be declared before static mounts take effect)
# =============================================================================

SITE_URL = os.getenv("SITE_URL", "http://localhost:5174")


@app.get("/share/{media_id}", response_class=HTMLResponse, include_in_schema=False)
async def share_media(media_id: str, request: Request):
    """
    Social-share page for a published gallery item.
    Returns HTML with Open Graph / Twitter Card meta tags and a JS redirect
    to the main SPA with ?openItem={media_id} so the gallery auto-opens it.
    """
    from gallery_api import get_supabase_client  # local import to avoid circular

    supabase = get_supabase_client()
    title = "Oelala – AI-generated media"
    description = "Check out this AI-generated creation on Oelala."
    og_image = f"{SITE_URL}/og-default.jpg"
    media_type_str = "video"

    if supabase:
        try:
            result = (
                supabase.table("published_media")
                .select("id,title,description,thumbnail_url,media_type,is_nsfw")
                .eq("id", media_id)
                .single()
                .execute()
            )
            if result.data:
                item = result.data
                if item.get("is_nsfw"):
                    description = (
                        "⚠️ This content is marked NSFW and requires login to view."
                    )
                else:
                    title = item.get("title") or title
                    description = item.get("description") or description
                media_type_str = item.get("media_type", "video")
                thumb = item.get("thumbnail_url")
                if thumb:
                    og_image = (
                        thumb if thumb.startswith("http") else f"{SITE_URL}{thumb}"
                    )
                elif media_type_str == "image":
                    og_image = f"{SITE_URL}/api/gallery/{media_id}/file"
        except Exception as exc:
            logger.warning(f"⚠️ Share page: failed to fetch media {media_id}: {exc}")

    safe_media_id = quote(media_id, safe="")
    share_url = f"{SITE_URL}/share/{safe_media_id}"
    redirect_url = f"{SITE_URL}/?openItem={safe_media_id}"
    og_type = "video.other" if media_type_str == "video" else "website"
    title_html = html_lib.escape(title, quote=True)
    description_html = html_lib.escape(description, quote=True)
    og_image_html = html_lib.escape(og_image, quote=True)
    share_url_html = html_lib.escape(share_url, quote=True)
    redirect_url_html = html_lib.escape(redirect_url, quote=True)
    og_type_html = html_lib.escape(og_type, quote=True)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{title_html}</title>

  <!-- Open Graph -->
    <meta property="og:title" content="{title_html}">
    <meta property="og:description" content="{description_html}">
    <meta property="og:image" content="{og_image_html}">
    <meta property="og:url" content="{share_url_html}">
    <meta property="og:type" content="{og_type_html}">
  <meta property="og:site_name" content="Oelala">

  <!-- Twitter Card -->
  <meta name="twitter:card" content="summary_large_image">
    <meta name="twitter:title" content="{title_html}">
    <meta name="twitter:description" content="{description_html}">
    <meta name="twitter:image" content="{og_image_html}">

  <!-- Redirect humans to the SPA -->
    <meta http-equiv="refresh" content="0;url={redirect_url_html}">
    <link rel="canonical" href="{share_url_html}">
</head>
<body>
    <p>Redirecting… <a href="{redirect_url_html}">Click here if not redirected.</a></p>
</body>
</html>"""
    return HTMLResponse(content=html)


def create_progress_callback(prompt_id: str):
    """
    Create a progress callback function for broadcasting WebSocket updates.

    Args:
        prompt_id: The job/prompt ID to broadcast progress for

    Returns:
        Async callback function that broadcasts progress updates
    """

    async def progress_callback(progress: int, node_name: str):
        """Broadcast progress to user's WebSocket connections"""
        if ws_manager:
            await ws_manager.broadcast_progress(
                job_id=prompt_id,
                progress=progress,
                node_name=node_name,
                message=f"Processing {node_name}",
            )

    return progress_callback


async def upload_generated_media(
    user_id: str,
    file_path: Path,
    generation_type: str,
    prompt: str,
    workflow_id: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
) -> Optional[MediaRecord]:
    """
    Upload a generated media file to MinIO and sync metadata to Supabase.

    Args:
        user_id: User ID who owns the media
        file_path: Path to the generated file
        generation_type: Type of generation (t2v, i2v, t2i, etc.)
        prompt: The prompt used for generation
        workflow_id: Optional ComfyUI workflow ID
        extra_metadata: Additional metadata (resolution, fps, model_name, etc.)

    Returns:
        MediaRecord if successful, None if failed
    """
    try:
        media_service = get_media_service()

        # Read file data
        file_data = file_path.read_bytes()

        # Merge extra metadata
        metadata = extra_metadata or {}
        metadata["prompt"] = prompt
        metadata["generation_type"] = generation_type
        metadata["size_bytes"] = len(file_data)

        # Upload to storage + sync metadata
        record = await media_service.upload(
            user_id=user_id,
            file_data=file_data,
            filename=file_path.name,
            generation_type=generation_type,
            prompt=prompt,
            workflow_id=workflow_id,
            extra_metadata=metadata,
        )

        logger.info(f"🗄️ Media uploaded to storage: {record.storage_path}")

        # Clean up local file after successful upload to keep media dirs empty
        try:
            if file_path.exists():
                file_path.unlink()
                logger.info(f"🗑️ Cleaned up local file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up local file {file_path}: {e}")

        return record

    except Exception as e:
        logger.error(f"❌ Failed to upload media to storage: {e}")
        return None


def get_signed_media_url(storage_path: str, expires_in: int = 3600) -> str:
    """
    Generate a signed URL for a media file.

    Args:
        storage_path: Full storage path (e.g., users/{user_id}/videos/file.mp4)
        expires_in: URL expiration time in seconds (default 1 hour)

    Returns:
        Signed URL for public access
    """
    media_service = get_media_service()
    return media_service.generate_signed_url(storage_path, expires_in)


# =============================================================================
# API Endpoints
# =============================================================================


@app.get("/logs")
async def get_logs():
    """Get recent server logs"""
    return list(log_buffer)


@app.websocket("/ws/logs")
async def websocket_logs(websocket: WebSocket):
    """WebSocket endpoint for streaming server logs"""
    await websocket.accept()
    log_subscribers.add(websocket)
    logger.info(f"📡 Log WebSocket connected (total: {len(log_subscribers)})")
    try:
        # Send recent logs on connect
        for log_entry in list(log_buffer)[-50:]:  # Last 50 logs
            await websocket.send_text(json.dumps(log_entry))
        # Keep connection alive
        while True:
            try:
                # Wait for pings/close from client
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
    finally:
        log_subscribers.discard(websocket)
        logger.info(
            f"📡 Log WebSocket disconnected (remaining: {len(log_subscribers)})"
        )


@app.websocket("/ws/progress")
async def websocket_progress(websocket: WebSocket):
    """
    WebSocket endpoint for real-time job progress updates.
    Requires JWT authentication via initial message.
    """
    await websocket.accept()

    if not ws_manager:
        await websocket.close(code=1011, reason="WebSocket progress not available")
        return

    user_id = None

    try:
        # Wait for authentication message
        auth_message = await asyncio.wait_for(
            websocket.receive_text(), timeout=WEBSOCKET_AUTH_TIMEOUT
        )
        auth_data = json.loads(auth_message)

        if auth_data.get("type") != "auth":
            await websocket.close(code=1008, reason="Expected auth message")
            return

        # Verify JWT token
        token = auth_data.get("token")
        if not token:
            await websocket.close(code=1008, reason="Missing token")
            return

        # Try to decode JWT (imported at top of file)
        payload = decode_jwt_with_secret(token)
        if not payload:
            payload = decode_jwt_with_jwks(token)

        if not payload:
            await websocket.close(code=1008, reason="Invalid token")
            return

        user_id = payload.get("sub")
        if not user_id:
            await websocket.close(code=1008, reason="Missing user_id in token")
            return

        # Authentication successful
        await ws_manager.connect(websocket, user_id=user_id)
        await websocket.send_json({"type": "auth_success"})
        logger.info(f"📡 Progress WebSocket authenticated for user {user_id}")

        # Keep connection alive and listen for close
        while True:
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break

    except asyncio.TimeoutError:
        await websocket.close(code=1008, reason="Authentication timeout")
    except json.JSONDecodeError:
        await websocket.close(code=1008, reason="Invalid JSON")
    except Exception as e:
        logger.error(f"WebSocket progress error: {e}")
        await websocket.close(code=1011, reason="Internal error")
    finally:
        if user_id and ws_manager:
            ws_manager.disconnect(websocket, user_id=user_id)
        logger.info(f"📡 Progress WebSocket disconnected for user {user_id}")


@app.get("/progress/{job_id}")
async def get_progress(job_id: str):
    data = progress_store.get(job_id, None)
    if not data:
        return {"job_id": job_id, "progress": 0, "status": "unknown"}
    return data


@app.get("/video-test.html")
async def serve_video_test():
    """Serve video test page"""
    test_page = FRONTEND_DIR / "video-test.html"
    if not test_page.exists():
        raise HTTPException(status_code=404, detail="Test page not found")
    return FileResponse(test_page, media_type="text/html")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # Check ComfyUI availability
    client = get_comfyui_client()
    if client and client.is_available():
        logger.info("✅ ComfyUI backend available and ready!")
    else:
        logger.warning("⚠️ ComfyUI backend not available - some features may not work")

    # Start WebSocket progress monitoring
    if ws_manager and job_queue_manager and progress_monitor:
        logger.info("🔄 Starting WebSocket progress monitoring...")
        # Start background queue polling
        await job_queue_manager.start_polling(
            ws_manager, interval=QUEUE_POLLING_INTERVAL
        )
        # Start ComfyUI progress monitor
        progress_monitor.start()
        logger.info("✅ WebSocket progress monitoring started!")
    else:
        logger.warning(
            "⚠️ WebSocket progress modules not available - real-time updates disabled"
        )

    # Start LLM queue worker
    if llm_queue_manager:
        logger.info("🔄 Starting LLM queue worker...")
        asyncio.create_task(llm_queue_manager.start_worker(_process_llm_job))
        logger.info("✅ LLM queue worker started!")
    else:
        logger.warning(
            "⚠️ LLM queue not available — prompt enhancement will use sync fallback"
        )

    # Start webhook retry worker
    logger.info("🪝 Starting webhook retry worker...")
    await webhook_service.start_retry_worker(interval=30.0)
    logger.info("✅ Webhook retry worker started!")

    # Backfill generation times from ComfyUI history
    try:
        from websocket_handler import backfill_generation_times_from_comfyui

        count = await backfill_generation_times_from_comfyui()
        if count:
            logger.info(f"⏱ Backfilled {count} generation times from ComfyUI history")
    except Exception as e:
        logger.warning(f"Generation time backfill failed: {e}")

    # Start background cloud job poller
    global _cloud_poller_task, _local_poller_task
    if _runpod:
        _cloud_poller_task = asyncio.create_task(_cloud_job_poller())
        logger.info("☁️ Background cloud job poller started")
    else:
        logger.info("☁️ Cloud job poller skipped (RunPod not available)")

    _local_poller_task = asyncio.create_task(_local_job_poller())
    logger.info("🖼️ Background local job poller started")
    # ── V2 Unified Generation Core ──────────────────────────────
    try:
        from src.backend.generation.factory import create_registry
        from src.backend.generation.router import GenerationRouter
        from src.backend.generation.v2_api import init_v2_api

        v2_registry = create_registry(
            comfyui_client_fn=get_comfyui_client,
            submit_to_runpod_fn=_submit_to_runpod if _runpod else None,
            face_service_mod=face_service,
            runpod_endpoint_wan22=os.getenv("RUNPOD_ENDPOINT_ID"),
            runpod_endpoint_ltx23=os.getenv("RUNPOD_LTX23_ENDPOINT_ID"),
            runpod_endpoint_i2i=os.getenv("RUNPOD_I2I_ENDPOINT_ID"),
        )

        # Async wrapper for ComfyUI b64 image upload (used by router)
        async def _comfyui_upload_b64(b64_data: str, filename: str) -> str:
            import base64 as _b64

            b64_data = "".join(b64_data.split())
            missing = (-len(b64_data)) % 4
            if missing:
                b64_data = b64_data + ("=" * missing)

            img_bytes = _b64.b64decode(b64_data)
            client = get_comfyui_client()
            result = client.upload_image_from_bytes(img_bytes, filename)
            if result is None:
                raise RuntimeError(f"ComfyUI image upload failed for {filename}")
            return result

        v2_gen_router = GenerationRouter(
            v2_registry,
            comfyui_upload_fn=_comfyui_upload_b64,
            track_local_job_fn=track_v2_local_job,
        )
        init_v2_api(
            registry=v2_registry,
            gen_router=v2_gen_router,
            get_current_user=get_current_user,
            check_credits=check_credits,
            deduct_credits=deduct_credits,
        )

        # Initialize V1 compat layer so dispatch_v1() endpoints work
        from src.backend.generation.v1_compat import init_v1_compat

        init_v1_compat(
            router=v2_gen_router,
            check_credits=check_credits,
            deduct_credits=deduct_credits,
            get_comfyui_client=get_comfyui_client,
        )
        logger.info(f"🚀 V2 Generation API ready ({len(v2_registry)} adapters)")
    except Exception as e:
        logger.error(f"❌ V2 Generation API failed to initialize: {e}", exc_info=True)


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown"""
    if job_queue_manager:
        logger.info("🛑 Stopping queue polling...")
        await job_queue_manager.stop_polling()

    if progress_monitor:
        logger.info("🛑 Stopping progress monitor...")
        progress_monitor.stop()

    # Stop webhook retry worker
    logger.info("🛑 Stopping webhook retry worker...")
    await webhook_service.stop_retry_worker()

    # Stop cloud job poller
    if _cloud_poller_task and not _cloud_poller_task.done():
        logger.info("🛑 Stopping cloud job poller...")
        _cloud_poller_task.cancel()
        try:
            await _cloud_poller_task
        except asyncio.CancelledError:
            pass
    if _local_poller_task and not _local_poller_task.done():
        logger.info("🛑 Stopping local job poller...")
        _local_poller_task.cancel()
        try:
            await _local_poller_task
        except asyncio.CancelledError:
            pass

    logger.info("✅ Shutdown complete")


@app.get("/")
async def root():
    """Root endpoint"""
    client = get_comfyui_client()
    comfyui_ok = client and client.is_available() if client else False
    return {
        "message": "Oelala AI Video Generator API",
        "version": "1.0.0",
        "status": "ready" if comfyui_ok else "error",
        "endpoints": {
            "POST /generate": "Generate video from image",
            "POST /generate-text": "Generate video from text prompt",
            "POST /generate-image": "Generate image from text prompt",
            "GET /files/{filename}": "Download generated video",
            "GET /health": "Health check",
        },
    }


@app.get("/list-comfyui-media")
async def list_comfyui_media(
    type: str = "all",
    grouped: bool = False,
    include_metadata: bool = False,
    hide_start_images: bool = True,
    user: User = Depends(get_current_user),
):
    """List media files from ComfyUI output directory (admin only)

    Args:
        type: Filter by media type ('all', 'video', 'image', 'audio')
        grouped: Group videos with source images (not implemented yet)
        include_metadata: Include PNG metadata in response
        hide_start_images: Hide images that are start frames for videos (default True)
    """
    # Admin-only: these are shared server directories
    if not await check_admin(user):
        raise HTTPException(status_code=403, detail="Admin access required")

    comfyui_output = Path("/home/flip/oelala/ComfyUI/output")

    if not comfyui_output.exists():
        return {"media": [], "stats": {"videos": 0, "images": 0, "audio": 0}}

    media = []
    video_count = 0
    image_count = 0
    audio_count = 0

    # First pass: collect all files and extract timestamps from videos
    video_timestamps = set()
    all_files = []

    for file_path in comfyui_output.iterdir():
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext in [".mp4", ".webm", ".mov", ".avi"]:
            # Extract timestamp from video filename (e.g., oelala_20260102_075057)
            import re

            match = re.search(r"(\d{8}_\d{6})", file_path.name)
            if match:
                video_timestamps.add(match.group(1))
            all_files.append((file_path, "video"))
        elif ext in [".png", ".jpg", ".jpeg", ".webp"]:
            all_files.append((file_path, "image"))
        elif ext in [".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac"]:
            all_files.append((file_path, "audio"))

    # Second pass: process files and mark start images
    for file_path, media_type in all_files:
        if media_type == "video":
            video_count += 1
        elif media_type == "image":
            image_count += 1
        elif media_type == "audio":
            audio_count += 1

        # Filter by type if requested
        if type != "all" and media_type != type:
            continue

        # Check if this image is a start image for a video
        is_start_image = False
        if media_type == "image" and hide_start_images:
            import re

            match = re.search(r"(\d{8}_\d{6})", file_path.name)
            if match and match.group(1) in video_timestamps:
                is_start_image = True
                # Skip this image if hiding start images
                continue

        stat = file_path.stat()
        item = {
            "filename": file_path.name,
            "type": media_type,
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "url": f"/comfyui-output/{file_path.name}",
            "is_start_image": is_start_image,
        }

        # Include metadata if requested (for images with embedded prompts)
        if include_metadata and media_type == "image":
            try:
                from PIL import Image

                img = Image.open(file_path)
                metadata = {"has_metadata": False}

                if hasattr(img, "info") and img.info:
                    # Try to extract prompt from ComfyUI workflow JSON
                    if "prompt" in img.info:
                        metadata["has_metadata"] = True
                        try:
                            workflow = json.loads(img.info["prompt"])
                            # Collect all text prompts for later analysis
                            all_texts = []
                            loras_found = []

                            # Extract prompts from various node types
                            for node_id, node in workflow.items():
                                if isinstance(node, dict):
                                    inputs = node.get("inputs", {})
                                    class_type = node.get("class_type", "")

                                    # Wan2.2 / standard positive_prompt
                                    if "positive_prompt" in inputs and isinstance(
                                        inputs["positive_prompt"], str
                                    ):
                                        text = inputs["positive_prompt"].strip()
                                        if len(text) > 20:
                                            metadata["positive_prompt"] = text
                                    # Negative prompt
                                    if "negative_prompt" in inputs and isinstance(
                                        inputs["negative_prompt"], str
                                    ):
                                        text = inputs["negative_prompt"].strip()
                                        if text:
                                            metadata["negative_prompt"] = text

                                    # CLIPTextEncode text - collect all for analysis
                                    if "text" in inputs and isinstance(
                                        inputs["text"], str
                                    ):
                                        text = inputs["text"].strip()
                                        if len(text) > 10:
                                            all_texts.append(
                                                {
                                                    "text": text,
                                                    "class_type": class_type,
                                                    "node_id": node_id,
                                                }
                                            )

                                    # Extract generation params
                                    if "steps" in inputs and isinstance(
                                        inputs["steps"], (int, float)
                                    ):
                                        metadata["steps"] = int(inputs["steps"])
                                    if "cfg" in inputs and isinstance(
                                        inputs["cfg"], (int, float)
                                    ):
                                        metadata["cfg"] = float(inputs["cfg"])
                                    if "seed" in inputs and isinstance(
                                        inputs["seed"], (int, float)
                                    ):
                                        metadata["seed"] = int(inputs["seed"])

                                    # Extract sampler info
                                    if "sampler_name" in inputs and isinstance(
                                        inputs["sampler_name"], str
                                    ):
                                        metadata["sampler"] = inputs["sampler_name"]
                                    if "scheduler" in inputs and isinstance(
                                        inputs["scheduler"], str
                                    ):
                                        metadata["scheduler"] = inputs["scheduler"]

                                    # Extract resolution from EmptyLatentImage or similar
                                    if "width" in inputs and "height" in inputs:
                                        w = inputs.get("width")
                                        h = inputs.get("height")
                                        if isinstance(w, (int, float)) and isinstance(
                                            h, (int, float)
                                        ):
                                            metadata["width"] = int(w)
                                            metadata["height"] = int(h)

                                    # Extract LoRA info
                                    if (
                                        "LoraLoader" in class_type
                                        or "lora" in class_type.lower()
                                    ):
                                        lora_name = inputs.get("lora_name", "")
                                        lora_strength = inputs.get(
                                            "strength_model",
                                            inputs.get("strength", 1.0),
                                        )
                                        if lora_name:
                                            loras_found.append(
                                                {
                                                    "name": lora_name,
                                                    "strength": float(lora_strength)
                                                    if isinstance(
                                                        lora_strength, (int, float)
                                                    )
                                                    else 1.0,
                                                }
                                            )

                                    # Wan2.2 specific LoRA loader
                                    if (
                                        "WanVideoLoraSelect" in class_type
                                        or "lora_high" in inputs
                                        or "lora_low" in inputs
                                    ):
                                        for key in [
                                            "lora_high",
                                            "lora_low",
                                            "lora_name",
                                        ]:
                                            if key in inputs and inputs[key]:
                                                lora_name = inputs[key]
                                                if isinstance(
                                                    lora_name, str
                                                ) and lora_name not in [
                                                    "None",
                                                    "none",
                                                    "",
                                                ]:
                                                    strength = inputs.get(
                                                        "strength",
                                                        inputs.get(
                                                            "lora_strength", 1.0
                                                        ),
                                                    )
                                                    loras_found.append(
                                                        {
                                                            "name": lora_name,
                                                            "strength": float(strength)
                                                            if isinstance(
                                                                strength, (int, float)
                                                            )
                                                            else 1.0,
                                                        }
                                                    )

                                    # Extract model/checkpoint info
                                    if "ckpt_name" in inputs and isinstance(
                                        inputs["ckpt_name"], str
                                    ):
                                        metadata["model"] = inputs["ckpt_name"]
                                    if "unet_name" in inputs and isinstance(
                                        inputs["unet_name"], str
                                    ):
                                        if not metadata.get("model"):
                                            metadata["model"] = inputs["unet_name"]

                            # Store unique LoRAs
                            if loras_found:
                                # Deduplicate by name
                                seen = set()
                                unique_loras = []
                                for lora in loras_found:
                                    if lora["name"] not in seen:
                                        seen.add(lora["name"])
                                        unique_loras.append(lora)
                                metadata["loras"] = unique_loras

                            # If no positive_prompt found, analyze CLIPTextEncode texts
                            if not metadata.get("positive_prompt") and all_texts:
                                # Heuristics: negative prompts often contain these keywords
                                negative_indicators = [
                                    "worst",
                                    "bad",
                                    "ugly",
                                    "blurry",
                                    "low quality",
                                    "低质量",
                                    "最差",
                                    "deformed",
                                ]

                                for item_text in all_texts:
                                    text = item_text["text"]
                                    text_lower = text.lower()

                                    # Check if it looks like a negative prompt
                                    is_negative = any(
                                        ind in text_lower for ind in negative_indicators
                                    )

                                    if is_negative and not metadata.get(
                                        "negative_prompt"
                                    ):
                                        metadata["negative_prompt"] = text
                                    elif not is_negative and not metadata.get(
                                        "positive_prompt"
                                    ):
                                        metadata["positive_prompt"] = text

                                # Fallback: if still no positive, use first text
                                if not metadata.get("positive_prompt") and all_texts:
                                    metadata["positive_prompt"] = all_texts[0]["text"]

                        except json.JSONDecodeError:
                            # Ignore malformed JSON in PNG 'prompt' metadata
                            pass

                    # Oelala params format
                    if "oelala_params" in img.info:
                        metadata["has_metadata"] = True
                        try:
                            params = json.loads(img.info["oelala_params"])
                            if params.get("prompt"):
                                metadata["positive_prompt"] = params["prompt"]
                            if params.get("negative_prompt"):
                                metadata["negative_prompt"] = params["negative_prompt"]
                            if params.get("steps"):
                                metadata["steps"] = params["steps"]
                            if params.get("cfg"):
                                metadata["cfg"] = params["cfg"]
                            if params.get("seed"):
                                metadata["seed"] = params["seed"]
                        except json.JSONDecodeError:
                            logging.getLogger(__name__).debug(
                                "🐛 Failed to decode oelala_params JSON from image metadata",
                                exc_info=True,
                            )

                img.close()
                item["metadata"] = metadata
            except Exception as e:
                logger.debug("Metadata extraction failed for %s: %s", file_path.name, e)
                item["metadata"] = {
                    "has_metadata": False,
                    "error": "metadata unavailable",
                }

        # For videos, try to find associated PNG with same timestamp or base name
        if include_metadata and media_type == "video":
            import re

            metadata_found = False

            def extract_metadata_from_png(png_path):
                """Extract full metadata from PNG file including LoRAs, sampler, model etc."""
                from PIL import Image

                img = Image.open(png_path)
                metadata = {"has_metadata": False}

                if hasattr(img, "info") and "prompt" in img.info:
                    metadata["has_metadata"] = True
                    try:
                        workflow = json.loads(img.info["prompt"])
                        all_texts = []
                        loras_found = []

                        for node_id, node in workflow.items():
                            if isinstance(node, dict):
                                inputs = node.get("inputs", {})
                                class_type = node.get("class_type", "")

                                # Wan2.2 / standard positive_prompt
                                if "positive_prompt" in inputs and isinstance(
                                    inputs["positive_prompt"], str
                                ):
                                    text = inputs["positive_prompt"].strip()
                                    if len(text) > 20:
                                        metadata["positive_prompt"] = text
                                if "negative_prompt" in inputs and isinstance(
                                    inputs["negative_prompt"], str
                                ):
                                    text = inputs["negative_prompt"].strip()
                                    if text:
                                        metadata["negative_prompt"] = text

                                # CLIPTextEncode text
                                if "text" in inputs and isinstance(inputs["text"], str):
                                    text = inputs["text"].strip()
                                    if len(text) > 10:
                                        all_texts.append(
                                            {"text": text, "class_type": class_type}
                                        )

                                # Generation params
                                if "steps" in inputs and isinstance(
                                    inputs["steps"], (int, float)
                                ):
                                    metadata["steps"] = int(inputs["steps"])
                                if "cfg" in inputs and isinstance(
                                    inputs["cfg"], (int, float)
                                ):
                                    metadata["cfg"] = float(inputs["cfg"])
                                if "seed" in inputs and isinstance(
                                    inputs["seed"], (int, float)
                                ):
                                    metadata["seed"] = int(inputs["seed"])

                                # Sampler info
                                if "sampler_name" in inputs and isinstance(
                                    inputs["sampler_name"], str
                                ):
                                    metadata["sampler"] = inputs["sampler_name"]
                                if "scheduler" in inputs and isinstance(
                                    inputs["scheduler"], str
                                ):
                                    metadata["scheduler"] = inputs["scheduler"]

                                # Resolution
                                if "width" in inputs and "height" in inputs:
                                    w, h = inputs.get("width"), inputs.get("height")
                                    if isinstance(w, (int, float)) and isinstance(
                                        h, (int, float)
                                    ):
                                        metadata["width"] = int(w)
                                        metadata["height"] = int(h)

                                # LoRA info
                                if (
                                    "LoraLoader" in class_type
                                    or "lora" in class_type.lower()
                                ):
                                    lora_name = inputs.get("lora_name", "")
                                    lora_strength = inputs.get(
                                        "strength_model", inputs.get("strength", 1.0)
                                    )
                                    if lora_name:
                                        loras_found.append(
                                            {
                                                "name": lora_name,
                                                "strength": float(lora_strength)
                                                if isinstance(
                                                    lora_strength, (int, float)
                                                )
                                                else 1.0,
                                            }
                                        )

                                # Wan2.2 LoRA loader
                                if (
                                    "WanVideoLoraSelect" in class_type
                                    or "lora_high" in inputs
                                    or "lora_low" in inputs
                                ):
                                    for key in ["lora_high", "lora_low", "lora_name"]:
                                        if key in inputs and inputs[key]:
                                            lora_name = inputs[key]
                                            if isinstance(
                                                lora_name, str
                                            ) and lora_name not in ["None", "none", ""]:
                                                strength = inputs.get(
                                                    "strength",
                                                    inputs.get("lora_strength", 1.0),
                                                )
                                                loras_found.append(
                                                    {
                                                        "name": lora_name,
                                                        "strength": float(strength)
                                                        if isinstance(
                                                            strength, (int, float)
                                                        )
                                                        else 1.0,
                                                    }
                                                )

                                # Model/checkpoint info
                                if "ckpt_name" in inputs and isinstance(
                                    inputs["ckpt_name"], str
                                ):
                                    metadata["model"] = inputs["ckpt_name"]
                                if "unet_name" in inputs and isinstance(
                                    inputs["unet_name"], str
                                ):
                                    if not metadata.get("model"):
                                        metadata["model"] = inputs["unet_name"]

                        # Store unique LoRAs
                        if loras_found:
                            seen = set()
                            unique_loras = []
                            for lora in loras_found:
                                if lora["name"] not in seen:
                                    seen.add(lora["name"])
                                    unique_loras.append(lora)
                            metadata["loras"] = unique_loras

                        # Analyze CLIPTextEncode texts if no positive_prompt found
                        if not metadata.get("positive_prompt") and all_texts:
                            negative_indicators = [
                                "worst",
                                "bad",
                                "ugly",
                                "blurry",
                                "low quality",
                                "低质量",
                                "最差",
                                "deformed",
                            ]
                            for item in all_texts:
                                text = item["text"]
                                text_lower = text.lower()
                                is_negative = any(
                                    ind in text_lower for ind in negative_indicators
                                )
                                if is_negative and not metadata.get("negative_prompt"):
                                    metadata["negative_prompt"] = text
                                elif not is_negative and not metadata.get(
                                    "positive_prompt"
                                ):
                                    metadata["positive_prompt"] = text
                            if not metadata.get("positive_prompt") and all_texts:
                                metadata["positive_prompt"] = all_texts[0]["text"]

                    except json.JSONDecodeError as exc:
                        logging.debug(
                            "🐛 Failed to decode JSON metadata from PNG prompt in %s: %s",
                            png_path,
                            exc,
                        )

                img.close()
                return metadata

            # Method 1: Look for PNG with same timestamp
            match = re.search(r"(\d{8}_\d{6})", file_path.name)
            if match:
                timestamp = match.group(1)
                for png_file in comfyui_output.glob(f"*{timestamp}*.png"):
                    try:
                        metadata = extract_metadata_from_png(png_file)
                        item["metadata"] = metadata
                        metadata_found = True
                        break  # Use first matching PNG
                    except Exception:
                        # Ignore errors reading individual PNG metadata
                        pass

            # Method 2: Look for PNG with same base name (video.mp4 -> video.png)
            if not metadata_found:
                base_name = file_path.stem  # filename without extension
                png_candidates = [
                    comfyui_output / f"{base_name}.png",
                    comfyui_output / f"{base_name}_00001.png",  # ComfyUI pattern
                ]
                for png_file in png_candidates:
                    if png_file.exists():
                        try:
                            metadata = extract_metadata_from_png(png_file)
                            item["metadata"] = metadata
                            metadata_found = True
                            break
                        except Exception:
                            # Ignore errors reading PNG metadata from start image
                            pass

            if not metadata_found:
                item["metadata"] = {"has_metadata": False}

        media.append(item)

    # Sort by modified time descending
    media.sort(key=lambda x: x["modified"], reverse=True)

    return {
        "media": media,
        "videos": video_count,
        "images": image_count,
        "audio": audio_count,
        "stats": {"videos": video_count, "images": image_count, "audio": audio_count},
    }


from pydantic import BaseModel


class DeleteMediaRequest(BaseModel):
    filenames: List[str]


@app.delete("/delete-comfyui-media")
async def delete_comfyui_media(request: DeleteMediaRequest):
    """Delete media files via MinIO with local fallback."""
    storage = get_storage_client()
    deleted = []
    errors = []

    logger.info(f"🗑️ Delete request for {len(request.filenames)} files")

    for filename in request.filenames:
        safe_filename = _safe_filename(filename)
        found = False

        # Try storage buckets first
        for bucket in ("comfyui-local", "generated"):
            try:
                if storage.delete(bucket, safe_filename):
                    deleted.append(safe_filename)
                    found = True
                    logger.info(f"   ✅ Deleted from storage/{bucket}: {safe_filename}")
                    break
            except Exception:
                continue

        # Fallback: try local ComfyUI output dir (ComfyUI may write here directly)
        if not found:
            comfyui_output = Path("/home/flip/oelala/ComfyUI/output")
            file_path = _safe_child_path(comfyui_output, safe_filename)
            if (
                str(file_path.resolve()).startswith(str(comfyui_output.resolve()))
                and file_path.exists()
            ):
                try:
                    file_path.unlink()
                    deleted.append(safe_filename)
                    found = True
                    logger.info(f"   ✅ Deleted from local ComfyUI: {safe_filename}")
                except Exception as e:
                    logger.warning(
                        f"Failed to delete local ComfyUI media {safe_filename}: {e}"
                    )
                    errors.append({"filename": safe_filename, "error": "delete failed"})
                    found = True

        if not found:
            errors.append({"filename": safe_filename, "error": "File not found"})
            logger.warning(f"   ⚠️ Not found: {safe_filename}")

    logger.info(f"🗑️ Delete complete: {len(deleted)} deleted, {len(errors)} errors")
    return {"deleted": deleted, "errors": errors, "count": len(deleted)}


@app.get("/loras")
async def list_loras():
    """
    List available LoRA models from ComfyUI/models/loras folder.
    Returns LoRAs grouped by noise type (high/low) for Wan2.2 dual-pass workflow.
    Includes NSFW detection based on filename patterns.
    """
    loras_dir = Path("/home/flip/oelala/ComfyUI/models/loras")
    ssd_loras_dir = Path("/mnt/ssd/loras")

    lora_dirs = [d for d in [loras_dir, ssd_loras_dir] if d.exists()]

    if not lora_dirs:
        return {
            "loras": [],
            "high_noise": [],
            "low_noise": [],
            "general": [],
            "by_category": {},
        }

    # NSFW keywords for detection
    NSFW_KEYWORDS = [
        "nsfw",
        "nude",
        "naked",
        "sex",
        "porn",
        "xxx",
        "adult",
        "erotic",
        "cumshot",
        "cum",
        "anal",
        "blowjob",
        "bj",
        "fuck",
        "cock",
        "dick",
        "pussy",
        "boob",
        "tit",
        "nipple",
        "ass",
        "butt",
        "penis",
        "vagina",
        "masturbat",
        "orgasm",
        "penetrat",
        "bbc",
        "creampie",
        "gangbang",
        "threesome",
        "foursome",
        "orgy",
        "handjob",
        "footjob",
        "titjob",
        "lesbian",
        "gay",
        "milf",
        "teen",
        "hentai",
        "ahegao",
        "ecchi",
        "bounce",
        "ride",
        "cowgirl",
        "doggy",
        "missionary",
        "facial",
        "deepthroat",
        "swallow",
        "squirt",
        "fetish",
        "bdsm",
        "bondage",
        "dominat",
        "submiss",
        "slave",
        "whip",
        "spank",
        "choke",
    ]

    # Model name patterns that cause false positives with NSFW keywords
    # e.g., "LTXXX" is a stylized "LTX" model name, not adult "XXX"
    MODEL_NAME_STRIPS = ["ltxxx", "ltx-xxx"]

    def is_nsfw(name: str, path: str) -> bool:
        """Check if a LoRA is NSFW based on name/path."""
        check_str = f"{name} {path}".lower()
        # Strip model name patterns that cause false positives
        for strip in MODEL_NAME_STRIPS:
            check_str = check_str.replace(strip, "ltx")
        # Check for NSFW keywords
        for kw in NSFW_KEYWORDS:
            if kw in check_str:
                return True
        return False

    all_loras = []
    high_noise = []
    low_noise = []
    general = []
    by_category = {}  # Group by subdirectory

    for scan_dir in lora_dirs:
        for lora_path in scan_dir.rglob("*.safetensors"):
            # Get relative path from loras folder
            rel_path = str(lora_path.relative_to(scan_dir))
            name = lora_path.stem

            # Get category (subdirectory name, or "root" for top-level files)
            parent = lora_path.parent.relative_to(scan_dir)
            category = str(parent) if str(parent) != "." else "root"

            # Detect NSFW
            nsfw = is_nsfw(name, rel_path)

            lora_info = {
                "path": rel_path,
                "name": name,
                "category": category,
                "size_mb": round(lora_path.stat().st_size / (1024 * 1024), 1),
                "nsfw": nsfw,
            }
            all_loras.append(lora_info)

            # Group by category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(lora_info)

            # Categorize by noise type
            lower_name = name.lower()
            lower_path = rel_path.lower()

            if (
                "high" in lower_name
                or "high" in lower_path
                or "_h_" in lower_name
                or "-h-" in lower_name
            ):
                high_noise.append(lora_info)
            elif (
                "low" in lower_name
                or "low" in lower_path
                or "_l_" in lower_name
                or "-l-" in lower_name
            ):
                low_noise.append(lora_info)
            else:
                general.append(lora_info)

    # Sort by name
    all_loras.sort(key=lambda x: x["name"].lower())
    high_noise.sort(key=lambda x: x["name"].lower())
    low_noise.sort(key=lambda x: x["name"].lower())
    general.sort(key=lambda x: x["name"].lower())

    # Sort each category
    for cat in by_category:
        by_category[cat].sort(key=lambda x: x["name"].lower())

    return {
        "loras": all_loras,
        "high_noise": high_noise,
        "low_noise": low_noise,
        "general": general,
        "by_category": by_category,
        "count": len(all_loras),
    }


@app.get("/loras/registry")
async def get_lora_registry():
    """Get all LoRAs enriched with registry metadata (trigger words, strengths, source URLs)."""
    from lora_scanner import lora_cache

    all_loras = lora_cache.get_all()
    return {
        "loras": [lora_cache.to_dict(l) for l in all_loras],
        "count": len(all_loras),
        "registry_count": sum(1 for l in all_loras if l.registry is not None),
    }


@app.post("/loras/validate")
async def validate_lora_config(request: Request):
    """Validate LoRA usage: check trigger words in prompt, strength ranges, dual-noise pairs.

    Body: {"loras": [{"filename": "...", "strength": 1.0}], "positive_prompt": "..."}
    """
    from lora_scanner import validate_lora_batch
    from dataclasses import asdict

    body = await request.json()
    loras = body.get("loras", [])
    prompt = body.get("positive_prompt", "")

    if not loras:
        return {"validations": [], "all_valid": True}

    results = validate_lora_batch(loras, prompt)
    return {
        "validations": [asdict(v) for v in results],
        "all_valid": all(v.is_valid for v in results),
    }


@app.post("/ai-suggest")
async def ai_suggest_settings(request: Request, user: User = Depends(get_current_user)):
    """Analyze current generation settings with LLM and suggest improvements.

    Takes current form state (prompt, LoRAs, resolution, etc.) and available LoRA
    registry data. Returns actionable suggestions the user can accept/reject.

    Body: {
        "prompt": str,
        "negative_prompt": str,
        "tool": "i2v" | "t2v",
        "model_mode": str,
        "resolution": str,
        "steps": int,
        "cfg": float,
        "fps": int,
        "duration": int,
        "loras": [{"filename": str, "strength": float}],
        "model": str | null  # LLM model override
    }
    """

    body = await request.json()
    prompt = body.get("prompt", "").strip()
    negative_prompt = body.get("negative_prompt", "").strip()
    tool = body.get("tool", "i2v")
    model_mode = body.get("model_mode", "")
    resolution = body.get("resolution", "")
    steps = body.get("steps", 6)
    cfg = body.get("cfg", 3.0)
    fps = body.get("fps", 16)
    duration = body.get("duration", 5)
    current_loras = body.get("loras", [])
    model_override = body.get("model")

    if not prompt:
        raise HTTPException(
            status_code=400, detail="Prompt is required for suggestions"
        )

    # Load LoRA registry for context
    from lora_scanner import lora_cache

    all_loras = lora_cache.get_all()

    # Map model_mode to base_model for LoRA filtering
    BASE_MODEL_MAP = {
        "wan2.2": "wan2.2",
        "wan22_standard": "wan2.2",
        "wan22_distorch": "wan2.2",
        "cloud_wan22": "wan2.2",
        "ltx2": "ltx",
        "ltx23": "ltx",
    }
    required_base = BASE_MODEL_MAP.get(model_mode, "")

    # Build compact LoRA catalog for LLM context — minimal fields to keep prompt small
    # STRICT filtering: only include LoRAs confirmed compatible with current tool+model
    lora_catalog = []
    for lora in all_loras:
        lora_base = lora.base_model or (
            lora.registry.base_model if lora.registry else ""
        )

        # Skip LoRAs with unknown base model — they're unverified and can't be applied
        if required_base and not lora_base:
            continue
        # Skip LoRAs that don't match the required base model
        if required_base and lora_base and lora_base != required_base:
            continue
        # Skip LoRAs whose registry modes don't include this tool
        if lora.registry and lora.registry.modes and tool not in lora.registry.modes:
            continue

        # Determine noise_level for frontend apply logic
        noise = lora.noise_level
        if not noise:
            # Infer from filename if scanner missed it
            lower_name = lora.filename.lower()
            lower_path = lora.path.lower()
            if any(
                k in lower_name or k in lower_path
                for k in ("high", "_h_", "-h-", "_hn", "-hn")
            ):
                noise = "high"
            elif any(
                k in lower_name or k in lower_path
                for k in ("low", "_l_", "-l-", "_ln", "-ln")
            ):
                noise = "low"

        entry = {
            "filename": lora.path,
            "name": lora.registry.display_name if lora.registry else lora.name,
            "noise_level": noise or "single",  # "high", "low", or "single" (non-paired)
        }
        if lora.registry:
            if lora.registry.trigger_words:
                entry["trigger_words"] = lora.registry.trigger_words
            if lora.registry.trigger_mode and lora.registry.trigger_mode != "none":
                entry["trigger_mode"] = lora.registry.trigger_mode
            entry["strength"] = lora.registry.recommended_strength
        lora_catalog.append(entry)

    # Build current settings summary
    current_lora_info = []
    for lc in current_loras:
        fn = lc.get("filename", "")
        strength = lc.get("strength", 1.0)
        # Find registry info
        reg_info = next(
            (l for l in all_loras if l.path == fn or l.filename == fn), None
        )
        info = {"filename": fn, "strength": strength}
        if reg_info and reg_info.registry:
            info["trigger_words"] = reg_info.registry.trigger_words
            info["trigger_mode"] = reg_info.registry.trigger_mode
            info["recommended_strength"] = reg_info.registry.recommended_strength
        current_lora_info.append(info)

    # Model-specific constraints for the LLM
    MODEL_CONSTRAINTS = {
        "wan2.2": {
            "cfg_range": "1.0-5.0 (default 1.0, higher = stronger prompt adherence but more artifacts)",
            "steps_range": "4-30 (default 6, higher = better quality but slower)",
            "notes": "Wan 2.2 supports CFG guidance. LoRAs are dual-noise (high+low pairs).",
        },
        "wan22_standard": {
            "cfg_range": "1.0-5.0 (default 1.0, higher = stronger prompt adherence but more artifacts)",
            "steps_range": "4-30 (default 6, higher = better quality but slower)",
            "notes": "Wan 2.2 supports CFG guidance. LoRAs are dual-noise (high+low pairs).",
        },
        "cloud_wan22": {
            "cfg_range": "1.0-5.0 (default 1.0)",
            "steps_range": "4-30 (default 6)",
            "notes": "Wan 2.2 on cloud GPU. Same constraints as wan2.2.",
        },
        "ltx2": {
            "cfg_range": "1.0 ONLY — DO NOT suggest changing CFG. LTX 2.3 distilled is trained without classifier-free guidance. Any value above 1.0 DEGRADES quality.",
            "steps_range": "6-10 (default 8, distilled model needs few steps)",
            "notes": "LTX 2.3 distilled model. CFG must stay at 1.0. Do NOT suggest CFG changes.",
        },
        "ltx23": {
            "cfg_range": "1.0 ONLY — DO NOT suggest changing CFG.",
            "steps_range": "6-10 (default 8)",
            "notes": "LTX 2.3 distilled. CFG must stay at 1.0.",
        },
    }
    constraints = MODEL_CONSTRAINTS.get(model_mode, {})
    constraints_text = ""
    if constraints:
        constraints_text = f"\nModel constraints for {model_mode}:\n- CFG: {constraints['cfg_range']}\n- Steps: {constraints['steps_range']}\n- {constraints['notes']}"

    system_prompt = f"""You are an AI video generation settings optimizer. Analyze user settings and suggest concrete improvements. Max 6 suggestions. Output ONLY a valid JSON array.
{constraints_text}

CRITICAL RULES:
- ONLY suggest LoRAs from the "Available LoRAs" list. Do NOT invent LoRA filenames.
- Each LoRA has a noise_level: "high", "low", or "single". Include it in lora_add suggestions.
- For Wan2.2: LoRAs are dual-noise pairs (high+low). If suggesting a high-noise LoRA, also check if a matching low-noise LoRA exists and suggest both.
- Do NOT suggest lora_trigger separately when lora_add already includes trigger_words — the frontend handles trigger word injection automatically.
- Only suggest lora_trigger for LoRAs that are ALREADY active but missing trigger words in the prompt.

Each suggestion: {{"id":"s1","type":"TYPE","title":"Short title","description":"Why","priority":"high|medium|low","apply":{{...}}}}

Types and apply shapes:
- prompt_add: {{"text":"append this"}}
- prompt_replace: {{"find":"old","replace":"new"}}
- negative_add: {{"text":"append this"}}
- lora_add: {{"filename":"exact_path.safetensors","strength":1.0,"noise_level":"high|low|single","trigger_words":["word"]}}
- lora_strength: {{"filename":"exact_path.safetensors","new_strength":0.8}}
- lora_trigger: {{"text":"trigger words to add","lora_filename":"exact_path.safetensors"}}
- setting_change: {{"setting":"steps|cfg|fps|resolution","value":6}}"""

    user_prompt = f"""Analyze these {tool.upper()} settings and suggest improvements.

Tool: {tool.upper()}, Model: {model_mode}, Resolution: {resolution}
Steps: {steps}, CFG: {cfg}, FPS: {fps}, Duration: {duration}s
Prompt: "{prompt}"
Negative: "{negative_prompt[:200]}"

Active LoRAs: {json.dumps(current_lora_info, separators=(",", ":")) if current_lora_info else "None"}

Available LoRAs (suggest from these only):
{json.dumps(lora_catalog, separators=(",", ":"))}

Focus on: missing trigger words, strength adjustments, matching LoRAs not yet active, prompt improvements. Return JSON array."""

    # Determine LLM model — prefer Gemma 4 (fast MoE, good at structured JSON)
    AI_SUGGEST_MODEL = "Huihui-gemma-4-26B-A4B-it-abliterated"
    ai_settings = load_ai_settings()
    model = model_override or AI_SUGGEST_MODEL

    # Gemma 4 26B needs ~14GB VRAM — free ComfyUI first
    from guardian_client import (
        wait_for_comfyui_idle,
        free_comfyui_vram as _free_comfy_vram,
    )

    await wait_for_comfyui_idle()
    await _free_comfy_vram()

    # Retry loop for 503 (Guardian reloading model)
    max_retries = 3
    retry_delay = 15
    data = None

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(
                timeout=300.0, headers=_guardian_headers()
            ) as client:
                resp = await client.post(
                    f"{GUARDIAN_BASE}/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.3,
                        "max_tokens": 4096,
                    },
                )

                if resp.status_code == 503 and attempt < max_retries - 1:
                    logger.info(
                        f"⏳ Guardian 503 (model loading), retry {attempt + 1}/{max_retries} in {retry_delay}s..."
                    )
                    await asyncio.sleep(retry_delay)
                    continue

                resp.raise_for_status()
                data = resp.json()
                break

        except httpx.ReadTimeout:
            if attempt < max_retries - 1:
                logger.warning(
                    f"⏳ AI suggest timeout, retry {attempt + 1}/{max_retries}..."
                )
                continue
            logger.error("AI suggest error: LLM request timed out after retries")
            raise HTTPException(
                status_code=504,
                detail="LLM took too long to respond. Try again or use a shorter prompt.",
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Guardian LLM not available")
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                logger.info(
                    f"⏳ Guardian 503, retry {attempt + 1}/{max_retries} in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
                continue
            raise

    if data is None:
        raise HTTPException(
            status_code=503,
            detail="Guardian still loading model after retries — try again",
        )

    try:
        finish_reason = data.get("choices", [{}])[0].get("finish_reason", "unknown")
        logger.info(
            f"🐛 AI suggest LLM responded: finish_reason={finish_reason}, model={data.get('model', '?')}"
        )

        msg = data["choices"][0]["message"]
        content = (msg.get("content") or msg.get("reasoning_content", "")).strip()
        content = _strip_think_tags(content)
        logger.info(
            f"🐛 AI suggest LLM response ({len(content)} chars, finish={finish_reason}): {content[:300]}"
        )

        # Parse JSON from response (handle markdown code blocks)
        if not content:
            detail = (
                "LLM returned empty response"
                + (" (hit token limit)" if finish_reason == "length" else "")
                + ". Try again."
            )
            logger.warning(f"AI suggest: {detail}")
            return JSONResponse(status_code=502, content={"detail": detail})
        if content.startswith("```"):
            # Remove markdown code fences
            lines = content.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines).strip()

        # Try to extract JSON array if LLM added extra text around it
        if not content.startswith("["):
            start = content.find("[")
            end = content.rfind("]")
            if start != -1 and end != -1:
                content = content[start : end + 1]

        suggestions = json.loads(content)
        if not isinstance(suggestions, list):
            suggestions = []

        # Validate and sanitize suggestions
        valid_types = {
            "prompt_add",
            "prompt_replace",
            "negative_add",
            "lora_add",
            "lora_strength",
            "lora_trigger",
            "setting_change",
        }
        sanitized = []
        for s in suggestions:
            if isinstance(s, dict) and s.get("type") in valid_types and "apply" in s:
                sanitized.append(
                    {
                        "id": s.get("id", f"s{len(sanitized)}"),
                        "type": s["type"],
                        "title": s.get("title", "Suggestion"),
                        "description": s.get("description", ""),
                        "priority": s.get("priority", "medium"),
                        "apply": s["apply"],
                        "checked": True,
                    }
                )
        sanitized.sort(
            key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x["priority"], 1)
        )

        return {
            "suggestions": sanitized,
            "model_used": model,
            "settings_analyzed": {
                "tool": tool,
                "model_mode": model_mode,
                "resolution": resolution,
                "active_loras": len(current_loras),
            },
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM suggestions JSON: {e}")
        raise HTTPException(
            status_code=502, detail="LLM returned invalid suggestion format"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI suggest error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail="AI suggestion failed")


@app.get("/unet-models")
async def list_unet_models():
    """
    List available GGUF unet models for Wan2.2 I2V.
    Returns pairs of high/low noise models.
    """
    unet_dir = Path("/home/flip/oelala/ComfyUI/models/unet")

    if not unet_dir.exists():
        return {"models": [], "pairs": []}

    all_models = []
    high_noise = []
    low_noise = []

    for model_path in unet_dir.rglob("*.gguf"):
        rel_path = str(model_path.relative_to(unet_dir))
        name = model_path.stem

        model_info = {
            "path": rel_path,
            "name": name,
            "size_gb": round(model_path.stat().st_size / (1024 * 1024 * 1024), 2),
        }
        all_models.append(model_info)

        lower_name = name.lower()
        lower_path = rel_path.lower()

        if "high" in lower_name or "high" in lower_path:
            high_noise.append(model_info)
        elif "low" in lower_name or "low" in lower_path:
            low_noise.append(model_info)

    # Sort
    all_models.sort(key=lambda x: x["name"].lower())
    high_noise.sort(key=lambda x: x["name"].lower())
    low_noise.sort(key=lambda x: x["name"].lower())

    # Try to match pairs by similar names
    pairs = []
    for h in high_noise:
        h_base = (
            h["name"]
            .lower()
            .replace("high", "")
            .replace("_h_", "_")
            .replace("-h-", "-")
        )
        for l in low_noise:
            l_base = (
                l["name"]
                .lower()
                .replace("low", "")
                .replace("_l_", "_")
                .replace("-l-", "-")
            )
            # Check similarity
            if h_base == l_base or h_base.replace("noise", "") == l_base.replace(
                "noise", ""
            ):
                pairs.append(
                    {
                        "name": h["name"]
                        .replace("high", "")
                        .replace("High", "")
                        .replace("_H_", "_")
                        .replace("HIGH", "")
                        .strip("_- ")
                        or h["name"],
                        "high": h,
                        "low": l,
                    }
                )
                break

    return {
        "models": all_models,
        "high_noise": high_noise,
        "low_noise": low_noise,
        "pairs": pairs,
        "count": len(all_models),
    }


# =============================================================================
# ComfyUI Queue Management Endpoints
# =============================================================================

# In-memory store for active jobs submitted through Oelala
# Maps prompt_id -> {status, prompt, created_at, output_path, ...}
active_jobs = {}

# Store for pending post-processing chains
# Maps prompt_id -> [{type: "upscale", scale: 2}, {type: "interpolate", target_fps: 60}]
pending_post_processing = {}

# Persistence file for cloud jobs (survives backend restarts)
CLOUD_JOBS_FILE = Path("/home/flip/oelala/data/cloud_jobs.json")


def _persist_cloud_jobs() -> None:
    """Save active cloud jobs to disk so they survive backend restarts."""
    try:
        cloud_jobs = {
            pid: info
            for pid, info in active_jobs.items()
            if info.get("compute_target") == "cloud"
            and not info.get("_cloud_completed")
        }
        if not cloud_jobs:
            # Remove file if no active cloud jobs
            if CLOUD_JOBS_FILE.exists():
                CLOUD_JOBS_FILE.unlink()
            return
        # Filter out non-serializable keys
        serializable = {}
        for pid, info in cloud_jobs.items():
            serializable[pid] = {
                k: v
                for k, v in info.items()
                if not k.startswith("_") or k in ("_cloud_status", "_start_time")
            }
        CLOUD_JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(CLOUD_JOBS_FILE, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.debug(f"☁️ Persisted {len(serializable)} cloud job(s) to disk")
    except Exception as e:
        logger.warning(f"⚠️ Failed to persist cloud jobs: {e}")


def _restore_cloud_jobs() -> int:
    """Restore cloud jobs from disk on startup. Returns count restored."""
    if not CLOUD_JOBS_FILE.exists():
        return 0
    try:
        with open(CLOUD_JOBS_FILE, "r") as f:
            saved_jobs = json.load(f)
        count = 0
        for pid, info in saved_jobs.items():
            if pid not in active_jobs:
                if not info.get("_start_time"):
                    created_at = info.get("created_at")
                    if created_at:
                        try:
                            info["_start_time"] = datetime.strptime(
                                created_at, "%Y%m%d_%H%M%S"
                            ).timestamp()
                        except Exception:
                            info["_start_time"] = time.time()
                    else:
                        info["_start_time"] = time.time()
                active_jobs[pid] = info
                count += 1
                logger.info(
                    f"☁️ Restored cloud job: {pid} (RunPod: {info.get('runpod_job_id', '?')})"
                )
        if count > 0:
            logger.info(f"☁️ Restored {count} cloud job(s) from previous session")
        return count
    except Exception as e:
        logger.warning(f"⚠️ Failed to restore cloud jobs: {e}")
        return 0


# Restore cloud jobs on module load
_restore_cloud_jobs()

# ── Background cloud job poller ─────────────────────────────────────────
# Polls active (non-completed) cloud jobs every CLOUD_POLL_INTERVAL seconds
# so completions are processed even if the frontend isn't watching.
CLOUD_POLL_INTERVAL = int(os.getenv("CLOUD_POLL_INTERVAL", "10"))
# Max age before a cloud job is considered abandoned (2 hours)
CLOUD_JOB_MAX_AGE = int(os.getenv("CLOUD_JOB_MAX_AGE", "7200"))

_cloud_poller_task: Optional[asyncio.Task] = None


async def _cloud_job_poller() -> None:
    """Background task: periodically poll all active cloud jobs for completion."""
    logger.info(f"☁️ Cloud job poller started (interval={CLOUD_POLL_INTERVAL}s)")
    while True:
        try:
            await asyncio.sleep(CLOUD_POLL_INTERVAL)
            if not _runpod:
                continue

            # Collect cloud jobs that need polling
            cloud_jobs = {
                pid: info
                for pid, info in active_jobs.items()
                if info.get("compute_target") == "cloud"
                and not info.get("_cloud_completed")
            }
            if not cloud_jobs:
                continue

            logger.debug(f"☁️ Background poll: {len(cloud_jobs)} active cloud job(s)")

            for prompt_id, job_info in list(cloud_jobs.items()):
                try:
                    # Check if job is too old (RunPod may have purged it)
                    job_age = time.time() - float(
                        job_info.get("_start_time", time.time())
                    )
                    if job_age > CLOUD_JOB_MAX_AGE:
                        logger.warning(
                            f"☁️ Expiring stale cloud job {prompt_id} "
                            f"(age={int(job_age)}s, runpod={job_info.get('runpod_job_id')})"
                        )
                        error_msg = f"Cloud job expired (age {int(job_age)}s exceeds {CLOUD_JOB_MAX_AGE}s limit)"
                        active_jobs[prompt_id]["_cloud_completed"] = True
                        active_jobs[prompt_id]["_cloud_status"] = "EXPIRED"
                        active_jobs[prompt_id]["_cloud_error"] = error_msg
                        _cloud_completed_cache[prompt_id] = {
                            "prompt_id": prompt_id,
                            "status": "failed",
                            "error": error_msg,
                            "compute_target": "cloud",
                        }
                        record_generation_complete(
                            prompt_id, success=False, error=error_msg
                        )
                        await _refund_cloud_job_credits(prompt_id, job_info, error_msg)
                        _persist_cloud_jobs()
                        continue

                    result = await _handle_cloud_job_status(prompt_id, job_info)
                    status = result.get("status")
                    if status in ("completed", "failed"):
                        logger.info(f"☁️ Background poll resolved {prompt_id}: {status}")
                except Exception as e:
                    logger.warning(f"☁️ Background poll error for {prompt_id}: {e}")
                # Small delay between individual polls to avoid hammering RunPod
                await asyncio.sleep(2)

            # Cleanup: remove completed cloud jobs from active_jobs after 5 min
            # (keeps them briefly so frontend can read the final status)
            CLOUD_CLEANUP_DELAY = 300  # 5 minutes
            stale_ids = [
                pid
                for pid, info in active_jobs.items()
                if info.get("compute_target") == "cloud"
                and info.get("_cloud_completed")
                and (time.time() - float(info.get("_start_time", time.time())))
                > CLOUD_CLEANUP_DELAY
            ]
            for pid in stale_ids:
                del active_jobs[pid]
                logger.debug(f"☁️ Cleaned up completed cloud job {pid} from memory")

        except asyncio.CancelledError:
            logger.info("☁️ Cloud job poller stopping")
            break
        except Exception as e:
            logger.error(f"☁️ Cloud poller unexpected error: {e}")
            await asyncio.sleep(10)


def _get_cached_local_result(
    prompt_id: str, user_id: Optional[str] = None
) -> Optional[dict]:
    """Return a cached local result when it belongs to the requesting user."""

    result = _local_completed_cache.get(prompt_id)
    if not result:
        return None
    if user_id is not None and result.get("user_id") != user_id:
        return None
    return {k: v for k, v in result.items() if k != "_cached_at"}


async def _resolve_local_job_result(prompt_id: str, job_info: dict) -> Optional[dict]:
    """Resolve a local ComfyUI job from history and trigger upload when ready."""
    import requests

    cached = _get_cached_local_result(prompt_id)
    if cached is not None:
        return cached

    # Resolve the ComfyUI client that actually ran the job. Prefer the
    # backend_id attached at dispatch (config-driven via the compute backend
    # inventory, e.g. Windows-PC for MiniMax-H3), falling back to the legacy
    # adapter-name check.
    adapter_name = job_info.get("adapter_name", "")
    backend_id = job_info.get("backend_id")
    comfyui = None
    if backend_id:
        try:
            from comfyui_client import get_comfyui_client_for_backend as _gcfb
            from src.backend.generation.compute_backends import get_backend

            _backend = get_backend(backend_id)
            if _backend is not None:
                comfyui = _gcfb(_backend)
        except Exception:
            # Keep good diagnostics: if resolving the configured backend fails
            # (bad inventory entry, import error, ...), log it with the
            # backend_id before falling back so polling/downloads never silently
            # hit the wrong ComfyUI server without breadcrumbs.
            logger.warning(
                f"⚠️ Failed to resolve ComfyUI client for backend_id={backend_id}; "
                "falling back",
                exc_info=True,
            )
            comfyui = None
    is_windows_h3 = adapter_name in (
        "minimax-h3-local-t2v",
        "minimax-h3-local-i2v",
    )
    if comfyui is None and is_windows_h3 and get_windows_comfyui_client is not None:
        comfyui = get_windows_comfyui_client()
    if comfyui is None:
        comfyui = get_comfyui_client()
    if comfyui is None:
        return None
    comfyui_base = comfyui.base_url

    history_resp = requests.get(
        f"{comfyui_base}/history/{quote(prompt_id, safe='')}", timeout=5
    )
    if history_resp.status_code != 200:
        return None

    history = history_resp.json()
    if prompt_id not in history:
        return None

    job_data = history[prompt_id]
    outputs = job_data.get("outputs", {})

    output_video = None
    output_image = None
    output_audio = None
    for node_output in outputs.values():
        if "gifs" in node_output and output_video is None:
            for gif in node_output["gifs"]:
                if gif.get("type") == "output":
                    output_video = gif
                    break
        # SaveVideo (MiniMax-H3 local) writes the mp4 under the "images" key
        # (node output) with a .mp4 filename — recognise it as a video.
        if (
            output_video is None
            and "images" in node_output
            and node_output.get("images")
        ):
            for img in node_output["images"]:
                if img.get("type") == "output" and str(
                    img.get("filename", "")
                ).lower().endswith((".mp4", ".webm", ".mov", ".mkv")):
                    output_video = img
                    break
        if "images" in node_output and output_image is None:
            for img in node_output["images"]:
                if img.get("type") == "output" and not str(
                    img.get("filename", "")
                ).lower().endswith((".mp4", ".webm", ".mov", ".mkv")):
                    output_image = (
                        f"/comfyui/output/{_safe_filename(img.get('filename'))}"
                    )
                    break
        if "audio" in node_output and output_audio is None:
            for audio in node_output["audio"]:
                if audio.get("type") == "output":
                    output_audio = (
                        f"/comfyui/output/{_safe_filename(audio.get('filename'))}"
                    )
                    break

    # If the video was found as an output record (SaveVideo / VHS), try to
    # download it from the job's ComfyUI server so we have a local file to
    # upload. For Windows-PC (H3 local) the file is NOT on ai-kvm2, so this
    # download is required.
    if output_video and isinstance(output_video, dict):
        vf = output_video.get("filename")
        output_filename = _safe_filename(vf) if vf else None
        if output_filename:
            dl_params = {
                "filename": vf,
                "subfolder": output_video.get("subfolder", ""),
                "type": output_video.get("type", "output"),
            }
            try:
                dl_resp = requests.get(
                    f"{comfyui_base}/view", params=dl_params, timeout=15
                )
                if dl_resp.status_code == 200:
                    out_dir = Path(COMFYUI_OUTPUT_DIR)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / output_filename
                    out_path.write_bytes(dl_resp.content)
                    output_video = f"/comfyui/output/{output_filename}"
                else:
                    logger.warning(
                        f"⚠️ Failed to download {output_filename}: HTTP {dl_resp.status_code}"
                    )
                    output_video = None
            except Exception as e:
                logger.error(
                    f"❌ Failed to download {output_filename} from {comfyui_base}: {e}"
                )
                output_video = None

    storage_path = None
    signed_url = None
    if comfyui and (output_video or output_image or output_audio):
        if output_video:
            output_type = "video"
            output_filename = output_video.split("/")[-1]
        elif output_image:
            output_type = "image"
            output_filename = output_image.split("/")[-1]
        else:
            output_type = "audio"
            output_filename = output_audio.split("/")[-1]

        output_path = _safe_child_path(COMFYUI_OUTPUT_DIR, output_filename)

        if prompt_id in pending_post_processing:
            chain_info = pending_post_processing.pop(prompt_id)
            logger.info(f"🔄 Continuing post-processing chain for {prompt_id}")
            await trigger_post_processing_chain(
                prompt_id=prompt_id,
                video_path=str(output_path),
                post_processing=chain_info["steps"],
                user_id=chain_info["user_id"],
                post_audio_path=chain_info.get("post_audio_path"),
            )
        elif job_info.get("post_processing") and output_video:
            logger.info(f"🔄 Starting post-processing chain for {prompt_id}")
            await trigger_post_processing_chain(
                prompt_id=prompt_id,
                video_path=str(output_path),
                post_processing=job_info["post_processing"],
                user_id=job_info.get("user_id", "unknown"),
                post_audio_path=job_info.get("post_audio_path"),
            )

        if output_path.exists():
            storage_path = await comfyui.on_job_complete_async(
                prompt_id, str(output_path), output_type
            )
            if storage_path:
                logger.info(
                    f"✅ Auto-uploaded {output_type} for job {prompt_id}: {storage_path}"
                )
                signed_url = get_signed_media_url(storage_path, expires_in=86400)

    _local_log = format_comfyui_history_log(prompt_id, job_data)
    record_generation_complete(prompt_id, success=True, log_text=_local_log)

    result = {
        "prompt_id": prompt_id,
        "status": "completed",
        "output_video": output_video,
        "output_image": output_image,
        "output_audio": output_audio,
        "url": signed_url or output_image or output_video or output_audio,
        "signed_url": signed_url,
        "storage_path": storage_path,
        **job_info,
        "_cached_at": time.time(),
    }
    _local_completed_cache[prompt_id] = result
    return _get_cached_local_result(prompt_id)


async def _local_job_poller() -> None:
    """Background task: resolve local ComfyUI completions without frontend polling."""

    logger.info(f"🖼️ Local job poller started (interval={LOCAL_JOB_POLL_INTERVAL}s)")
    while True:
        try:
            await asyncio.sleep(LOCAL_JOB_POLL_INTERVAL)

            local_jobs = {
                pid: info
                for pid, info in active_jobs.items()
                if info.get("compute_target") == "local"
            }

            for prompt_id, job_info in list(local_jobs.items()):
                try:
                    result = await _resolve_local_job_result(prompt_id, job_info)
                    if result and result.get("status") in ("completed", "failed"):
                        active_jobs.pop(prompt_id, None)
                        logger.info(
                            f"🖼️ Background poll resolved {prompt_id}: {result.get('status')}"
                        )
                except Exception as e:
                    logger.warning(f"🖼️ Background poll error for {prompt_id}: {e}")
                await asyncio.sleep(0.5)

            stale_ids = [
                pid
                for pid, result in _local_completed_cache.items()
                if (time.time() - float(result.get("_cached_at", time.time())))
                > LOCAL_COMPLETED_CACHE_TTL
            ]
            for pid in stale_ids:
                del _local_completed_cache[pid]

        except asyncio.CancelledError:
            logger.info("🖼️ Local job poller stopping")
            break
        except Exception as e:
            logger.error(f"🖼️ Local poller unexpected error: {e}")
            await asyncio.sleep(10)


# Generation stats file for analysis
GENERATION_STATS_FILE = Path("/home/flip/oelala/data/generation_stats.json")


def load_generation_stats() -> list:
    """Load generation stats from file"""
    if GENERATION_STATS_FILE.exists():
        try:
            with open(GENERATION_STATS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load generation stats: {e}")
    return []


def save_generation_stat(stat: dict) -> bool:
    """Append a generation stat to the stats file"""
    try:
        GENERATION_STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        stats = load_generation_stats()
        stats.append(stat)
        # Keep only last 10000 entries to prevent file from growing too large
        if len(stats) > 10000:
            stats = stats[-10000:]
        with open(GENERATION_STATS_FILE, "w") as f:
            json.dump(stats, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save generation stat: {e}")
        return False


def record_generation_start(prompt_id: str, job_info: dict) -> None:
    """Record the start of a generation job"""
    import time

    if prompt_id in active_jobs:
        active_jobs[prompt_id]["_start_time"] = time.time()
        active_jobs[prompt_id]["_job_type"] = job_info.get("job_type", "unknown")
        # Persist user_id so record_generation_complete can upload log to user bucket
        if "user_id" in job_info:
            active_jobs[prompt_id]["user_id"] = job_info["user_id"]


def track_v2_local_job(prompt_id: str, job_info: dict) -> None:
    """Register a queued local V2 job in active_jobs for queue/status polling."""

    active_jobs[prompt_id] = {
        "prompt_id": prompt_id,
        "user_id": job_info.get("user_id"),
        "prompt": job_info.get("prompt", ""),
        "job_type": job_info.get("job_type", "unknown"),
        "model_name": job_info.get("model_name") or job_info.get("model_type"),
        "model_type": job_info.get("model_type"),
        "resolution": job_info.get("resolution"),
        "aspect_ratio": job_info.get("aspect_ratio"),
        "num_frames": job_info.get("num_frames"),
        "fps": job_info.get("fps"),
        "steps": job_info.get("steps"),
        "cfg": job_info.get("cfg"),
        "sampler": job_info.get("sampler"),
        "scheduler": job_info.get("scheduler"),
        "adapter_name": job_info.get("adapter_name"),
        "source": job_info.get("source", "v2"),
        "compute_target": "local",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    record_generation_start(prompt_id, active_jobs[prompt_id])


def record_generation_complete(
    prompt_id: str, success: bool = True, error: str = None, log_text: str = None
) -> None:
    """Record completion of a generation job and save stats + execution log to user bucket."""
    import time

    job_info = active_jobs.get(prompt_id, {})
    start_time = job_info.get("_start_time")

    if not start_time:
        logger.warning(f"No start time recorded for job {prompt_id}")
        return

    duration_seconds = time.time() - start_time

    stat = {
        "prompt_id": prompt_id,
        "timestamp": datetime.now().isoformat(),
        "duration_seconds": round(duration_seconds, 2),
        "success": success,
        "error": error,
        "job_type": job_info.get("_job_type", "unknown"),
        "resolution": job_info.get("resolution", "unknown"),
        "aspect_ratio": job_info.get("aspect_ratio", "unknown"),
        "num_frames": job_info.get("num_frames", 0),
        "fps": job_info.get("fps", 0),
        "steps": job_info.get("steps", 0),
        "model_mode": job_info.get("model_mode", "wan2.2"),
        "extend_mode": job_info.get("extend_mode", False),
        "clip_count": job_info.get("clip_count", 1),
        "lora_count": job_info.get("lora_count", 0),
        "cfg": job_info.get("cfg", 1.0),
    }

    save_generation_stat(stat)
    logger.info(
        f"📊 Generation stats recorded: {prompt_id} - {duration_seconds:.1f}s {'✅' if success else '❌'}"
    )

    # Save execution log to user's generation artifact bundle
    user_id = job_info.get("user_id")
    if user_id:
        status_str = "completed" if success else "failed"
        # Build log content: use provided log_text or generate a summary from stat
        if not log_text:
            log_lines = [
                f"prompt_id: {prompt_id}",
                f"status: {status_str}",
                f"timestamp: {stat['timestamp']}",
                f"duration_seconds: {stat['duration_seconds']}",
                f"job_type: {stat['job_type']}",
                f"resolution: {stat['resolution']}",
                f"num_frames: {stat['num_frames']}",
                f"model_mode: {stat['model_mode']}",
                f"steps: {stat['steps']}",
                f"compute_target: {job_info.get('compute_target', 'local')}",
            ]
            if error:
                log_lines.append(f"error: {error}")
            log_text = "\n".join(log_lines) + "\n"
        save_gen_logs(user_id, prompt_id, log_text, status_str, duration_seconds)


# LoRA source directory (SSD)
LORA_DIR = Path("/mnt/ssd/loras")

# HuggingFace CDN sources for LoRAs — faster than self-hosted download.
# Map: resolved_filename → {"repo": "user/repo", "path": "file_in_repo.safetensors"}
# LoRAs listed here are downloaded from HF CDN by RunPod workers.
# LoRAs NOT listed here fall back to self-hosted download via api.oelala.xyz.
LORA_HF_SOURCES: dict[str, dict[str, str]] = {
    "ltx/DR34ML4Y_LTXXX_PREVIEW_RC1.safetensors": {
        "repo": "m0nk111/oelala-loras",
        "path": "ltx/DR34ML4Y_LTXXX_PREVIEW_RC1.safetensors",
    },
    "ltx/LTX-2.3 - Ahegao Face v1.safetensors": {
        "repo": "m0nk111/oelala-loras",
        "path": "ltx/LTX-2.3 - Ahegao Face v1.safetensors",
    },
    "ltx/SexGod_Nudity_LTX23_v2_0.safetensors": {
        "repo": "m0nk111/oelala-loras",
        "path": "ltx/SexGod_Nudity_LTX23_v2_0.safetensors",
    },
    "ltx/bounceV2_LTX23_I2V.comfy.safetensors": {
        "repo": "m0nk111/oelala-loras",
        "path": "ltx/bounceV2_LTX23_I2V.comfy.safetensors",
    },
    "ltx/head_swap_v3_rank_adaptive_fro_098.safetensors": {
        "repo": "m0nk111/oelala-loras",
        "path": "ltx/head_swap_v3_rank_adaptive_fro_098.safetensors",
    },
    "ltx/ltx2.3_nsfw_furry.safetensors": {
        "repo": "m0nk111/oelala-loras",
        "path": "ltx/ltx2.3_nsfw_furry.safetensors",
    },
    "ltx/ltxdeepthroat_v01.safetensors": {
        "repo": "m0nk111/oelala-loras",
        "path": "ltx/ltxdeepthroat_v01.safetensors",
    },
    "ltx/sfbehind_LTX2_3_v0_1.safetensors": {
        "repo": "m0nk111/oelala-loras",
        "path": "ltx/sfbehind_LTX2_3_v0_1.safetensors",
    },
}
# HF token for private repos (optional — public repos don't need it)
HF_LORA_TOKEN = os.getenv("HF_LORA_TOKEN", "")

# Cache for completed cloud jobs (prevent re-processing on repeated polls)
_cloud_completed_cache: dict[str, dict] = {}
# Cache for completed local jobs so the frontend can still fetch the final result
# after the queue entry has been cleaned up.
_local_completed_cache: dict[str, dict] = {}

# Guard against cloud jobs that stay queued forever when RunPod never provisions a worker.
CLOUD_QUEUE_TIMEOUT_SECONDS = int(os.getenv("RUNPOD_QUEUE_TIMEOUT_SECONDS", "300"))
LOCAL_JOB_POLL_INTERVAL = int(os.getenv("LOCAL_JOB_POLL_INTERVAL", "3"))
LOCAL_COMPLETED_CACHE_TTL = int(os.getenv("LOCAL_COMPLETED_CACHE_TTL", "600"))
_local_poller_task: Optional[asyncio.Task] = None


def _lora_download_token(filename: str) -> str:
    """Generate HMAC-SHA256 token for LoRA download URL validation."""
    import hmac
    import hashlib

    key = os.getenv("RUNPOD_API_KEY", "fallback-lora-key").encode()
    return hmac.new(key, filename.encode(), hashlib.sha256).hexdigest()[:32]


def _resolve_lora_path(name: str) -> tuple[Path, str] | tuple[None, None]:
    """
    Resolve a LoRA name to its actual file path.
    Handles names with or without .safetensors extension,
    and LoRAs in subdirectories.

    Returns (full_path, filename) or (None, None) if not found.
    """
    try:
        name = _normalize_storage_key(name)
    except HTTPException:
        return None, None

    lora_root = LORA_DIR.resolve()

    # Try exact path first (may include extension and/or subdirectory)
    exact = (lora_root / name).resolve()
    if exact.is_file():
        try:
            return exact, str(exact.relative_to(lora_root))
        except ValueError:
            return None, None

    # Try adding .safetensors extension
    with_ext = (lora_root / f"{name}.safetensors").resolve()
    if with_ext.is_file():
        try:
            return with_ext, str(with_ext.relative_to(lora_root))
        except ValueError:
            return None, None

    if "/" in name:
        return None, None

    # Search subdirectories for exact filename match
    for match in lora_root.rglob(name):
        if match.is_file():
            return match, str(match.relative_to(lora_root))

    # Search subdirectories with extension added
    for match in lora_root.rglob(f"{name}.safetensors"):
        if match.is_file():
            return match, str(match.relative_to(lora_root))

    return None, None


CLOUD_LOGS_DIR = Path("/home/flip/oelala/logs/cloud")
CLOUD_LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _save_cloud_logs(runpod_job_id: str, prompt_id: str, rp_job) -> Optional[Path]:
    """Save raw ComfyUI logs from a cloud job to logs/cloud/."""
    try:
        output = rp_job.output
        logs_text = None
        if isinstance(output, dict):
            logs_text = output.get("logs")
        if not logs_text:
            return None
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        status_val = (
            rp_job.status.value
            if hasattr(rp_job.status, "value")
            else str(rp_job.status)
        )
        log_file = CLOUD_LOGS_DIR / f"{ts}_{status_val}_{runpod_job_id}.log"
        log_file.write_text(logs_text)
        logger.info(f"☁️ Saved cloud logs: {log_file.name} ({len(logs_text)} chars)")
        return log_file
    except Exception as e:
        logger.warning(f"⚠️ Failed to save cloud logs: {e}")
        return None


def _mark_cloud_job_timed_out(
    prompt_id: str, job_info: dict, queue_age: float
) -> Optional[dict]:
    """Mark a cloud job as failed when it sits in queue beyond the allowed timeout."""
    if queue_age < CLOUD_QUEUE_TIMEOUT_SECONDS:
        return None

    runpod_job_id = job_info.get("runpod_job_id", "?")
    error_msg = f"RunPod job stuck in queue for {int(queue_age)}s; no cloud worker was provisioned"
    logger.error(f"☁️ Cloud job timed out in queue: {runpod_job_id} — {error_msg}")

    result = {
        "prompt_id": prompt_id,
        "status": "failed",
        "error": error_msg,
        "compute_target": "cloud",
        **job_info,
    }
    _cloud_completed_cache[prompt_id] = result

    if prompt_id in active_jobs:
        active_jobs[prompt_id]["_cloud_completed"] = True
        active_jobs[prompt_id]["_cloud_status"] = "TIMED_OUT"
        active_jobs[prompt_id]["_cloud_error"] = error_msg

    _persist_cloud_jobs()
    record_generation_complete(prompt_id, success=False, error=error_msg)
    return result


async def _refund_cloud_job_credits(
    prompt_id: str,
    job_info: dict,
    reason: str = "Cloud generation failed - credits refunded",
) -> bool:
    """
    Refund credits for a failed/timed-out cloud job.
    Idempotent — uses the _credit_refunded flag to prevent double refunds.
    """
    # Guard against double refund
    if active_jobs.get(prompt_id, {}).get("_credit_refunded"):
        logger.debug(f"⚠️ Refund skipped for {prompt_id} — already refunded")
        return False

    user_id = job_info.get("user_id")
    credits_required = job_info.get("credits_required", 0)

    if not user_id or not credits_required:
        logger.debug(
            f"⚠️ Refund skipped for {prompt_id}: user_id={user_id}, credits={credits_required}"
        )
        return False

    try:
        manager = get_credit_manager()
        await manager.refund(
            user_id=user_id,
            amount=credits_required,
            reference_id=prompt_id,
            reason=reason,
        )
        logger.info(
            f"🔄 Refunded {credits_required} credits to user {user_id} for failed cloud job {prompt_id}"
        )
        if prompt_id in active_jobs:
            active_jobs[prompt_id]["_credit_refunded"] = True
        return True
    except Exception as e:
        logger.error(f"❌ Failed to refund credits for {prompt_id}: {e}")
        return False


async def _handle_cloud_job_status(prompt_id: str, job_info: dict) -> dict:
    """
    Handle status polling for a cloud (RunPod) job.

    When the job completes, this function:
    1. Decodes the base64 video output from RunPod
    2. Saves it locally to media/generated/cloud-wan22/
    3. Uploads to MinIO via MediaService
    4. Returns the same response format as local jobs
    5. Saves raw ComfyUI logs to logs/cloud/
    """
    # Return cached result if already processed
    if prompt_id in _cloud_completed_cache:
        return _cloud_completed_cache[prompt_id]

    runpod_job_id = job_info.get("runpod_job_id")
    if not runpod_job_id:
        return {
            "prompt_id": prompt_id,
            "status": "failed",
            "error": "Missing runpod_job_id",
            **job_info,
        }

    try:
        rp_job = await _runpod.get_job_status(
            runpod_job_id, endpoint_id=job_info.get("runpod_endpoint_id")
        )
    except Exception as e:
        error_str = str(e)
        # 404 = RunPod purged the job — it's gone forever
        if "404" in error_str:
            error_msg = f"RunPod job expired/purged (404): {runpod_job_id}"
            logger.warning(f"☁️ {error_msg}")
            result = {
                "prompt_id": prompt_id,
                "status": "failed",
                "error": error_msg,
                "compute_target": "cloud",
                **job_info,
            }
            _cloud_completed_cache[prompt_id] = result
            if prompt_id in active_jobs:
                active_jobs[prompt_id]["_cloud_completed"] = True
                active_jobs[prompt_id]["_cloud_status"] = "EXPIRED"
                active_jobs[prompt_id]["_cloud_error"] = error_msg
            _persist_cloud_jobs()
            record_generation_complete(prompt_id, success=False, error=error_msg)
            await _refund_cloud_job_credits(prompt_id, job_info, error_msg)
            return result
        logger.error(f"☁️ Failed to poll RunPod job {runpod_job_id}: {e}")
        return {
            "prompt_id": prompt_id,
            "status": "running",
            "error": "Cloud job status temporarily unavailable",
            **job_info,
        }

    status_val = (
        rp_job.status.value if hasattr(rp_job.status, "value") else str(rp_job.status)
    )

    # Map RunPod statuses to our status values
    if status_val in ("IN_QUEUE",):
        queue_age = max(
            0.0, time.time() - float(job_info.get("_start_time") or time.time())
        )
        timed_out_result = _mark_cloud_job_timed_out(prompt_id, job_info, queue_age)
        if timed_out_result:
            await _refund_cloud_job_credits(
                prompt_id,
                job_info,
                "RunPod queue timeout — no worker provisioned, credits refunded",
            )
            return timed_out_result
        active_jobs[prompt_id]["_cloud_status"] = status_val
        return {
            "prompt_id": prompt_id,
            "status": "pending",
            "compute_target": "cloud",
            "queue_age_seconds": int(queue_age),
            **job_info,
        }
    elif status_val in ("IN_PROGRESS",):
        active_jobs[prompt_id]["_cloud_status"] = status_val
        # Include progress info from RunPod if available
        progress = {}
        if rp_job.output and isinstance(rp_job.output, dict):
            progress["message"] = rp_job.output.get("message", "Generating...")
        return {
            "prompt_id": prompt_id,
            "status": "running",
            "compute_target": "cloud",
            **progress,
            **job_info,
        }
    elif status_val in ("FAILED", "CANCELLED", "TIMED_OUT"):
        error_msg = rp_job.error or f"RunPod job {status_val}"
        logger.error(f"☁️ Cloud job failed: {runpod_job_id} — {error_msg}")
        _save_cloud_logs(runpod_job_id, prompt_id, rp_job)
        _rp_log_text = (
            (rp_job.output or {}).get("logs")
            if isinstance(rp_job.output, dict)
            else None
        )
        record_generation_complete(
            prompt_id, success=False, error=error_msg, log_text=_rp_log_text
        )
        result = {
            "prompt_id": prompt_id,
            "status": "failed",
            "error": error_msg,
            "compute_target": "cloud",
            **job_info,
        }
        _cloud_completed_cache[prompt_id] = result
        # Clean up active_jobs after a delay (keep for a bit so UI can read the failure)
        active_jobs[prompt_id]["_cloud_completed"] = True
        active_jobs[prompt_id]["_cloud_status"] = status_val
        active_jobs[prompt_id]["_cloud_error"] = error_msg
        _persist_cloud_jobs()
        await _refund_cloud_job_credits(
            prompt_id, job_info, f"RunPod job {status_val} — credits refunded"
        )
        # WS notification for failure
        if ws_manager:
            cloud_user_id = job_info.get("user_id")
            if cloud_user_id:
                await ws_manager.broadcast_to_user(
                    cloud_user_id,
                    "job_failed",
                    {
                        "job_id": prompt_id,
                        "prompt_id": prompt_id,
                        "error": error_msg,
                        "compute_target": "cloud",
                    },
                )
        return result
    elif status_val != "COMPLETED":
        # Unknown status — treat as running
        return {
            "prompt_id": prompt_id,
            "status": "running",
            "compute_target": "cloud",
            **job_info,
        }

    # ── COMPLETED — process output ──────────────────────────────────────
    logger.info(f"☁️ Cloud job COMPLETED: {runpod_job_id}")
    _save_cloud_logs(runpod_job_id, prompt_id, rp_job)

    output = rp_job.output
    _rp_log_text = output.get("logs") if isinstance(output, dict) else None
    if not output or not isinstance(output, dict):
        error_msg = "RunPod returned empty output"
        record_generation_complete(
            prompt_id, success=False, error=error_msg, log_text=_rp_log_text
        )
        result = {
            "prompt_id": prompt_id,
            "status": "failed",
            "error": error_msg,
            "compute_target": "cloud",
            **job_info,
        }
        _cloud_completed_cache[prompt_id] = result
        active_jobs[prompt_id]["_cloud_completed"] = True
        _persist_cloud_jobs()
        return result

    # Check for handler-level errors
    if "error" in output:
        error_msg = output["error"]
        logger.error(f"☁️ Cloud handler error: {error_msg}")
        record_generation_complete(
            prompt_id, success=False, error=error_msg, log_text=_rp_log_text
        )
        result = {
            "prompt_id": prompt_id,
            "status": "failed",
            "error": error_msg,
            "compute_target": "cloud",
            **job_info,
        }
        _cloud_completed_cache[prompt_id] = result
        active_jobs[prompt_id]["_cloud_completed"] = True
        _persist_cloud_jobs()
        return result

    files = output.get("files", [])
    if not files:
        error_msg = "No output files in RunPod response"
        record_generation_complete(
            prompt_id, success=False, error=error_msg, log_text=_rp_log_text
        )
        result = {
            "prompt_id": prompt_id,
            "status": "failed",
            "error": error_msg,
            "compute_target": "cloud",
            **job_info,
        }
        _cloud_completed_cache[prompt_id] = result
        active_jobs[prompt_id]["_cloud_completed"] = True
        _persist_cloud_jobs()
        return result

    # Decode and save all output files (usually 1 video)
    import base64
    from datetime import datetime as _dt

    output_video = None
    output_image = None
    saved_path = None

    for i, f in enumerate(files):
        b64_data = f.get("data")
        if not b64_data:
            continue

        # Generate filename
        timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
        orig_name = _safe_filename(f.get("filename", f"output_{i:03d}.mp4"))
        ext = Path(orig_name).suffix or ".mp4"
        save_name = f"cloud_wan22_{timestamp}_{i:03d}{ext}"

        try:
            file_bytes = base64.b64decode(b64_data)

            # Save to MinIO instead of local disk
            storage = get_storage_client()
            storage.put("generated", f"cloud-wan22/{save_name}", file_bytes)
            logger.info(
                f"☁️ Saved cloud output to storage: generated/cloud-wan22/{save_name} ({len(file_bytes)} bytes)"
            )

            mime = f.get("type", "video/mp4")
            if "video" in mime:
                output_video = f"/media/generated/cloud-wan22/{save_name}"
                saved_path = f"generated/cloud-wan22/{save_name}"
            elif "image" in mime:
                output_image = f"/media/generated/cloud-wan22/{save_name}"
                saved_path = f"generated/cloud-wan22/{save_name}"
        except Exception as e:
            logger.error(f"☁️ Failed to save cloud output file {i}: {e}")

    if not saved_path:
        error_msg = "Failed to save any output files"
        record_generation_complete(prompt_id, success=False, error=error_msg)
        result = {
            "prompt_id": prompt_id,
            "status": "failed",
            "error": error_msg,
            "compute_target": "cloud",
            **job_info,
        }
        _cloud_completed_cache[prompt_id] = result
        active_jobs[prompt_id]["_cloud_completed"] = True
        _persist_cloud_jobs()
        return result

    # Upload to MinIO via MediaService
    storage_path = None
    signed_url = None
    upload_ok = False
    try:
        comfyui = get_comfyui_client()
        if comfyui:
            # Register job metadata so on_job_complete_async can upload to user storage
            cloud_user_id = job_info.get("user_id")
            cloud_prompt = job_info.get("prompt", "")
            if cloud_user_id:
                comfyui.register_job(
                    prompt_id,
                    user_id=cloud_user_id,
                    prompt=cloud_prompt,
                    settings={
                        k: v
                        for k, v in job_info.items()
                        if k not in ("user_id", "prompt")
                    },
                )

            output_type = "video" if output_video else "image"
            storage_path = await comfyui.on_job_complete_async(
                prompt_id, str(saved_path), output_type
            )
            if storage_path:
                logger.info(f"☁️ Auto-uploaded cloud output: {storage_path}")
                signed_url = get_signed_media_url(storage_path, expires_in=86400)
                upload_ok = True
                # Remove staging copy — the file now lives in user storage
                try:
                    storage = get_storage_client()
                    parts = saved_path.replace("\\", "/").split("/", 1)
                    if len(parts) == 2:
                        storage.delete(parts[0], parts[1])
                        logger.info(f"🗑️ Removed staging copy: {saved_path}")
                except Exception as cleanup_err:
                    logger.warning(
                        f"⚠️ Failed to remove staging copy {saved_path}: {cleanup_err}"
                    )
    except Exception as e:
        logger.warning(f"☁️ Storage upload failed — will retry next poll cycle: {e}")

    # If storage upload failed, DON'T mark as completed so poller retries
    if not upload_ok:
        logger.warning(f"☁️ Storage upload incomplete for {prompt_id}, will retry")
        return {
            "prompt_id": prompt_id,
            "status": "running",
            "compute_target": "cloud",
            **job_info,
        }

    # Record stats — fallback execution_time from _start_time if RunPod didn't provide it
    execution_time = output.get("execution_time_s")
    if execution_time is None:
        start_ts = job_info.get("_start_time")
        if start_ts:
            execution_time = round(time.time() - float(start_ts), 1)
    record_generation_complete(prompt_id, success=True, log_text=_rp_log_text)

    # Mark as completed in active_jobs
    active_jobs[prompt_id]["_cloud_completed"] = True
    _persist_cloud_jobs()

    result = {
        "prompt_id": prompt_id,
        "status": "completed",
        "output_video": output_video,
        "output_image": output_image,
        "url": signed_url or output_video or output_image,
        "signed_url": signed_url,
        "storage_path": storage_path,
        "compute_target": "cloud",
        "execution_time_s": execution_time,
        **{k: v for k, v in job_info.items() if not k.startswith("_")},
    }
    _cloud_completed_cache[prompt_id] = result
    logger.info(f"☁️ Cloud job complete: {prompt_id} → {output_video or output_image}")

    # WebSocket notification — push to user immediately
    if ws_manager:
        cloud_user_id = job_info.get("user_id")
        if cloud_user_id:
            await ws_manager.broadcast_to_user(
                cloud_user_id,
                "job_complete",
                {
                    "job_id": prompt_id,
                    "prompt_id": prompt_id,
                    "output_video": output_video,
                    "output_image": output_image,
                    "url": signed_url or output_video or output_image,
                    "signed_url": signed_url,
                    "storage_path": storage_path,
                    "compute_target": "cloud",
                    "execution_time_s": execution_time,
                },
            )
            logger.info(
                f"📡 WS notification sent to {cloud_user_id} for cloud job {prompt_id}"
            )

    return result


@app.get("/api/generation-stats")
async def get_generation_stats(
    limit: int = 100,
    job_type: str = None,
    success_only: bool = False,
):
    """
    Get generation statistics for analysis.

    Parameters:
    - limit: Max number of records to return (default 100)
    - job_type: Filter by job type (wan22_i2v, ltx2_i2v, post_process_*)
    - success_only: Only show successful generations
    """
    stats = load_generation_stats()

    # Apply filters
    if job_type:
        stats = [s for s in stats if s.get("job_type", "").startswith(job_type)]
    if success_only:
        stats = [s for s in stats if s.get("success", False)]

    # Most recent first
    stats = stats[-limit:][::-1]

    # Calculate summary stats
    if stats:
        durations = [s["duration_seconds"] for s in stats if s.get("duration_seconds")]
        successful = sum(1 for s in stats if s.get("success", False))
        summary = {
            "total": len(stats),
            "successful": successful,
            "failed": len(stats) - successful,
            "success_rate": round(successful / len(stats) * 100, 1) if stats else 0,
            "avg_duration": round(sum(durations) / len(durations), 1)
            if durations
            else 0,
            "min_duration": round(min(durations), 1) if durations else 0,
            "max_duration": round(max(durations), 1) if durations else 0,
        }
    else:
        summary = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "success_rate": 0,
            "avg_duration": 0,
        }

    return {"summary": summary, "records": stats}


async def trigger_post_processing_chain(
    prompt_id: str,
    video_path: str,
    post_processing: list,
    user_id: str,
    post_audio_path: str = None,
):
    """
    Trigger chained post-processing jobs after video generation completes.
    Each step queues a new ComfyUI job that triggers the next step on completion.
    """
    if not post_processing:
        return

    comfyui = get_comfyui_client()
    if not comfyui:
        logger.warning("⚠️ ComfyUI not available for post-processing chain")
        return

    current_video = video_path
    chain_id = f"chain_{prompt_id}"

    for idx, step in enumerate(post_processing):
        step_type = step.get("type")
        logger.info(
            f"🔄 Post-processing step {idx + 1}/{len(post_processing)}: {step_type}"
        )

        if step_type == "upscale":
            scale = step.get("scale", 2)
            # Queue upscale workflow
            workflow = comfyui.build_video_upscale_workflow(
                video_path=current_video,
                scale=scale,
                output_prefix=f"{chain_id}_upscale_{idx}",
            )
            if workflow:
                new_prompt_id = comfyui.queue_prompt(workflow)
                if new_prompt_id:
                    # Store remaining chain for this new job
                    remaining_steps = post_processing[idx + 1 :]
                    if remaining_steps:
                        pending_post_processing[new_prompt_id] = {
                            "steps": remaining_steps,
                            "user_id": user_id,
                            "post_audio_path": post_audio_path,
                        }
                    logger.info(f"   📈 Queued upscale job: {new_prompt_id}")
                return  # Chain continues via completion callback

        elif step_type == "interpolate":
            target_fps = step.get("target_fps", 60)
            # Queue RIFE interpolation workflow
            workflow = comfyui.build_rife_workflow(
                video_path=current_video,
                target_fps=target_fps,
                output_prefix=f"{chain_id}_rife_{idx}",
            )
            if workflow:
                new_prompt_id = comfyui.queue_prompt(workflow)
                if new_prompt_id:
                    remaining_steps = post_processing[idx + 1 :]
                    if remaining_steps:
                        pending_post_processing[new_prompt_id] = {
                            "steps": remaining_steps,
                            "user_id": user_id,
                            "post_audio_path": post_audio_path,
                        }
                    logger.info(f"   🔄 Queued RIFE job: {new_prompt_id}")
                return

        elif step_type == "add_audio":
            if post_audio_path and Path(post_audio_path).exists():
                # Use ffmpeg to add audio (simpler than ComfyUI workflow)
                import subprocess

                output_with_audio = current_video.replace(".mp4", "_audio.mp4")
                try:
                    subprocess.run(
                        [
                            "ffmpeg",
                            "-y",
                            "-i",
                            current_video,
                            "-i",
                            post_audio_path,
                            "-c:v",
                            "copy",
                            "-c:a",
                            "aac",
                            "-shortest",
                            output_with_audio,
                        ],
                        check=True,
                        capture_output=True,
                    )
                    logger.info(f"   🔊 Added audio: {output_with_audio}")
                    current_video = output_with_audio
                except Exception as e:
                    logger.error(f"   ❌ Failed to add audio: {e}")

    logger.info(f"✅ Post-processing chain completed for {prompt_id}")


@app.get("/comfyui/queue")
async def get_comfyui_queue(user: Optional[User] = Depends(get_optional_user)):
    """
    Get ComfyUI queue status including running and pending jobs.
    Enriches with Oelala job metadata where available.
    """
    import requests

    empty_queue = {
        "running": [],
        "pending": [],
        "failed": [],
        "training": [],
        "total_running": 0,
        "total_pending": 0,
        "total_failed": 0,
        "total_training": 0,
    }

    if not user:
        return empty_queue

    try:
        resp = requests.get("http://localhost:8188/queue", timeout=5)
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail="ComfyUI not responding")

        data = resp.json()

        # Parse queue data
        running = []
        for item in data.get("queue_running", []):
            if len(item) >= 2:
                prompt_id = item[1]
                active_job = active_jobs.get(prompt_id)
                if not active_job or active_job.get("user_id") != user.id:
                    continue
                job_info = {
                    "prompt_id": prompt_id,
                    "status": "running",
                    "queue_position": 0,
                }
                job_info.update(active_job)
                running.append(job_info)

        pending = []
        for idx, item in enumerate(data.get("queue_pending", [])):
            if len(item) >= 2:
                prompt_id = item[1]
                active_job = active_jobs.get(prompt_id)
                if not active_job or active_job.get("user_id") != user.id:
                    continue
                job_info = {
                    "prompt_id": prompt_id,
                    "status": "pending",
                    "queue_position": idx + 1,
                }
                job_info.update(active_job)
                pending.append(job_info)

        # ── Windows-PC ComfyUI (remote MiniMax-H3) queue ─────────────────
        # H3 local jobs run on 192.168.1.245, NOT on the ai-kvm2 localhost
        # server. Surface them in the queue indicator as running/pending so
        # the user sees remote jobs alongside local/cloud ones.
        windows_running = []
        windows_pending = []
        if get_windows_comfyui_client is not None:
            try:
                wclient = get_windows_comfyui_client()
                if wclient is not None:
                    wq_resp = requests.get(f"{wclient.base_url}/queue", timeout=5)
                    if wq_resp.status_code == 200:
                        wq = wq_resp.json()
                        # Which of this user's active jobs live on the Windows server?
                        windows_local = {
                            pid: info
                            for pid, info in active_jobs.items()
                            if info.get("user_id") == user.id
                            and info.get("adapter_name")
                            in ("minimax-h3-local-t2v", "minimax-h3-local-i2v")
                        }
                        seen = set()
                        for item in wq.get("queue_running", []):
                            if len(item) >= 2:
                                pid = item[1]
                                if pid not in windows_local:
                                    continue
                                seen.add(pid)
                                job_info = {
                                    "prompt_id": pid,
                                    "status": "running",
                                    "queue_position": 0,
                                    "server": "windows",
                                }
                                job_info.update(windows_local[pid])
                                windows_running.append(job_info)
                        for idx, item in enumerate(wq.get("queue_pending", [])):
                            if len(item) >= 2:
                                pid = item[1]
                                if pid not in windows_local:
                                    continue
                                seen.add(pid)
                                job_info = {
                                    "prompt_id": pid,
                                    "status": "pending",
                                    "queue_position": idx + 1,
                                    "server": "windows",
                                }
                                job_info.update(windows_local[pid])
                                windows_pending.append(job_info)
                        # Windows-local jobs tracked locally but no longer on
                        # Windows queue → treat as pending (still to be freed).
                        for pid in windows_local:
                            if pid not in seen:
                                job_info = {
                                    "prompt_id": pid,
                                    "status": "pending",
                                    "queue_position": 0,
                                    "server": "windows",
                                }
                                job_info.update(windows_local[pid])
                                windows_pending.append(job_info)
            except Exception as e:
                logger.warning(f"⚠️ Failed to get Windows ComfyUI queue: {e}")

        # Include face LoRA training jobs in the queue
        training = []
        if face_train_service:
            for job in face_train_service.list_jobs():
                if job.get("user_id") and job.get("user_id") != user.id:
                    continue
                if job.get("status") in ("pending", "running"):
                    progress = 0
                    if job.get("steps_total", 0) > 0:
                        progress = round(
                            (job.get("steps_done", 0) / job["steps_total"]) * 100
                        )
                    training.append(
                        {
                            "prompt_id": f"train_{job['id']}",
                            "job_id": job["id"],
                            "status": job["status"],
                            "job_type": "face_lora_training",
                            "name": job.get("name", ""),
                            "trigger": job.get("trigger", ""),
                            "steps_done": job.get("steps_done", 0),
                            "steps_total": job.get("steps_total", 0),
                            "progress": progress,
                            "images_count": job.get("images_count", 0),
                            "created_at": job.get("created_at"),
                            "started_at": job.get("started_at"),
                        }
                    )

        # Include cloud (RunPod) jobs from active_jobs tracking
        cloud_running = []
        cloud_failed = []
        for pid, info in list(active_jobs.items()):
            if info.get("user_id") != user.id:
                continue
            if info.get("compute_target") != "cloud":
                continue
            cloud_status = info.get("_cloud_status", "IN_QUEUE")
            cloud_job = {
                "prompt_id": pid,
                "compute_target": "cloud",
                "runpod_job_id": info.get("runpod_job_id"),
                "prompt": info.get("prompt", ""),
                "resolution": info.get("resolution", ""),
                "aspect_ratio": info.get("aspect_ratio", ""),
                "num_frames": info.get("num_frames"),
                "model_name": info.get("model_name", "Cloud Wan22"),
                "queue_position": 0,
            }
            # Completed-successfully jobs are hidden from queue
            if info.get("_cloud_completed") and cloud_status not in (
                "FAILED",
                "CANCELLED",
                "TIMED_OUT",
            ):
                continue
            # Failed/cancelled/timed-out → show as failed so user can dismiss
            if cloud_status in ("FAILED", "CANCELLED", "TIMED_OUT") or info.get(
                "_cloud_completed"
            ):
                cloud_job["status"] = "failed"
                cloud_job["error"] = info.get(
                    "_cloud_error", f"RunPod job {cloud_status}"
                )
                cloud_failed.append(cloud_job)
            elif cloud_status in ("IN_QUEUE", "IN_PROGRESS"):
                # Show as pending/running based on local cache. Do NOT timeout here —
                # timeouts only fire when RunPod is actually polled via get_job_status,
                # to avoid killing jobs that are IN_PROGRESS but not yet polled.
                queue_age = max(
                    0.0, time.time() - float(info.get("_start_time") or time.time())
                )
                cloud_job["status"] = (
                    "running" if cloud_status == "IN_PROGRESS" else "pending"
                )
                cloud_job["queue_age_seconds"] = int(queue_age)
                if cloud_status == "IN_PROGRESS":
                    cloud_running.append(cloud_job)
                else:
                    pending.append(cloud_job)
            else:
                # Only real in-progress cloud jobs belong in the running list.
                cloud_job["status"] = "running"
                cloud_running.append(cloud_job)

        all_running = running + cloud_running + windows_running
        all_pending = pending + windows_pending

        return {
            "running": all_running,
            "pending": all_pending,
            "failed": cloud_failed,
            "training": training,
            "total_running": len(all_running),
            "total_pending": len(all_pending),
            "total_failed": len(cloud_failed),
            "total_training": len(training),
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get ComfyUI queue: {e}")
        raise HTTPException(
            status_code=502, detail=f"ComfyUI connection failed: {str(e)}"
        )


@app.get("/comfyui/job/{prompt_id}")
async def get_job_status(
    prompt_id: str,
    user: Optional[User] = Depends(get_optional_user),
):
    """
    Get status of a specific job by prompt_id.
    Returns status (queued/running/completed/failed) and output if available.
    Handles both local ComfyUI jobs and cloud (RunPod) jobs transparently.
    """
    import requests

    safe_prompt_id = _safe_external_id(prompt_id, "prompt ID")

    if not user:
        raise HTTPException(status_code=404, detail="Job not found")

    cached_local = _get_cached_local_result(safe_prompt_id, user.id)
    if cached_local is not None:
        return cached_local

    # Check in our active jobs store
    job_info = active_jobs.get(safe_prompt_id, {})
    if not job_info or job_info.get("user_id") != user.id:
        raise HTTPException(status_code=404, detail="Job not found")

    # ── Cloud (RunPod) job handling ──────────────────────────────────────
    if job_info.get("compute_target") == "cloud" and _runpod:
        return await _handle_cloud_job_status(safe_prompt_id, job_info)

    # ── Local ComfyUI job handling ──────────────────────────────────────
    # Check ComfyUI history for completion status
    try:
        result = await _resolve_local_job_result(safe_prompt_id, job_info)
        if result is not None:
            return result
    except Exception as e:
        logger.warning(f"Error checking history for {safe_prompt_id}: {e}")

    # Check if it's in the queue
    try:
        queue_resp = requests.get("http://localhost:8188/queue", timeout=5)
        if queue_resp.status_code == 200:
            queue_data = queue_resp.json()

            # Check running
            for item in queue_data.get("queue_running", []):
                if len(item) >= 2 and item[1] == safe_prompt_id:
                    return {
                        "prompt_id": safe_prompt_id,
                        "status": "running",
                        **job_info,
                    }

            # Check pending
            for idx, item in enumerate(queue_data.get("queue_pending", [])):
                if len(item) >= 2 and item[1] == safe_prompt_id:
                    return {
                        "prompt_id": safe_prompt_id,
                        "status": "pending",
                        "queue_position": idx + 1,
                        **job_info,
                    }
    except Exception as e:
        logger.warning(f"Error checking queue for {safe_prompt_id}: {e}")

    # Not found anywhere - might have failed or been cancelled
    return {"prompt_id": safe_prompt_id, "status": "unknown", **job_info}


@app.get("/comfyui/output/{filename}")
async def get_comfyui_output(filename: str, request: Request):
    """Serve ComfyUI output files via MinIO proxy."""
    safe_filename = _safe_filename(filename)
    return _storage_proxy_response("comfyui-local", safe_filename, request)


@app.get("/media/generated/cloud-wan22/{filename}")
async def get_cloud_wan22_media(filename: str, request: Request):
    """Serve Cloud Wan22 output files via MinIO proxy."""
    safe_filename = _safe_filename(filename)
    return _storage_proxy_response("generated", f"cloud-wan22/{safe_filename}", request)


# Legacy alias — existing storage data is under cloud-max/
@app.get("/media/generated/cloud-max/{filename}")
async def get_cloud_max_media_legacy(filename: str, request: Request):
    """Backwards-compat: serve old cloud-max output files."""
    safe_filename = _safe_filename(filename)
    return _storage_proxy_response("generated", f"cloud-max/{safe_filename}", request)


@app.get("/loras/download/{filename:path}")
async def download_lora_for_cloud(filename: str, token: str = Query(...)):
    """
    Serve LoRA files for RunPod cloud workers.
    Protected with HMAC-signed token (derived from RUNPOD_API_KEY).
    Files are streamed from /mnt/ssd/loras/.
    Handles filenames with or without .safetensors extension.
    """
    import hmac as _hmac_mod

    safe_filename = _normalize_storage_key(filename)
    expected = _lora_download_token(safe_filename)
    if not _hmac_mod.compare_digest(token, expected):
        logger.warning(f"⚠️ Invalid LoRA download token for: {safe_filename}")
        raise HTTPException(status_code=403, detail="Invalid download token")

    lora_path, resolved_name = _resolve_lora_path(safe_filename)
    if not lora_path:
        raise HTTPException(status_code=404, detail="LoRA not found")

    size_mb = lora_path.stat().st_size / (1024 * 1024)
    logger.info(f"☁️ Serving LoRA for cloud worker: {resolved_name} ({size_mb:.0f}MB)")
    return FileResponse(
        lora_path,
        media_type="application/octet-stream",
        filename=resolved_name,
        headers={
            "Content-Disposition": f'attachment; filename="{resolved_name}"',
            "Cache-Control": "private, no-cache",
        },
    )


@app.get("/media/generated/{filename:path}")
async def get_generated_media(
    filename: str, request: Request, user: User = Depends(get_current_user)
):
    """Serve files from generated bucket via MinIO (authenticated users)."""
    if not await check_admin(user):
        raise HTTPException(status_code=403, detail="Admin access required")

    safe_filename = _normalize_storage_key(filename)
    return _storage_proxy_response(
        "generated",
        safe_filename,
        request,
        cache_control="public, max-age=31536000, immutable",
    )


# =============================================================================
# Unified storage proxy route (Phase 4)
# =============================================================================

ALLOWED_STORAGE_BUCKETS = {"generated", "comfyui-local", "avatars"}


@app.get("/storage/{bucket}/{key:path}")
async def unified_storage_proxy(bucket: str, key: str, request: Request):
    """
    Unified storage proxy endpoint.
    Serves content from MinIO buckets via a single route pattern.
    Only whitelisted public buckets are accessible without auth.
    """
    if bucket not in ALLOWED_STORAGE_BUCKETS:
        raise HTTPException(status_code=404, detail="Bucket not found")

    safe_key = _normalize_storage_key(key)

    return _storage_proxy_response(
        bucket,
        safe_key,
        request,
        cache_control="public, max-age=3600, must-revalidate",
    )


@app.get("/comfyui-metadata/{filename}")
async def get_comfyui_metadata(filename: str, user: User = Depends(get_current_user)):
    """
    Extract and return the ComfyUI workflow/metadata from an output file.
    Works with videos (mp4, webm, mov) and images (png).
    Checks local ComfyUI output first, then fetches from storage.
    """
    import subprocess
    import tempfile

    # Check local ComfyUI output directory first
    output_path = None
    safe_filename = _safe_filename(filename)
    candidate = _safe_child_path(COMFYUI_OUTPUT_DIR, safe_filename)
    if candidate.exists():
        output_path = candidate

    # Fall back to storage (download to temp file for ffprobe)
    tmp_file = None
    if not output_path:
        storage = get_storage_client()
        ext = Path(safe_filename).suffix.lower()
        subfolder = (
            "videos"
            if ext in [".mp4", ".webm", ".mov"]
            else ("audio" if ext in [".wav", ".mp3", ".flac"] else "images")
        )

        # Check user bucket first, then public generated bucket
        sources = [
            ("users", f"{user.id}/{subfolder}/{safe_filename}"),
            ("generated", safe_filename),
        ]

        for bucket, key in sources:
            try:
                data, _, _ = storage.get_with_metadata(bucket, key)
                tmp_file = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                tmp_file.write(data)
                tmp_file.close()
                output_path = Path(tmp_file.name)
                break
            except Exception:
                continue

    if not output_path:
        raise HTTPException(status_code=404, detail="Output file not found")

    ext = output_path.suffix.lower()
    metadata = None

    try:
        if ext in [".mp4", ".webm", ".mov"]:
            # Extract from video metadata using ffprobe
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "quiet",
                    "-print_format",
                    "json",
                    "-show_format",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                probe_data = json.loads(result.stdout)
                comment = (
                    probe_data.get("format", {}).get("tags", {}).get("comment", "")
                )
                if comment and comment.startswith("{"):
                    workflow_data = json.loads(comment)
                    prompt = workflow_data.get("prompt", workflow_data)
                    # Handle double-encoded JSON
                    if isinstance(prompt, str):
                        metadata = json.loads(prompt)
                    else:
                        metadata = prompt

        elif ext in [".png"]:
            # Extract from PNG metadata
            from PIL import Image

            img = Image.open(str(output_path))
            if hasattr(img, "text"):
                if "prompt" in img.text:
                    metadata = json.loads(img.text["prompt"])
                elif "workflow" in img.text:
                    metadata = json.loads(img.text["workflow"])

        if metadata:
            return {"metadata": metadata, "filename": safe_filename}
        else:
            raise HTTPException(status_code=404, detail="No metadata found in file")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to extract metadata from {safe_filename}: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract metadata")
    finally:
        # Clean up temp file if we downloaded from storage
        if tmp_file:
            try:
                Path(tmp_file.name).unlink(missing_ok=True)
            except Exception:
                pass


@app.delete("/comfyui/queue/{prompt_id}")
async def cancel_job(prompt_id: str):
    """Cancel or dismiss a queued, running, or failed job."""
    import requests

    try:
        # Cloud job — just remove from tracking (dismiss)
        if (
            prompt_id in active_jobs
            and active_jobs[prompt_id].get("compute_target") == "cloud"
        ):
            del active_jobs[prompt_id]
            _persist_cloud_jobs()
            logger.info(f"☁️ Dismissed cloud job: {prompt_id}")
            return {"success": True, "prompt_id": prompt_id}

        # Local ComfyUI job — interrupt + remove from queue
        resp = requests.post(
            "http://localhost:8188/interrupt", json={"prompt_id": prompt_id}, timeout=5
        )
        resp.raise_for_status()

        requests.post(
            "http://localhost:8188/queue", json={"delete": [prompt_id]}, timeout=5
        )

        if prompt_id in active_jobs:
            del active_jobs[prompt_id]

        return {"success": True, "prompt_id": prompt_id}
    except Exception as e:
        logger.error(f"Failed to cancel job {prompt_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to cancel job")


@app.post("/extract-metadata")
async def extract_metadata(file: UploadFile = File(...)):
    """
    Extract workflow/prompt metadata from uploaded PNG/image files.

    Generated images from T2I or I2V have embedded metadata containing:
    - prompt: The positive prompt used
    - negative_prompt: The negative prompt
    - workflow: The ComfyUI workflow used
    - oelala_params: Additional generation parameters

    Returns extracted metadata or empty dict if none found.
    """
    content = await file.read()
    metadata = {}
    try:
        # Try to read PNG metadata
        img = Image.open(io.BytesIO(content))

        # Check for various metadata formats
        if hasattr(img, "info"):
            info = img.info

            # Oelala params (our format)
            if "oelala_params" in info:
                try:
                    params = json.loads(info["oelala_params"])
                    metadata["prompt"] = params.get("prompt", "")
                    metadata["negative_prompt"] = params.get("negative_prompt", "")
                    metadata["workflow"] = params.get("workflow", "")
                    metadata["resolution"] = params.get("resolution", "")
                    metadata["steps"] = params.get("steps", "")
                    metadata["cfg"] = params.get("cfg", "")
                    metadata["seed"] = params.get("seed", "")
                    metadata["source"] = "oelala"
                    # Check for preserved original T2I prompt (longer, more descriptive)
                    if params.get("original_t2i_prompt"):
                        metadata["prompt"] = params["original_t2i_prompt"]
                        metadata["source"] = "oelala_t2i"
                    # Store oelala prompt separately so we can compare later
                    metadata["oelala_prompt"] = params.get("prompt", "")
                except json.JSONDecodeError as exc:
                    logger.debug(
                        "🐛 Failed to decode JSON from PNG 'oelala_params' metadata: %s",
                        exc,
                    )

            # ComfyUI workflow format - extract longer prompts from workflow nodes
            if "prompt" in info:
                try:
                    workflow = json.loads(info["prompt"])
                    # Extract prompt from various node types
                    for node_id, node in workflow.items():
                        if isinstance(node, dict):
                            inputs = node.get("inputs", {})
                            class_type = node.get("class_type", "")

                            # WanVideo text encoder (our I2V workflow) - skip short motion prompts
                            if "positive_prompt" in inputs and isinstance(
                                inputs["positive_prompt"], str
                            ):
                                if len(
                                    inputs["positive_prompt"]
                                ) > 50 and not metadata.get("prompt"):
                                    metadata["prompt"] = inputs["positive_prompt"]
                                    metadata["source"] = "comfyui_wan"
                            if "negative_prompt" in inputs and isinstance(
                                inputs["negative_prompt"], str
                            ):
                                if len(
                                    inputs["negative_prompt"]
                                ) > 10 and not metadata.get("negative_prompt"):
                                    metadata["negative_prompt"] = inputs[
                                        "negative_prompt"
                                    ]

                            # CLIPTextEncode (standard ComfyUI T2I) - prefer longer prompts
                            if "text" in inputs and isinstance(inputs["text"], str):
                                text = inputs["text"]
                                if len(text) > 20:
                                    # Check if it's a positive or negative prompt
                                    if "negative" in class_type.lower():
                                        if not metadata.get("negative_prompt") or len(
                                            text
                                        ) > len(metadata.get("negative_prompt", "")):
                                            metadata["negative_prompt"] = text
                                    else:
                                        # Prefer longer prompts (T2I prompts are usually longer than I2V motion prompts)
                                        current = metadata.get("prompt", "")
                                        if len(text) > len(current):
                                            metadata["prompt"] = text
                                            metadata["source"] = "comfyui"
                except json.JSONDecodeError as exc:
                    logger.debug(
                        "🐛 Failed to decode JSON from PNG 'prompt' metadata: %s", exc
                    )

            # A1111/Invoke AI format (parameters in 'parameters' key)
            if "parameters" in info and not metadata.get("prompt"):
                params_text = info["parameters"]
                # Format: "prompt text\nNegative prompt: negative text\nSteps: X, ..."
                lines = params_text.split("\n")
                if lines:
                    # First line(s) until "Negative prompt:" is the positive prompt
                    positive_lines = []
                    negative_started = False
                    for line in lines:
                        if line.startswith("Negative prompt:"):
                            negative_started = True
                            neg = line.replace("Negative prompt:", "").strip()
                            if neg:
                                metadata["negative_prompt"] = neg
                        elif line.startswith("Steps:"):
                            # Parse generation params
                            parts = line.split(",")
                            for part in parts:
                                if ":" in part:
                                    k, v = part.split(":", 1)
                                    k = k.strip().lower().replace(" ", "_")
                                    v = v.strip()
                                    if k in ["steps", "cfg", "seed", "sampler"]:
                                        metadata[k] = v
                        elif not negative_started:
                            positive_lines.append(line)

                    if positive_lines:
                        metadata["prompt"] = "\n".join(positive_lines).strip()
                    metadata["source"] = "a1111"

        logger.info(
            f"📋 Extracted metadata from {file.filename}: {list(metadata.keys())}"
        )

    except Exception as e:
        logger.warning(f"⚠️ Failed to extract metadata from {file.filename}: {e}")

    return metadata


class ExtractMetadataURLRequest(BaseModel):
    image_url: str


@app.post("/extract-metadata-url")
async def extract_metadata_from_url(request: ExtractMetadataURLRequest):
    """
    Extract workflow/prompt metadata from an image URL.
    Supports ComfyUI output URLs and local backend URLs.
    """
    image_url = _validate_public_image_url(request.image_url)
    metadata = {}

    try:
        # Download image from URL
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, timeout=30.0, follow_redirects=False)
            response.raise_for_status()
            content = response.content
        content_type = response.headers.get("content-type", "").lower()
        if "image/" not in content_type:
            raise HTTPException(status_code=400, detail="URL did not return an image")
        if len(content) > MAX_METADATA_IMAGE_BYTES:
            raise HTTPException(status_code=400, detail="Image is too large")

        # Extract metadata (same logic as file upload)
        img = Image.open(io.BytesIO(content))

        if hasattr(img, "info"):
            info = img.info

            # Oelala params (our format)
            if "oelala_params" in info:
                try:
                    params = json.loads(info["oelala_params"])
                    metadata["positive_prompt"] = params.get("prompt", "")
                    metadata["negative_prompt"] = params.get("negative_prompt", "")
                    metadata["workflow"] = params.get("workflow", "")
                    metadata["source"] = "oelala"
                    # Check for preserved original T2I prompt
                    if params.get("original_t2i_prompt"):
                        metadata["positive_prompt"] = params["original_t2i_prompt"]
                        metadata["source"] = "oelala_t2i"
                except json.JSONDecodeError:
                    # Ignore malformed oelala_params JSON in PNG metadata
                    pass

            # ComfyUI workflow format
            if "prompt" in info and not metadata.get("positive_prompt"):
                try:
                    workflow = json.loads(info["prompt"])
                    for node_id, node in workflow.items():
                        if isinstance(node, dict):
                            inputs = node.get("inputs", {})
                            class_type = node.get("class_type", "")

                            # Look for prompt inputs in various node types
                            # Wan nodes use positive_prompt/negative_prompt
                            if "positive_prompt" in inputs and isinstance(
                                inputs["positive_prompt"], str
                            ):
                                text = inputs["positive_prompt"].strip()
                                if text and len(text) > 5:
                                    current = metadata.get("positive_prompt", "")
                                    if len(text) > len(current):
                                        metadata["positive_prompt"] = text
                                        metadata["source"] = "comfyui_wan"

                            if "negative_prompt" in inputs and isinstance(
                                inputs["negative_prompt"], str
                            ):
                                text = inputs["negative_prompt"].strip()
                                if text and len(text) > 3:
                                    metadata["negative_prompt"] = text

                            # CLIP/text nodes use 'text' key
                            if "text" in inputs and isinstance(inputs["text"], str):
                                text = inputs["text"].strip()
                                if text and len(text) > 10:
                                    if "negative" in class_type.lower():
                                        if not metadata.get("negative_prompt") or len(
                                            text
                                        ) > len(metadata.get("negative_prompt", "")):
                                            metadata["negative_prompt"] = text
                                    else:
                                        current = metadata.get("positive_prompt", "")
                                        if len(text) > len(current):
                                            metadata["positive_prompt"] = text
                                            metadata["source"] = "comfyui"
                except json.JSONDecodeError as exc:
                    # 🐛 Ignore malformed JSON in embedded prompt metadata, but log for debugging
                    logger.debug(
                        "Failed to decode JSON from PNG 'prompt' metadata, skipping: %s",
                        exc,
                    )

            # A1111 format
            if "parameters" in info and not metadata.get("positive_prompt"):
                params_text = info["parameters"]
                lines = params_text.split("\n")
                positive_lines = []
                for line in lines:
                    if line.startswith("Negative prompt:"):
                        metadata["negative_prompt"] = line.replace(
                            "Negative prompt:", ""
                        ).strip()
                        break
                    elif not line.startswith("Steps:"):
                        positive_lines.append(line)
                if positive_lines:
                    metadata["positive_prompt"] = "\n".join(positive_lines).strip()
                    metadata["source"] = "a1111"

        logger.info(f"📋 Extracted metadata from URL: {list(metadata.keys())}")

    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"⚠️ Failed to extract metadata from URL: {e}")
        metadata["error"] = "Failed to extract metadata from URL"

    return metadata


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    # Check ComfyUI availability (our primary backend now)
    comfyui_available = False
    if get_comfyui_client:
        try:
            client = get_comfyui_client()
            comfyui_available = client.is_available() if client else False
        except Exception:
            # ComfyUI client not available or failed to connect
            pass

    # We're healthy if ComfyUI is available
    is_healthy = comfyui_available

    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "comfyui_available": comfyui_available,
        "runpod_available": bool(_runpod and _runpod.is_available()),
        "upload_dir": str(UPLOAD_DIR),
        "output_dir": str(OUTPUT_DIR),
    }


@app.get("/health/deep")
async def deep_health_check():
    """Extended health check — tests ComfyUI, storage, and Supabase connectivity."""
    checks = {}

    # ComfyUI
    try:
        if get_comfyui_client:
            client = get_comfyui_client()
            checks["comfyui"] = {"ok": client.is_available() if client else False}
        else:
            checks["comfyui"] = {"ok": False, "error": "client not loaded"}
    except Exception as e:
        logger.warning(f"ComfyUI health check failed: {e}")
        checks["comfyui"] = {"ok": False, "error": "ComfyUI health check failed"}

    # MinIO storage
    try:
        from storage_client import get_client as _get_sc

        _sc = _get_sc()
        _sh = _sc.health()
        checks["storage"] = {"ok": _sh.get("status") == "healthy", "backend": "minio"}
    except Exception as e:
        logger.warning(f"MinIO health check failed: {e}")
        checks["storage"] = {"ok": False, "error": "MinIO health check failed"}

    # Supabase
    try:
        import httpx as _hx

        _sb_url = os.getenv("SUPABASE_URL", "")
        _sb_key = os.getenv("SUPABASE_ANON_KEY", "")
        if _sb_url and _sb_key:
            async with _hx.AsyncClient(timeout=3) as hc:
                r = await hc.get(
                    f"{_sb_url}/rest/v1/",
                    headers={"apikey": _sb_key, "Authorization": f"Bearer {_sb_key}"},
                )
                checks["supabase"] = {"ok": r.status_code in (200, 401, 406)}
        else:
            checks["supabase"] = {"ok": False, "error": "not configured"}
    except Exception as e:
        logger.warning(f"Supabase health check failed: {e}")
        checks["supabase"] = {"ok": False, "error": "Supabase health check failed"}

    # Disk space
    try:
        disk = shutil.disk_usage("/home/flip/oelala")
        checks["disk"] = {
            "ok": disk.free > 1_000_000_000,  # >1 GB free
            "total_gb": round(disk.total / 1e9, 1),
            "free_gb": round(disk.free / 1e9, 1),
            "used_pct": round(disk.used / disk.total * 100, 1),
        }
    except Exception as e:
        logger.warning(f"Disk health check failed: {e}")
        checks["disk"] = {"ok": False, "error": "Disk health check failed"}

    all_ok = all(c.get("ok", False) for c in checks.values())
    return {
        "status": "healthy" if all_ok else "degraded",
        "timestamp": datetime.now().isoformat(),
        "checks": checks,
    }


@app.get("/api/admin/metrics")
async def get_metrics(user: User = Depends(get_current_user)):
    """Admin-only request metrics dashboard data."""
    await check_admin(user)

    # Compute per-endpoint latency summaries
    latency_summary = {}
    for path, values in _request_metrics["endpoint_latencies"].items():
        if values:
            vals = sorted(values)
            n = len(vals)
            latency_summary[path] = {
                "count": n,
                "avg_ms": round(sum(vals) / n, 1),
                "p50_ms": round(vals[n // 2], 1),
                "p95_ms": round(vals[int(n * 0.95)], 1)
                if n >= 2
                else round(vals[-1], 1),
                "max_ms": round(vals[-1], 1),
            }

    # Sort by request count descending
    latency_summary = dict(
        sorted(latency_summary.items(), key=lambda kv: kv[1]["count"], reverse=True)
    )

    return {
        "total_requests": _request_metrics["total_requests"],
        "total_errors": _request_metrics["total_errors"],
        "error_rate_pct": (
            round(
                _request_metrics["total_errors"]
                / max(_request_metrics["total_requests"], 1)
                * 100,
                2,
            )
        ),
        "status_counts": dict(_request_metrics["status_counts"]),
        "started_at": _request_metrics["started_at"],
        "uptime_seconds": round(
            (
                datetime.now() - datetime.fromisoformat(_request_metrics["started_at"])
            ).total_seconds()
        ),
        "endpoint_latencies": latency_summary,
        "sentry_enabled": bool(_sentry_dsn),
    }


# =============================================================================
# RUNPOD CLOUD GPU ENDPOINTS
# =============================================================================


@app.get("/runpod/status")
async def runpod_status(user: User = Depends(get_current_user)):
    """Check if RunPod cloud GPU is configured and available."""
    if not _runpod:
        return {
            "available": False,
            "reason": "RunPod not configured (RUNPOD_API_KEY missing)",
        }

    try:
        account = await _runpod.get_account_info()
        endpoints = await _runpod.list_endpoints()
        return {
            "available": True,
            "has_endpoint": _runpod.has_endpoint(),
            "default_endpoint_id": _runpod.default_endpoint_id,
            "endpoint_count": len(endpoints),
            "endpoints": [
                {"id": ep.get("id"), "name": ep.get("name"), "gpuIds": ep.get("gpuIds")}
                for ep in endpoints
            ],
            "account": {
                "balance": account.get("clientBalance", 0),
                "spend_limit": account.get("spendLimit", 0),
                "current_spend_per_hr": account.get("currentSpendPerHr", 0),
            },
            "active_jobs": _runpod.get_job_stats(),
        }
    except Exception as e:
        logger.error(f"RunPod status check failed: {e}")
        return {"available": False, "reason": "RunPod status check failed"}


@app.get("/runpod/health")
async def runpod_endpoint_health(user: User = Depends(get_current_user)):
    """Check health of the RunPod serverless endpoint (worker status)."""
    if not _runpod or not _runpod.has_endpoint():
        return {"status": "unavailable", "reason": "No RunPod endpoint configured"}

    try:
        health = await _runpod.get_endpoint_health()
        return {"status": "ok", "endpoint_id": _runpod.default_endpoint_id, **health}
    except Exception as e:
        logger.warning(f"RunPod health check failed: {e}")
        return {"status": "error", "error": "RunPod health check failed"}


@app.get("/runpod/job/{job_id}")
async def runpod_job_status(job_id: str, user: User = Depends(get_current_user)):
    """Get status of a specific RunPod job."""
    if not _runpod:
        raise HTTPException(status_code=503, detail="RunPod not configured")

    try:
        job = await _runpod.get_job_status(job_id)
        return {
            "job_id": job.id,
            "status": job.status.value,
            "output": job.output,
            "error": job.error,
            "execution_time_ms": job.execution_time_ms,
            "created_at": job.created_at,
            "completed_at": job.completed_at,
        }
    except Exception as e:
        logger.error(f"RunPod job status failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to get RunPod job status")


@app.post("/runpod/cancel/{job_id}")
async def runpod_cancel_job(job_id: str, user: User = Depends(get_current_user)):
    """Cancel a running RunPod job."""
    if not _runpod:
        raise HTTPException(status_code=503, detail="RunPod not configured")

    success = await _runpod.cancel_job(job_id)
    return {"success": success, "job_id": job_id}


@app.post("/runpod/submit")
async def runpod_submit_workflow(
    request: Request, user: User = Depends(get_current_user)
):
    """
    Submit a raw ComfyUI workflow to RunPod (admin/advanced use).
    Body: {"workflow": {...}, "images": {"filename": "<base64>"}}
    """
    if not _runpod or not _runpod.has_endpoint():
        raise HTTPException(
            status_code=503,
            detail="RunPod not available or no endpoint deployed",
        )

    body = await request.json()
    workflow = body.get("workflow")
    if not workflow:
        raise HTTPException(status_code=400, detail="workflow field required")

    images = body.get("images", {})
    extra = {}
    if images:
        extra["images"] = images

    try:
        job = await _runpod.submit_workflow(workflow, extra_input=extra or None)
        logger.info(f"☁️ RunPod job submitted: {job.id}")
        return {
            "success": True,
            "job_id": job.id,
            "status": job.status.value,
            "endpoint_id": job.endpoint_id,
        }
    except Exception as e:
        logger.error(f"RunPod submit failed: {e}")
        raise HTTPException(status_code=500, detail="RunPod submit failed")


def _parse_mtime(val) -> float:
    """Convert modified_at (ISO string or datetime) to unix timestamp for sorting."""
    if val is None:
        return 0.0
    if hasattr(val, "timestamp"):
        return val.timestamp()
    if isinstance(val, str) and val:
        try:
            from dateutil.parser import isoparse

            return isoparse(val).timestamp()
        except Exception:
            pass
    return 0.0


@app.get("/api/media/unified")
async def list_unified_media(
    type: str = "all",
    source: str = "all",
    # Admin-only filters
    filter_user_id: Optional[str] = None,  # Admin: filter by specific user
    include_all_users: bool = False,  # Admin: show media from all users
    user: User = Depends(get_current_user),
):
    """
    Unified media endpoint that aggregates all media sources.

    Args:
        type: Filter by media type ('all', 'video', 'image', 'audio')
        source: Filter by source ('all', 'user', 'generated', 'comfyui-local', 'public')
               - 'user': User's private storage (always visible to owner)
               - 'generated': Generated media (admin only)
               - 'comfyui-local': ComfyUI output (admin only)
               - 'public': Published gallery items (everyone)
        filter_user_id: (Admin only) Filter to show media from specific user
        include_all_users: (Admin only) Include media from all users

    Returns:
        Combined list of media from all accessible sources
        For admin: includes owner_id, owner_email, visibility fields
    """
    try:
        storage = get_storage_client()
        all_media = []

        # Check admin status for accessing generated/comfyui-local
        is_admin = await check_admin(user)

        # Determine which user's media to fetch
        # Admin can view other users' media
        target_user_id = user.id
        if is_admin:
            if filter_user_id:
                target_user_id = filter_user_id
            elif include_all_users:
                target_user_id = None  # Signal to fetch all users

        # User's own media (or filtered user for admin)
        if source in ("all", "user"):
            try:
                media_type = None
                if type != "all":
                    type_map = {"video": "videos", "image": "images", "audio": "audio"}
                    media_type = type_map.get(type, type)

                if target_user_id:
                    # Fetch specific user's media
                    objects = storage.list_user_media(target_user_id, media_type)
                    for obj in objects:
                        key = obj.get("key", "")
                        filename = obj.get(
                            "filename", key.split("/")[-1] if "/" in key else key
                        )
                        obj_type = obj.get("media_type", "")

                        if obj_type == "videos" or obj.get(
                            "content_type", ""
                        ).startswith("video/"):
                            item_type = "video"
                        elif obj_type == "audio" or obj.get(
                            "content_type", ""
                        ).startswith("audio/"):
                            item_type = "audio"
                        else:
                            item_type = "image"

                        item = {
                            "name": filename,
                            "filename": filename,
                            "type": item_type,
                            "url": f"/user/media/{obj_type}/{filename}",
                            "size": obj.get("size", 0),
                            "modified": obj.get("modified_at", ""),
                            "mtime": _parse_mtime(obj.get("modified_at")),
                            "source": "user",
                            "visibility": "private",  # User storage = private by default
                        }
                        # Admin gets extra info
                        if is_admin:
                            item["owner_id"] = target_user_id
                        all_media.append(item)
                elif is_admin and include_all_users:
                    # Admin: list all users' media
                    # List buckets that start with "users/"
                    try:
                        # Discover user IDs from storage
                        all_user_objects = storage.list("users")
                        user_ids: set[str] = set()
                        for obj in all_user_objects:
                            key = obj.get("key", "")
                            if "/" in key:
                                user_ids.add(key.split("/")[0])
                        for uid in user_ids:
                            try:
                                objects = storage.list_user_media(uid, media_type)
                                for obj in objects:
                                    key = obj.get("key", "")
                                    filename = obj.get(
                                        "filename",
                                        key.split("/")[-1] if "/" in key else key,
                                    )
                                    obj_type = obj.get("media_type", "")

                                    if obj_type == "videos" or obj.get(
                                        "content_type", ""
                                    ).startswith("video/"):
                                        item_type = "video"
                                    elif obj_type == "audio" or obj.get(
                                        "content_type", ""
                                    ).startswith("audio/"):
                                        item_type = "audio"
                                    else:
                                        item_type = "image"

                                    all_media.append(
                                        {
                                            "name": filename,
                                            "filename": filename,
                                            "type": item_type,
                                            "url": f"/admin/user-media/{uid}/{obj_type}/{filename}",
                                            "size": obj.get("size", 0),
                                            "modified": obj.get("modified_at", ""),
                                            "mtime": _parse_mtime(
                                                obj.get("modified_at")
                                            ),
                                            "source": "user",
                                            "visibility": "private",
                                            "owner_id": uid,
                                        }
                                    )
                            except Exception:
                                pass  # Skip users with no media
                    except Exception as e:
                        logger.debug(f"Error listing all users: {e}")
            except Exception as e:
                logger.debug(f"User media not found: {e}")

        # Generated media (admin only)
        if is_admin and source in ("all", "generated"):
            try:
                objects = storage.list("generated")
                for obj in objects:
                    key = obj.get("key", "")
                    if not key or key == ".":
                        continue
                    # Skip user generation artifacts that leaked into generated bucket
                    if key.startswith("users/"):
                        continue
                    # Keep full key as filename for proper deletion
                    filename = key

                    # Determine type from extension — skip non-media files
                    ext = filename.lower().split(".")[-1] if "." in filename else ""
                    if ext in ("mp4", "webm", "mov", "avi"):
                        item_type = "video"
                    elif ext in ("mp3", "wav", "flac", "ogg"):
                        item_type = "audio"
                    elif ext in ("png", "jpg", "jpeg", "webp", "gif", "bmp", "tiff"):
                        item_type = "image"
                    else:
                        continue  # Skip non-media files (logs, json, etc.)

                    # Filter by type if specified
                    if type != "all" and item_type != type:
                        continue

                    all_media.append(
                        {
                            "name": key.split("/")[-1]
                            if "/" in key
                            else key,  # Display name
                            "filename": key,  # Full path for deletion
                            "type": item_type,
                            "url": f"/media/generated/{key}",
                            "size": obj.get("size", 0),
                            "modified": obj.get("modified_at", ""),
                            "mtime": _parse_mtime(obj.get("modified_at")),
                            "source": "generated",
                            "visibility": "dev",  # Generated = dev visibility
                        }
                    )
            except Exception as e:
                logger.debug(f"Generated media error: {e}")

        # ComfyUI local output (admin only)
        if is_admin and source in ("all", "comfyui-local"):
            try:
                objects = storage.list("comfyui-local")
                for obj in objects:
                    key = obj.get("key", "")
                    if not key or key == ".":
                        continue
                    filename = key
                    ext = filename.lower().split(".")[-1] if "." in filename else ""

                    if ext in ("mp4", "webm", "mov", "avi"):
                        item_type = "video"
                    elif ext in ("mp3", "wav", "flac", "ogg"):
                        item_type = "audio"
                    elif ext in ("png", "jpg", "jpeg", "webp", "gif"):
                        item_type = "image"
                    else:
                        continue

                    if type != "all" and item_type != type:
                        continue

                    all_media.append(
                        {
                            "name": filename,
                            "filename": filename,
                            "type": item_type,
                            "url": f"/comfyui/output/{filename}",
                            "size": obj.get("size", 0),
                            "modified": obj.get("modified_at", ""),
                            "mtime": _parse_mtime(obj.get("modified_at")),
                            "source": "comfyui-local",
                            "visibility": "dev",
                        }
                    )
            except Exception as e:
                logger.debug(f"ComfyUI media error: {e}")

        # Public gallery items (everyone can see)
        if source in ("all", "public"):
            try:
                # Query published_media from Supabase
                from supabase_utils import get_supabase_client

                supabase = get_supabase_client()
                if supabase:
                    query = supabase.table("published_media").select(
                        "id,user_id,storage_path,title,media_type,is_nsfw,thumbnail_url,created_at"
                    )

                    # Filter by type if specified
                    if type != "all":
                        query = query.eq("media_type", type)

                    # Non-admin users don't see NSFW by default
                    if not is_admin:
                        query = query.eq("is_nsfw", False)

                    response = query.order("created_at", desc=True).limit(100).execute()

                    for item in response.data or []:
                        # Determine URL based on storage_path
                        storage_path = item.get("storage_path", "")
                        if storage_path.startswith("video/"):
                            url = f"/user/media/videos/{storage_path.split('/', 1)[1]}"
                        elif storage_path.startswith("image/"):
                            url = f"/user/media/images/{storage_path.split('/', 1)[1]}"
                        else:
                            url = f"/gallery/media/{item['id']}"

                        all_media.append(
                            {
                                "id": item.get("id"),
                                "name": item.get("title", "Untitled"),
                                "filename": storage_path.split("/")[-1]
                                if "/" in storage_path
                                else storage_path,
                                "type": item.get("media_type", "image"),
                                "url": url,
                                "thumbnail_url": item.get("thumbnail_url"),
                                "source": "public",
                                "visibility": "public",
                                "is_nsfw": item.get("is_nsfw", False),
                                "owner_id": item.get("user_id"),
                                "mtime": _parse_mtime(
                                    item.get("created_at") or item.get("updated_at")
                                ),
                            }
                        )
            except Exception as e:
                logger.debug(f"Public gallery error: {e}")

        # Sort by mtime (newest first)
        all_media.sort(key=lambda x: x.get("mtime", 0), reverse=True)

        # Enrich items with generation time from lookup
        try:
            from websocket_handler import load_generation_times

            gen_times = load_generation_times()
            if gen_times:
                for item in all_media:
                    fname = item.get("filename", "")
                    # Also check just the basename (filename might have path prefix)
                    basename = fname.split("/")[-1] if "/" in fname else fname
                    gt = gen_times.get(basename) or gen_times.get(fname)
                    if gt:
                        item["generation_time"] = gt
        except Exception as e:
            logger.debug(f"Generation times enrichment skipped: {e}")

        # Deduplicate: if same file exists in both 'user' and 'generated' sources,
        # keep only the 'user' version (generated is dev/admin archive, user is canonical)
        if is_admin:
            # Build set of basenames+sizes from user source for fast lookup
            user_files = set()
            for m in all_media:
                if m.get("source") == "user":
                    basename = (
                        m.get("filename", "").split("/")[-1]
                        if "/" in m.get("filename", "")
                        else m.get("filename", "")
                    )
                    # Strip timestamp prefix (e.g. "20260408_151521_cloud_max..." -> "cloud_max...")
                    # User storage prepends "YYYYMMDD_HHMMSS_" prefix
                    user_files.add((basename, m.get("size", 0)))
                    # Also add without the timestamp prefix for matching
                    parts = basename.split("_", 2)
                    if len(parts) >= 3 and len(parts[0]) == 8 and len(parts[1]) == 6:
                        user_files.add((parts[2], m.get("size", 0)))

            # Filter out generated items that exist in user storage (same base name + size)
            before_count = len(all_media)
            all_media = [
                m
                for m in all_media
                if m.get("source") != "generated"
                or (
                    m.get("filename", "").split("/")[-1]
                    if "/" in m.get("filename", "")
                    else m.get("filename", ""),
                    m.get("size", 0),
                )
                not in user_files
            ]
            deduped = before_count - len(all_media)
            if deduped:
                logger.debug(
                    f"🔄 Deduped {deduped} generated items already in user storage"
                )

        # Count stats by source for admin
        stats = {
            "videos": sum(1 for m in all_media if m["type"] == "video"),
            "images": sum(1 for m in all_media if m["type"] == "image"),
            "audio": sum(1 for m in all_media if m["type"] == "audio"),
        }

        # Add source breakdown for admin
        source_stats = {}
        if is_admin:
            for m in all_media:
                src = m.get("source", "unknown")
                source_stats[src] = source_stats.get(src, 0) + 1

        return {
            "media": all_media,
            "stats": stats,
            "source_stats": source_stats if is_admin else {},
            "is_admin": is_admin,
            "total": len(all_media),
        }

    except Exception as e:
        logger.error(f"Failed to list unified media: {e}")
        raise HTTPException(status_code=500, detail="Failed to list media")


@app.get("/user/media")
async def list_user_media(type: str = "all", user: User = Depends(get_current_user)):
    """
    List media files for the authenticated user from MinIO.

    Args:
        type: Filter by media type ('all', 'images', 'videos', 'audio')

    Returns:
        List of user's media with metadata
    """
    try:
        storage = get_storage_client()

        # Map frontend types to storage types
        media_type = None
        if type != "all":
            type_map = {
                "video": "videos",
                "image": "images",
                "audio": "audio",
                "uploads": "uploads",
            }
            media_type = type_map.get(type, type)

        objects = storage.list_user_media(user.id, media_type)

        # Transform to match existing frontend format
        media = []
        for obj in objects:
            key = obj.get("key", "")
            filename = obj.get("filename", key.split("/")[-1] if "/" in key else key)
            obj_type = obj.get("media_type", "")

            # Determine type from media_type or content_type
            if obj_type == "videos" or (
                obj.get("content_type", "").startswith("video/")
            ):
                item_type = "video"
            elif obj_type == "audio" or (
                obj.get("content_type", "").startswith("audio/")
            ):
                item_type = "audio"
            else:
                item_type = "image"

            media.append(
                {
                    "name": filename,
                    "type": item_type,
                    "url": f"/user/media/{obj_type}/{filename}",
                    "size": obj.get("size", 0),
                    "modified": obj.get("modified_at", datetime.now()).isoformat()
                    if isinstance(obj.get("modified_at"), datetime)
                    else obj.get("modified_at", ""),
                    "mtime": _parse_mtime(obj.get("modified_at")),
                    "hash": obj.get("hash", ""),
                }
            )

        # Sort by modified (newest first)
        media.sort(key=lambda x: x.get("mtime", 0), reverse=True)

        # Count stats
        stats = {
            "videos": sum(1 for m in media if m["type"] == "video"),
            "images": sum(1 for m in media if m["type"] == "image"),
            "audio": sum(1 for m in media if m["type"] == "audio"),
        }

        return {"media": media, "stats": stats}

    except S3Error as e:
        # NoSuchBucket means user has no storage yet — return empty list
        if e.code in ("NoSuchKey", "NoSuchBucket"):
            logger.info(f"User {user.id} has no storage bucket yet")
            return {"media": [], "stats": {"videos": 0, "images": 0, "audio": 0}}
        logger.error(f"Failed to list user media: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user media")
    except Exception as e:
        logger.error(f"Failed to list user media: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve user media")


@app.get("/user/media/{media_type}/{filename:path}")
async def get_user_media(
    media_type: str,
    filename: str,
    request: Request,
    user: User = Depends(get_current_user),
):
    """
    Serve a user's media file from storage.
    Supports HTTP Range requests for video/audio seeking.
    """
    try:
        safe_media_type = _safe_user_media_type(media_type)
        safe_filename = _safe_filename(filename)
        storage = get_storage_client()
        debug_log(
            f"🔍 serving media {safe_media_type}/{safe_filename} for user {user.id}"
        )

        # Determine content type
        ext = Path(safe_filename).suffix.lower()
        content_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".gif": "image/gif",
            ".mp4": "video/mp4",
            ".webm": "video/webm",
            ".mov": "video/quicktime",
            ".wav": "audio/wav",
            ".mp3": "audio/mpeg",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
        }
        content_type = content_types.get(ext, "application/octet-stream")

        bucket = f"users/{user.id}"
        storage_key = f"{safe_media_type}/{safe_filename}"

        headers = {
            "Content-Disposition": f'inline; filename="{safe_filename}"',
            "Accept-Ranges": "bytes",
            "Vary": "Origin",
        }

        origin = request.headers.get("origin")
        if origin and origin in ALLOWED_ORIGINS:
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Credentials"] = "true"

        # Handle Range request — needs stat() for total size
        range_header = request.headers.get("range")
        if range_header and range_header.startswith("bytes="):
            meta = storage.stat(bucket, storage_key)
            if meta is None:
                raise HTTPException(status_code=404, detail="Media not found")

            total_size = meta["size"]
            if meta.get("etag"):
                headers["ETag"] = f'"{meta["etag"]}"'
            if meta.get("last_modified"):
                headers["Last-Modified"] = format_last_modified(meta["last_modified"])

            try:
                parsed = parse_range_header(range_header, total_size)
                if parsed is not None:
                    range_start, range_end = parsed
                    content_length = range_end - range_start + 1
                    content = storage.get_object_range(
                        bucket,
                        storage_key,
                        offset=range_start,
                        length=content_length,
                    )
                    headers["Content-Range"] = (
                        f"bytes {range_start}-{range_end}/{total_size}"
                    )
                    headers["Content-Length"] = str(content_length)
                    return Response(
                        content=content,
                        status_code=206,
                        media_type=content_type,
                        headers=headers,
                    )
            except ValueError:
                # Unsatisfiable range → 416
                headers["Content-Range"] = f"bytes */{total_size}"
                return Response(
                    status_code=416,
                    headers=headers,
                    media_type=content_type,
                )

        # Full response — stream without buffering (skip stat)
        chunks, _ct, total_size, etag, last_modified = storage.stream_with_metadata(
            bucket, storage_key
        )
        if etag:
            headers["ETag"] = etag
        if last_modified:
            headers["Last-Modified"] = last_modified
        if total_size:
            headers["Content-Length"] = str(total_size)
        return StreamingResponse(
            chunks,
            media_type=content_type,
            headers=headers,
        )

    except Exception as e:
        logger.error(f"Failed to get user media: {e}")
        raise HTTPException(status_code=404, detail="Media not found")


@app.get("/user/media/{media_type}/{filename:path}/workflow")
async def get_user_media_workflow(
    media_type: str, filename: str, user: User = Depends(get_current_user)
):
    """
    Extract and return the ComfyUI workflow JSON embedded in a media file.
    ComfyUI stores workflow data in the 'comment' metadata tag for videos
    and in PNG tEXt chunks for images.
    """
    import subprocess
    import tempfile

    try:
        safe_media_type = _safe_user_media_type(media_type)
        safe_filename = _safe_filename(filename)
        storage = get_storage_client()
        data = storage.get_user_media(user.id, safe_media_type, safe_filename)

        ext = Path(safe_filename).suffix.lower()

        # Write to temp file for ffprobe/exiftool analysis
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name

        workflow_json = None

        try:
            if ext in [".mp4", ".webm", ".mov"]:
                # Extract from video metadata using ffprobe
                result = subprocess.run(
                    [
                        "ffprobe",
                        "-v",
                        "quiet",
                        "-print_format",
                        "json",
                        "-show_format",
                        tmp_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    import json as json_module

                    probe_data = json_module.loads(result.stdout)
                    comment = (
                        probe_data.get("format", {}).get("tags", {}).get("comment", "")
                    )
                    if comment and comment.startswith("{"):
                        # ComfyUI stores {"prompt": "..."} where prompt is a JSON string
                        workflow_data = json_module.loads(comment)
                        prompt = workflow_data.get("prompt", workflow_data)
                        # Handle double-encoded JSON (prompt might be a string)
                        if isinstance(prompt, str):
                            workflow_json = json_module.loads(prompt)
                        else:
                            workflow_json = prompt

            elif ext in [".png"]:
                # Extract from PNG metadata
                from PIL import Image

                img = Image.open(tmp_path)
                if hasattr(img, "text"):
                    import json as json_module

                    # ComfyUI stores in 'prompt' or 'workflow' text chunk
                    if "prompt" in img.text:
                        workflow_json = json_module.loads(img.text["prompt"])
                    elif "workflow" in img.text:
                        workflow_json = json_module.loads(img.text["workflow"])
        finally:
            import os

            os.unlink(tmp_path)

        if workflow_json:
            return {"workflow": workflow_json}
        else:
            raise HTTPException(
                status_code=404, detail="No workflow found in media file"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to extract workflow: {e}")
        raise HTTPException(status_code=500, detail="Failed to extract workflow")


class MediaMoveRequest(BaseModel):
    media_type: str
    src_filename: str
    dest_filename: str


@app.post("/api/media/move")
async def move_media(req: MediaMoveRequest, user: User = Depends(get_current_user)):
    """Move/rename a user's media file."""
    try:
        storage = get_storage_client()
        success = storage.move_user_media(
            user.id,
            _safe_user_media_type(req.media_type),
            _safe_filename(req.src_filename),
            _safe_filename(req.dest_filename),
        )
        if not success:
            raise HTTPException(
                status_code=404, detail="Source file not found or move failed"
            )
        return {"success": True, "message": "Moved successfully"}
    except Exception as e:
        logger.error(f"Error moving media: {e}")
        raise HTTPException(status_code=500, detail="Failed to move media")


@app.post("/api/media/batch-download-zip")
async def batch_download_zip(
    request: Request,
    user: User = Depends(get_current_user),
):
    """
    Create a ZIP archive of selected media files for batch download.
    Accepts JSON body: { "items": [{ "url": string, "filename": string }] }
    Supports: /user/media/<type>/<path>, /media/generated/<fn>, /comfyui/output/<fn>
    Max 50 items per request.
    """
    import zipfile

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    items = body.get("items", [])

    if not items:
        raise HTTPException(status_code=400, detail="No items specified")

    if len(items) > 50:
        raise HTTPException(
            status_code=400, detail="Maximum 50 items per batch download"
        )

    debug_log(f"Batch ZIP download: {len(items)} items for user {user.id}")

    zip_buffer = io.BytesIO()
    seen_names: set = set()
    added = 0

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in items:
            url = item.get("url", "")
            filename = item.get("filename", "file")

            # Deduplicate filenames
            base_fn = filename
            counter = 1
            while filename in seen_names:
                parts = base_fn.rsplit(".", 1)
                filename = (
                    f"{parts[0]}_{counter}.{parts[1]}"
                    if len(parts) == 2
                    else f"{base_fn}_{counter}"
                )
                counter += 1
            seen_names.add(filename)

            try:
                # User storage items: /user/media/<type>/<filename>
                if url.startswith("/user/media/"):
                    parts = url.split("/", 4)  # ['', 'user', 'media', <type>, <path>]
                    if len(parts) == 5:
                        media_type, filepath = parts[3], parts[4]
                        storage = get_storage_client()
                        data = storage.get_user_media(user.id, media_type, filepath)
                        zf.writestr(filename, data)
                        added += 1

                # Generated media: /media/generated/<filename>
                elif url.startswith("/media/generated/"):
                    fn = url.split("/media/generated/", 1)[1]
                    storage = get_storage_client()
                    try:
                        data = storage.get("generated", fn)
                        zf.writestr(filename, data)
                        added += 1
                    except Exception:
                        logger.warning(f"⚠️ Not found in storage: generated/{fn}")

                # ComfyUI output: /comfyui/output/<filename>
                elif url.startswith("/comfyui/output/"):
                    fn = url.split("/comfyui/output/", 1)[1]
                    storage = get_storage_client()
                    try:
                        data = storage.get("comfyui-local", fn)
                        zf.writestr(filename, data)
                        added += 1
                    except Exception:
                        logger.warning(f"⚠️ Not found in storage: comfyui-local/{fn}")

                # Public gallery item: /api/gallery/<id>/file
                elif "/api/gallery/" in url and url.endswith("/file"):
                    from supabase_utils import get_supabase_client as _get_sb

                    media_id = url.split("/api/gallery/", 1)[1].replace("/file", "")
                    sb = _get_sb()
                    if sb:
                        row = (
                            sb.table("published_media")
                            .select("user_id,storage_path")
                            .eq("id", media_id)
                            .execute()
                        )
                        if row.data:
                            pub = row.data[0]
                            pub_parts = pub["storage_path"].split("/", 1)
                            if len(pub_parts) == 2:
                                pub_type, pub_filename = pub_parts
                                storage = get_storage_client()
                                data = storage.get_user_media(
                                    pub["user_id"], pub_type, pub_filename
                                )
                                zf.writestr(filename or pub_filename, data)
                                added += 1

                else:
                    logger.warning(f"⚠️ Unrecognized URL for batch ZIP: {url}")

            except Exception as e:
                logger.warning(f"⚠️ Failed to add {filename} to ZIP: {e}")
                continue

    if added == 0:
        raise HTTPException(status_code=400, detail="No files could be added to ZIP")

    zip_buffer.seek(0)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="oelala_selection_{timestamp}.zip"'
        },
    )


@app.post("/user/media/{media_type}")
async def upload_user_media(
    media_type: str,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    """
    Upload media to user's storage.

    Args:
        media_type: 'images', 'videos', or 'audio'
        file: The file to upload
    """
    if media_type not in ("images", "videos", "audio", "uploads"):
        raise HTTPException(status_code=400, detail="Invalid media type")

    try:
        storage = get_storage_client()
        data = await file.read()

        # Generate unique filename if needed
        filename = _safe_filename(
            file.filename or f"{uuid.uuid4()}{Path(file.filename or '').suffix}"
        )
        safe_media_type = _safe_user_media_type(media_type)

        result = storage.put_user_media(
            user.id, safe_media_type, filename, data, file.content_type
        )

        return {
            "success": True,
            "filename": filename,
            "url": f"/user/media/{safe_media_type}/{filename}",
            "size": len(data),
            **result,
        }

    except Exception as e:
        logger.error(f"Failed to upload user media: {e}")
        raise HTTPException(status_code=500, detail="Failed to upload media")


@app.delete("/user/media/{media_type}/{filename:path}")
async def delete_user_media(
    media_type: str, filename: str, user: User = Depends(get_current_user)
):
    """
    Delete a user's media file.
    """
    try:
        safe_media_type = _safe_user_media_type(media_type)
        safe_filename = _safe_filename(filename)
        storage = get_storage_client()
        success = storage.delete_user_media(user.id, safe_media_type, safe_filename)

        if not success:
            raise HTTPException(status_code=404, detail="Media not found")

        return {"success": True, "deleted": safe_filename}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user media: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete media")


@app.get("/user/profile")
async def get_user_profile(user: User = Depends(get_current_user)):
    """
    Get authenticated user's profile info.
    """
    return {
        "id": user.id,
        "email": user.email,
        "role": user.role,
        "metadata": user.user_metadata,
    }


@app.get("/user/storage-quota")
async def get_user_storage_quota(user: User = Depends(get_current_user)):
    """
    Get storage quota information for the authenticated user.

    Returns quota usage, limits, and warnings.
    """
    try:
        media_service = get_media_service()
        quota_info = await media_service.get_user_quota(user.id)

        return {
            "success": True,
            "data": quota_info,
        }
    except Exception as e:
        logger.error(f"❌ Failed to get storage quota: {e}")
        return {
            "success": False,
            "error": "Failed to get storage quota",
            "data": {
                "used_bytes": 0,
                "quota_bytes": 0,
                "used_percent": 0,
                "warning": False,
                "upgrade_needed": False,
            },
        }


# =============================================================================
# WORKFLOW PRESETS API
# =============================================================================


@app.get("/api/presets")
async def get_presets(category: str = None):
    """Get available workflow presets from registry.json

    Args:
        category: Optional filter by category (ImageToVideo, TextToImage, etc.)

    Returns:
        List of presets with their parameters
    """
    registry_path = Path("/home/flip/oelala/workflows/registry.json")

    if not registry_path.exists():
        logger.warning("Workflow registry not found")
        return {"presets": [], "error": "Registry not found"}

    try:
        with open(registry_path, "r") as f:
            registry = json.load(f)

        presets = []
        for workflow_id, workflow in registry.get("workflows", {}).items():
            # Skip if category filter doesn't match
            if category and workflow.get("category") != category:
                continue

            preset = {
                "id": workflow_id,
                "name": workflow.get("name", workflow_id),
                "file": workflow.get("file"),
                "category": workflow.get("category", "Unknown"),
                "description": workflow.get("description", ""),
                "parameters": workflow.get("parameters", {}),
            }
            presets.append(preset)

        # Sort by category, then name
        presets.sort(key=lambda p: (p["category"], p["name"]))

        return {
            "presets": presets,
            "total": len(presets),
            "categories": list(set(p["category"] for p in presets)),
        }

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse registry.json: {e}")
        return {"presets": [], "error": "Invalid workflow registry"}
    except Exception as e:
        logger.error(f"Error loading presets: {e}")
        return {"presets": [], "error": "Failed to load presets"}


@app.get("/api/presets/{preset_id}")
async def get_preset(preset_id: str):
    """Get a specific preset by ID"""
    registry_path = Path("/home/flip/oelala/workflows/registry.json")

    if not registry_path.exists():
        raise HTTPException(status_code=404, detail="Registry not found")

    try:
        with open(registry_path, "r") as f:
            registry = json.load(f)

        workflow = registry.get("workflows", {}).get(preset_id)
        if not workflow:
            raise HTTPException(
                status_code=404, detail=f"Preset '{preset_id}' not found"
            )

        return {
            "id": preset_id,
            "name": workflow.get("name", preset_id),
            "file": workflow.get("file"),
            "category": workflow.get("category", "Unknown"),
            "description": workflow.get("description", ""),
            "parameters": workflow.get("parameters", {}),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error loading preset {preset_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to load preset")


@app.post("/restart")
async def restart_backend():
    """Restart the backend server (uvicorn --reload will handle this)"""
    import signal
    import os

    logger.info("🔄 Backend restart requested via API")

    # Send SIGHUP to trigger uvicorn reload
    def delayed_restart():
        import time

        time.sleep(0.5)
        os.kill(os.getpid(), signal.SIGHUP)

    threading.Thread(target=delayed_restart, daemon=True).start()

    return {"status": "restarting", "message": "Backend will restart shortly"}


@app.get("/files/{filename}")
async def get_file(filename: str, request: Request):
    """Serve generated files via MinIO proxy, with local fallback."""
    safe_filename = _safe_filename(filename)
    # Try storage first (new path)
    try:
        return _storage_proxy_response("generated", safe_filename, request)
    except HTTPException:
        pass

    # Fallback to local OUTPUT_DIR for legacy files
    file_path = _safe_child_path(OUTPUT_DIR, safe_filename)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    ext = file_path.suffix.lower()
    media_type = {
        ".mp4": "video/mp4",
        ".gif": "image/gif",
        ".json": "application/json",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".webp": "image/webp",
    }.get(ext, "application/octet-stream")
    return FileResponse(file_path, media_type=media_type, filename=safe_filename)


@app.post("/client-log")
async def client_log(payload: dict):
    """Receive client-side log messages and persist to disk for analysis.

    Expected JSON payload: {"level": "error", "message": "...", "stack": "...", "url": "...", "userAgent": "...", "timestamp": "...", "meta": {...}}
    """
    try:
        logs_dir = Path("/home/flip/oelala/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / "ui_client.log"

        entry = {"received_at": datetime.now().isoformat(), "payload": payload}

        # Append JSON line to log file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        logger.info(
            f"Received client log: {payload.get('level', 'info')} {payload.get('message', '')}"
        )
        return {"success": True}
    except Exception as e:
        logger.error(f"Error saving client log: {e}")
        raise HTTPException(status_code=500, detail="Failed to save client log")


@app.post("/generate-image")
async def generate_image_legacy(
    prompt: str = Form(...),
    aspect_ratio: str = Form("1:1"),
    mode: str = Form("normal"),
    output_filename: str = Form(""),
    job_id: str = Form(None),
    model: str = Form("sdxl"),
    user: User = Depends(get_current_user),  # Require authenticated user
):
    """Legacy endpoint - redirects to SDXL via ComfyUI (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            mode=mode,
            output_filename=output_filename,
            job_id=job_id,
            model=model,
        ),
        files={},
        operation=Operation.GENERATE,
        target_type=MediaType.IMAGE,
        adapter_hint="sdxl-local-t2i",
        user=user,
        register_job_settings={
            "job_type": "t2i",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# SDXL Text-to-Image via ComfyUI
# ─────────────────────────────────────────────────────────────────────────────

# Available SDXL checkpoints (auto-detected from ComfyUI models folder)


@app.get("/api/models/checkpoints")
def list_comfyui_checkpoints():
    """List available checkpoints dynamically from ComfyUI"""
    import requests

    client = get_comfyui_client()
    try:
        resp = requests.get(
            f"{client.base_url}/object_info/CheckpointLoaderSimple", timeout=5
        )
        data = resp.json()
        models = (
            data.get("CheckpointLoaderSimple", {})
            .get("input", {})
            .get("required", {})
            .get("ckpt_name", [[]])[0]
        )
        # Filter for safetensors to exclude unet/vae if needed, though CheckpointLoader usually filters by default
        return {"checkpoints": [m for m in models if m.endswith(".safetensors")]}
    except Exception as e:
        logger.error(f"Error fetching checkpoints from ComfyUI: {e}")
        return {
            "checkpoints": [
                "CyberRealistic_Pony_v14.1_FP16.safetensors",
                "ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
                "reapony_v90.safetensors",
                "flux1-dev-fp8.safetensors",
            ]
        }  # Fallback list


# Backwards compatibility
@app.get("/sdxl/checkpoints")
def list_sdxl_checkpoints():
    return list_comfyui_checkpoints()


@app.post("/generate-sdxl")
async def generate_sdxl_image(
    prompt: str = Form(...),
    negative_prompt: str = Form(
        "ugly, deformed, blurry, low quality, bad anatomy, watermark, signature, text"
    ),
    checkpoint: str = Form(None),
    aspect_ratio: str = Form("1:1"),
    steps: int = Form(30),
    cfg: float = Form(7.5),
    seed: int = Form(-1),
    sampler_name: str = Form("dpmpp_2m"),
    scheduler: str = Form("karras"),
    lora_configs: str = Form("[]"),  # JSON string of [{name, strength}]
    user: User = Depends(get_current_user),  # Require authenticated user
):
    """Queue SDXL image generation via ComfyUI (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            checkpoint=checkpoint,
            aspect_ratio=aspect_ratio,
            steps=steps,
            cfg=cfg,
            seed=seed,
            sampler_name=sampler_name,
            scheduler=scheduler,
            lora_configs=lora_configs,
        ),
        files={},
        operation=Operation.GENERATE,
        target_type=MediaType.IMAGE,
        adapter_hint="sdxl-local-t2i",
        user=user,
        register_job_settings={
            "job_type": "t2i",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Flux Text-to-Image via ComfyUI
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/generate-flux")
async def generate_flux_image(
    prompt: str = Form(...),
    aspect_ratio: str = Form("1:1"),
    steps: int = Form(20),
    guidance: float = Form(3.5),
    seed: int = Form(-1),
    lora_configs: str = Form("[]"),  # JSON string of [{name, strength}]
    user: User = Depends(get_current_user),  # Require authenticated user
):
    """Generate image using Flux Dev via ComfyUI (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            prompt=prompt,
            aspect_ratio=aspect_ratio,
            steps=steps,
            guidance=guidance,
            seed=seed,
            lora_configs=lora_configs,
        ),
        files={},
        operation=Operation.GENERATE,
        target_type=MediaType.IMAGE,
        adapter_hint="flux-local-t2i",
        user=user,
        register_job_settings={
            "job_type": "t2i",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Wan2.2 Text-to-Video via ComfyUI (next section)
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/generate")
async def generate_video(
    file: UploadFile = File(...),
    prompt: str = Form(""),
    num_frames: int = Form(41, description="Number of frames in video"),
    output_filename: str = Form("", description="Custom output filename"),
    resolution: str = Form("480p", description="Video resolution: 480p, 720p"),
    fps: int = Form(16, description="Frames per second: 8, 12, 16, 24"),
    aspect_ratio: str = Form("1:1", description="Video aspect ratio"),
    user: User = Depends(get_current_user),  # Require authenticated user
):
    """Generate video from uploaded image via ComfyUI (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            prompt=prompt,
            num_frames=num_frames,
            output_filename=output_filename,
            resolution=resolution,
            fps=fps,
            aspect_ratio=aspect_ratio,
        ),
        files={"file": file},
        operation=Operation.GENERATE,
        target_type=MediaType.VIDEO,
        adapter_hint="wan22-local-i2v-q6",
        user=user,
        register_job_settings={
            "job_type": "i2v",
        },
    )


@app.get("/api/i2v-modes")
async def get_i2v_generation_modes():
    """
    Get available I2V generation modes.
    Each mode has different workflow presets (LoRAs, models, etc.)
    """
    from comfyui_client import get_available_i2v_modes

    return {
        "modes": get_available_i2v_modes(),
        "default": "standard",
    }


@app.get("/api/t2v-modes")
async def get_t2v_generation_modes():
    """
    Get available T2V (Text-to-Video) generation modes.
    Different base models: wan22 (Wan2.2 14B), ltx2 (LTX-2 19B).
    """
    from comfyui_client import get_available_t2v_modes

    return {
        "modes": get_available_t2v_modes(),
        "default": "wan22",
    }


@app.get("/api/v2v-modes")
async def get_v2v_generation_modes():
    """
    Get available V2V (Video-to-Video) style transfer modes.
    Uses I2V pipeline with extracted first frame.
    """
    from comfyui_client import get_available_i2v_modes

    # V2V uses I2V modes under the hood
    i2v_modes = get_available_i2v_modes()

    # Add V2V-specific modes
    v2v_modes = {
        "style_transfer": {
            "name": "Style Transfer",
            "description": "Apply artistic style while preserving motion",
            "strength_range": [0.3, 0.8],
            "default_strength": 0.5,
        },
        "anime": {
            "name": "Anime Conversion",
            "description": "Convert video to anime/cartoon style",
            "strength_range": [0.4, 0.9],
            "default_strength": 0.7,
        },
        "enhance": {
            "name": "AI Enhancement",
            "description": "Enhance quality while preserving content",
            "strength_range": [0.1, 0.4],
            "default_strength": 0.25,
        },
    }

    return {
        "modes": v2v_modes,
        "i2v_modes": i2v_modes,  # Available I2V presets
        "default": "style_transfer",
    }


@app.post("/api/v2v/generate")
async def generate_video_to_video(
    file: UploadFile = File(..., description="Input video file"),
    style_prompt: str = Form(
        ..., description="Style description (e.g., 'anime style, vibrant colors')"
    ),
    mode: str = Form(
        "style_transfer", description="V2V mode: style_transfer, anime, enhance"
    ),
    strength: float = Form(
        0.5, description="Style strength (0.0-1.0, higher = more style change)"
    ),
    num_frames: int = Form(41, description="Output frames (4k+1 format for Wan2.2)"),
    resolution: str = Form("480p", description="Output resolution: 480p, 720p"),
    fps: int = Form(16, description="Output FPS"),
    preserve_motion: bool = Form(True, description="Try to preserve original motion"),
    seed: int = Form(-1, description="Random seed (-1 for random)"),
    generation_mode: str = Form("standard", description="I2V generation mode to use"),
    user: User = Depends(get_current_user),
):
    """Video-to-Video style transfer using AI (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            style_prompt=style_prompt,
            mode=mode,
            strength=strength,
            num_frames=num_frames,
            resolution=resolution,
            fps=fps,
            preserve_motion=preserve_motion,
            seed=seed,
            generation_mode=generation_mode,
        ),
        files={"file": file},
        operation=Operation.TRANSFORM,
        target_type=MediaType.VIDEO,
        adapter_hint="local-v2v",
        user=user,
        register_job_settings={
            "job_type": "v2v",
        },
    )


@app.post("/generate-wan22-comfyui")
async def generate_wan22_comfyui(
    file: UploadFile = File(...),
    prompt: str = Form("Motion, subject moving naturally"),
    num_frames: int = Form(41, description="Number of frames in video"),
    output_filename: str = Form("", description="Custom output filename"),
    resolution: str = Form("480p", description="Video resolution: 480p, 720p, 1080p"),
    fps: int = Form(16, description="Frames per second: 8, 12, 16, 24"),
    aspect_ratio: str = Form("1:1", description="Video aspect ratio"),
    steps: int = Form(6, description="Sampling steps"),
    cfg: float = Form(1.0, description="CFG guidance scale (1.0 for DisTorch2)"),
    seed: int = Form(-1, description="Random seed (-1 for random)"),
    generation_mode: str = Form(
        "standard", description="Generation mode: standard, nsfw_lora"
    ),
    unet_high_noise: str = Form(
        "wan2.2_i2v_high_noise_14B_Q6_K.gguf",
        description="GGUF model for high noise pass",
    ),
    unet_low_noise: str = Form(
        "wan2.2_i2v_low_noise_14B_Q6_K.gguf",
        description="GGUF model for low noise pass",
    ),
    lora_configs: str = Form(
        "", description="JSON array of LoRA configs [{high, low, strength}, ...]"
    ),
    extend_mode: str = Form("false", description="Enable sequential clip extension"),
    clip_count: int = Form(1, description="Number of sequential clips (1-5)"),
    user: User = Depends(get_current_user),  # Require authenticated user
):
    """Generate Wan2.2 I2V video via ComfyUI with DisTorch2 Dual-Pass workflow (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            prompt=prompt,
            num_frames=num_frames,
            output_filename=output_filename,
            resolution=resolution,
            fps=fps,
            aspect_ratio=aspect_ratio,
            steps=steps,
            cfg=cfg,
            seed=seed,
            generation_mode=generation_mode,
            unet_high_noise=unet_high_noise,
            unet_low_noise=unet_low_noise,
            lora_configs=lora_configs,
            extend_mode=extend_mode,
            clip_count=clip_count,
        ),
        files={"file": file},
        operation=Operation.GENERATE,
        target_type=MediaType.VIDEO,
        adapter_hint="wan22-local-i2v-q6",
        user=user,
        register_job_settings={
            "job_type": "i2v",
        },
    )


# -----------------------------------------------------------------------------
# Helper: Submit workflow to RunPod cloud GPU
# -----------------------------------------------------------------------------
async def _submit_to_runpod(
    workflow: dict,
    user_id: str,
    prompt_id: str,
    job_info: dict,
    images: dict = None,
    lora_downloads: list = None,
    prompt_full: str = None,
    input_image_path: str = None,
    endpoint_id: str = None,
) -> dict:
    """Submit a ComfyUI workflow to RunPod cloud GPU instead of local."""
    if not _runpod or not _runpod.has_endpoint():
        raise HTTPException(
            status_code=503,
            detail="RunPod cloud GPU not available. Deploy an endpoint first.",
        )

    extra = {}
    if images:
        extra["images"] = images
    if lora_downloads:
        extra["lora_downloads"] = lora_downloads
        logger.info(f"☁️ Including {len(lora_downloads)} LoRA download(s) in RunPod job")

    # Log workflow settings being sent to RunPod
    _wf_settings = []
    for _nid, _node in workflow.items():
        _ct = _node.get("class_type", "")
        _inp = _node.get("inputs", {})
        if _ct == "KSamplerAdvanced":
            _lbl = "pass1" if _inp.get("add_noise") == "enable" else "pass2"
            _wf_settings.append(
                f"{_lbl}: steps={_inp.get('steps')}, cfg={_inp.get('cfg')}, "
                f"sampler={_inp.get('sampler_name')}, range={_inp.get('start_at_step')}-{_inp.get('end_at_step')}"
            )
        elif _ct in ("WanImageToVideo", "EmptyWanLatentVideo"):
            _wf_settings.append(
                f"video: {_inp.get('width')}x{_inp.get('height')}, {_inp.get('length')}f"
            )
    if _wf_settings:
        logger.info(f"☁️ RunPod workflow settings: {' | '.join(_wf_settings)}")

    job = await _runpod.submit_workflow(
        workflow,
        endpoint_id=endpoint_id,
        extra_input=extra or None,
    )
    logger.info(
        f"☁️ RunPod job submitted: {job.id} (prompt_id={prompt_id}, user={user_id})"
    )

    # Track RunPod job alongside local job tracking
    job_info["compute_target"] = "cloud"
    job_info["runpod_job_id"] = job.id
    job_info["runpod_endpoint_id"] = job.endpoint_id
    active_jobs[prompt_id] = job_info
    record_generation_start(prompt_id, job_info)
    _persist_cloud_jobs()

    # Save generation artifacts to user storage bucket
    _img_path = None
    if job_info.get("input_image"):
        _img_path = _find_existing_media_path(job_info["input_image"])
    elif input_image_path:
        _img_path = _find_existing_media_path(input_image_path)
    save_gen_start_artifacts(
        user_id=user_id,
        prompt_id=prompt_id,
        workflow=workflow,
        prompt=prompt_full or job_info.get("prompt", ""),
        job_info=job_info,
        input_image_path=_img_path,
    )

    return {
        "success": True,
        "prompt_id": prompt_id,
        "runpod_job_id": job.id,
        "status": "queued_cloud",
        "compute_target": "cloud",
        "message": f"Job submitted to RunPod cloud GPU. Poll /runpod/job/{job.id} for status.",
        **{k: v for k, v in job_info.items() if not k.startswith("_")},
    }


@app.post("/generate-wan22-async")
async def generate_wan22_async(
    file: UploadFile = File(...),
    prompt: str = Form("Motion, subject moving naturally"),
    num_frames: int = Form(41, description="Number of frames in video"),
    output_filename: str = Form("", description="Custom output filename"),
    resolution: str = Form(
        "480p", description="Video resolution: 480p, 576p, 720p, 1080p"
    ),
    fps: int = Form(16, description="Frames per second: 8, 12, 16, 24"),
    aspect_ratio: str = Form("1:1", description="Video aspect ratio"),
    steps: int = Form(6, description="Sampling steps"),
    cfg: float = Form(1.0, description="CFG guidance scale (1.0 for DisTorch2)"),
    seed: int = Form(-1, description="Random seed (-1 for random)"),
    unet_high_noise: str = Form(
        "wan2.2_i2v_high_noise_14B_Q6_K.gguf",
        description="GGUF model for high noise pass",
    ),
    unet_low_noise: str = Form(
        "wan2.2_i2v_low_noise_14B_Q6_K.gguf",
        description="GGUF model for low noise pass",
    ),
    lora_configs: str = Form(
        "", description="JSON array of LoRA configs [{high, low, strength}, ...]"
    ),
    extend_mode: str = Form("false", description="Enable sequential clip extension"),
    clip_count: int = Form(1, description="Number of sequential clips (1-5)"),
    post_processing: str = Form(
        "", description="JSON array of post-processing steps [{type, ...}, ...]"
    ),
    post_audio_file: UploadFile = File(
        None, description="Audio file for add_audio post-processing"
    ),
    compute_target: str = Form(
        "local", description="Compute target: 'local' or 'cloud' (RunPod)"
    ),
    user: User = Depends(get_current_user),  # Require authenticated user
):
    """Queue Wan2.2 I2V video generation and return immediately (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            prompt=prompt,
            num_frames=num_frames,
            output_filename=output_filename,
            resolution=resolution,
            fps=fps,
            aspect_ratio=aspect_ratio,
            steps=steps,
            cfg=cfg,
            seed=seed,
            unet_high_noise=unet_high_noise,
            unet_low_noise=unet_low_noise,
            lora_configs=lora_configs,
            extend_mode=extend_mode,
            clip_count=clip_count,
            post_processing=post_processing,
            compute_target=compute_target,
        ),
        files={"file": file, "post_audio_file": post_audio_file},
        operation=Operation.GENERATE,
        target_type=MediaType.VIDEO,
        adapter_hint="wan22-local-i2v-q6",
        user=user,
        register_job_settings={
            "job_type": "i2v",
        },
    )


# =============================================================================
# BlockSwap Q8 Experimental I2V Async Endpoint
# =============================================================================


@app.post("/generate-blockswap-q8-async")
async def generate_blockswap_q8_async(
    file: UploadFile = File(...),
    prompt: str = Form("Motion, subject moving naturally"),
    negative_prompt: str = Form(
        "low quality, blurry, distorted, artifacts",
        description="Negative prompt",
    ),
    num_frames: int = Form(121, description="Number of frames (4k+1 format)"),
    resolution: str = Form("720p", description="Video resolution: 480p, 576p, 720p"),
    fps: int = Form(16, description="Frames per second"),
    aspect_ratio: str = Form("9:16", description="Video aspect ratio"),
    steps: int = Form(8, description="Sampling steps (4-12)"),
    cfg: float = Form(1.0, description="CFG guidance scale"),
    seed: int = Form(-1, description="Random seed (-1 for random)"),
    high_noise_steps: int = Form(4, description="Steps for high noise model"),
    shift: float = Form(9.0, description="ModelSamplingSD3 shift"),
    nag_scale: float = Form(11.0, description="NAG guidance scale"),
    enable_upscale: bool = Form(False, description="Enable 4x upscale"),
    enable_interpolation: bool = Form(
        False, description="Enable RIFE 2x interpolation"
    ),
    enable_florence2: bool = Form(True, description="Enable Florence2 auto-captioning"),
    lora_configs: str = Form(
        "", description="JSON array of LoRA configs [{high, low, strength}, ...]"
    ),
    compute_target: str = Form(
        "local", description="Compute target: 'local' or 'cloud' (RunPod)"
    ),
    user: User = Depends(get_current_user),
):
    """Queue BlockSwap Q8 experimental I2V video generation (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            resolution=resolution,
            fps=fps,
            aspect_ratio=aspect_ratio,
            steps=steps,
            cfg=cfg,
            seed=seed,
            high_noise_steps=high_noise_steps,
            shift=shift,
            nag_scale=nag_scale,
            enable_upscale=enable_upscale,
            enable_interpolation=enable_interpolation,
            enable_florence2=enable_florence2,
            lora_configs=lora_configs,
            compute_target=compute_target,
        ),
        files={"file": file},
        operation=Operation.GENERATE,
        target_type=MediaType.VIDEO,
        adapter_hint="wan22-local-i2v-blockswap",
        user=user,
        register_job_settings={
            "job_type": "i2v",
        },
    )


# =============================================================================
# DisTorch2 Q8 Experimental I2V Async Endpoint
# =============================================================================


@app.post("/generate-distorch2-q8-async")
async def generate_distorch2_q8_async(
    file: UploadFile = File(...),
    prompt: str = Form("Motion, subject moving naturally"),
    negative_prompt: str = Form(
        "low quality, blurry, distorted, artifacts",
        description="Negative prompt",
    ),
    num_frames: int = Form(161, description="Number of frames (4k+1 format)"),
    resolution: str = Form("480p", description="Video resolution: 480p, 576p, 720p"),
    fps: int = Form(16, description="Frames per second"),
    aspect_ratio: str = Form("9:16", description="Video aspect ratio"),
    steps: int = Form(8, description="Sampling steps (4-12)"),
    cfg: float = Form(1.0, description="CFG guidance scale"),
    seed: int = Form(-1, description="Random seed (-1 for random)"),
    high_noise_steps: int = Form(
        4, description="Steps for high noise model (4+4 split tested)"
    ),
    shift: float = Form(9.0, description="ModelSamplingSD3 shift"),
    nag_scale: float = Form(11.0, description="NAG guidance scale"),
    enable_upscale: bool = Form(False, description="Enable 4x upscale"),
    enable_interpolation: bool = Form(
        False, description="Enable RIFE 2x interpolation"
    ),
    enable_florence2: bool = Form(True, description="Enable Florence2 auto-captioning"),
    lora_configs: str = Form(
        "", description="JSON array of LoRA configs [{high, low, strength}, ...]"
    ),
    compute_target: str = Form(
        "local", description="Compute target: 'local' or 'cloud' (RunPod)"
    ),
    user: User = Depends(get_current_user),
):
    """Queue DisTorch2 Q8 experimental I2V video generation (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            resolution=resolution,
            fps=fps,
            aspect_ratio=aspect_ratio,
            steps=steps,
            cfg=cfg,
            seed=seed,
            high_noise_steps=high_noise_steps,
            shift=shift,
            nag_scale=nag_scale,
            enable_upscale=enable_upscale,
            enable_interpolation=enable_interpolation,
            enable_florence2=enable_florence2,
            lora_configs=lora_configs,
            compute_target=compute_target,
        ),
        files={"file": file},
        operation=Operation.GENERATE,
        target_type=MediaType.VIDEO,
        adapter_hint="wan22-local-i2v-distorch2",
        user=user,
        register_job_settings={
            "job_type": "i2v",
        },
    )


# =============================================================================
# Ultra Q8 I2V Async Endpoint — Max VRAM + Unlimited CPU RAM
# =============================================================================


@app.post("/generate-ultra-q8-async")
async def generate_ultra_q8_async(
    file: UploadFile = File(...),
    prompt: str = Form("Motion, subject moving naturally"),
    negative_prompt: str = Form(
        "low quality, blurry, distorted, artifacts",
        description="Negative prompt",
    ),
    num_frames: int = Form(161, description="Number of frames (4k+1 format)"),
    resolution: str = Form("576p", description="Video resolution: 480p, 576p, 720p"),
    fps: int = Form(16, description="Frames per second"),
    aspect_ratio: str = Form("9:16", description="Video aspect ratio"),
    steps: int = Form(8, description="Sampling steps (4-12)"),
    cfg: float = Form(1.0, description="CFG guidance scale"),
    seed: int = Form(-1, description="Random seed (-1 for random)"),
    high_noise_steps: int = Form(4, description="Steps for high noise model"),
    shift: float = Form(9.0, description="ModelSamplingSD3 shift"),
    nag_scale: float = Form(11.0, description="NAG guidance scale"),
    enable_upscale: bool = Form(False, description="Enable 4x upscale"),
    enable_interpolation: bool = Form(
        False, description="Enable RIFE 2x interpolation"
    ),
    enable_florence2: bool = Form(True, description="Enable Florence2 auto-captioning"),
    lora_configs: str = Form(
        "", description="JSON array of LoRA configs [{high, low, strength}, ...]"
    ),
    compute_target: str = Form(
        "local", description="Compute target: 'local' or 'cloud' (RunPod)"
    ),
    user: User = Depends(get_current_user),
):
    """Queue Ultra Q8 I2V video generation — max VRAM + unlimited CPU RAM (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            resolution=resolution,
            fps=fps,
            aspect_ratio=aspect_ratio,
            steps=steps,
            cfg=cfg,
            seed=seed,
            high_noise_steps=high_noise_steps,
            shift=shift,
            nag_scale=nag_scale,
            enable_upscale=enable_upscale,
            enable_interpolation=enable_interpolation,
            enable_florence2=enable_florence2,
            lora_configs=lora_configs,
            compute_target=compute_target,
        ),
        files={"file": file},
        operation=Operation.GENERATE,
        target_type=MediaType.VIDEO,
        adapter_hint="wan22-local-i2v-ultra",
        user=user,
        register_job_settings={
            "job_type": "i2v",
        },
    )


# =============================================================================
# Cloud Wan22 Async Endpoint (RunPod bf16 — cloud-only)
# =============================================================================


@app.post("/generate-cloud-wan22-async")
async def generate_cloud_wan22_async(
    file: UploadFile = File(None),
    prompt: str = Form("Motion, subject moving naturally, cinematic quality"),
    negative_prompt: str = Form(
        "low quality, blurry, distorted, artifacts, flickering, jitter",
        description="Negative prompt",
    ),
    mode: str = Form("i2v", description="Generation mode: 'i2v' or 't2v'"),
    num_frames: int = Form(81, description="Number of frames (4k+1 format)"),
    resolution: str = Form(
        "720p", description="Video resolution: 480p, 576p, 720p, 1080p"
    ),
    fps: int = Form(16, description="Frames per second"),
    aspect_ratio: str = Form("9:16", description="Video aspect ratio"),
    steps: int = Form(15, description="Sampling steps (15-25 recommended)"),
    cfg: float = Form(3.0, description="CFG guidance scale (3.0-5.0 recommended)"),
    seed: int = Form(-1, description="Random seed (-1 for random)"),
    high_noise_steps: int = Form(8, description="Steps for high noise pass"),
    shift: float = Form(8.0, description="ModelSamplingSD3 shift"),
    sampler_name: str = Form(
        "dpmpp_2m", description="Sampler: dpmpp_2m, euler, uni_pc"
    ),
    scheduler: str = Form("beta", description="Scheduler: beta, karras, normal"),
    lora_configs: str = Form(
        "", description="JSON array of LoRA configs [{high, low, strength}, ...]"
    ),
    user: User = Depends(get_current_user),
):
    """Queue Cloud Wan22 video generation on RunPod — bf16 full precision (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            mode=mode,
            num_frames=num_frames,
            resolution=resolution,
            fps=fps,
            aspect_ratio=aspect_ratio,
            steps=steps,
            cfg=cfg,
            seed=seed,
            high_noise_steps=high_noise_steps,
            shift=shift,
            sampler_name=sampler_name,
            scheduler=scheduler,
            lora_configs=lora_configs,
        ),
        files={"file": file},
        operation=Operation.GENERATE,
        target_type=MediaType.VIDEO,
        adapter_hint="wan22-cloud-i2v",
        user=user,
        register_job_settings={
            "job_type": "i2v",
        },
        v1_format="cloud",
    )


# =============================================================================
# LTX-2 Image-to-Video Async Endpoint
# =============================================================================


@app.post("/generate-ltx2-i2v-async")
async def generate_ltx2_i2v_async(
    file: UploadFile = File(...),
    prompt: str = Form("The subject in the image begins to move naturally"),
    num_frames: int = Form(97, description="Number of frames (LTX-2: 9-16384, step 8)"),
    output_filename: str = Form("", description="Custom output filename"),
    resolution: str = Form("576p", description="Video resolution: 480p, 576p, 720p"),
    fps: int = Form(25, description="Frames per second (LTX-2 default: 25)"),
    aspect_ratio: str = Form("9:16", description="Video aspect ratio"),
    steps: int = Form(8, description="Sampling steps (LTX-2.3 distilled default: 8)"),
    cfg: float = Form(1.0, description="CFG guidance scale (distilled default: 1.0)"),
    seed: int = Form(-1, description="Random seed (-1 for random)"),
    lora_configs: str = Form(
        "", description="JSON array of LoRA configs [{high, strength}, ...]"
    ),
    audio_prompt: str = Form(
        "", description="Audio description prompt for AV generation (LTX-2.3)"
    ),
    post_processing: str = Form(
        "", description="JSON array of post-processing steps [{type, ...}, ...]"
    ),
    post_audio_file: UploadFile = File(
        None, description="Audio file for add_audio post-processing"
    ),
    compute_target: str = Form(
        "local", description="Compute target: 'local' or 'cloud' (RunPod)"
    ),
    user: User = Depends(get_current_user),
):
    """Queue LTX-2 I2V video generation and return immediately (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            prompt=prompt,
            num_frames=num_frames,
            output_filename=output_filename,
            resolution=resolution,
            fps=fps,
            aspect_ratio=aspect_ratio,
            steps=steps,
            cfg=cfg,
            seed=seed,
            lora_configs=lora_configs,
            audio_prompt=audio_prompt,
            post_processing=post_processing,
            compute_target=compute_target,
        ),
        files={"file": file, "post_audio_file": post_audio_file},
        operation=Operation.GENERATE,
        target_type=MediaType.VIDEO,
        adapter_hint="ltx23-cloud-i2v",
        user=user,
        register_job_settings={
            "job_type": "i2v",
        },
        v1_format="cloud",
    )


# =============================================================================
# POST-PROCESSING ENDPOINT (Standalone for existing media)
# =============================================================================


@app.post("/post-process")
async def post_process_media(
    mode: str = Form(...),  # "upscale", "interpolate", "concat"
    files: List[UploadFile] = File(None),
    media_urls: str = Form(""),  # JSON array of existing media URLs/filenames
    model: str = Form("realesrgan-x4plus"),  # For upscale
    scale: int = Form(2),  # For upscale: 2 or 4
    target_fps: int = Form(60),  # For interpolate
    user: User = Depends(get_current_user),
):
    """
    Standalone post-processing endpoint for existing or uploaded media.

    Modes:
    - upscale: Upscale video using Real-ESRGAN
    - interpolate: Frame interpolation using RIFE
    - concat: Concatenate multiple videos into one
    """
    logger.info(f"🔧 Post-process request: mode={mode}, user={user.id}")

    if not get_comfyui_client:
        raise HTTPException(status_code=503, detail="ComfyUI not available")

    comfyui = get_comfyui_client()
    if not comfyui.is_available():
        raise HTTPException(status_code=503, detail="ComfyUI is not running")

    # Parse media URLs if provided
    existing_media = []
    if media_urls:
        try:
            existing_media = json.loads(media_urls)
            if isinstance(existing_media, str):
                existing_media = [existing_media]
        except json.JSONDecodeError:
            # Single URL as string
            existing_media = [media_urls]

    # Collect input files (uploaded + existing)
    input_paths = []

    # Handle uploaded files
    if files:
        for upload_file in files:
            if upload_file.filename:
                # Save to temp location
                temp_filename = _safe_filename(
                    f"pp_{uuid.uuid4().hex[:8]}_{upload_file.filename}"
                )
                temp_path = _safe_child_path(UPLOAD_DIR, temp_filename)
                content = await upload_file.read()
                temp_path.write_bytes(content)
                input_paths.append(str(temp_path))
                logger.info(f"   📤 Uploaded: {temp_filename}")

    # Handle existing media references
    for media_ref in existing_media:
        resolved_media = _find_existing_media_path(media_ref)
        if resolved_media:
            input_paths.append(resolved_media)
        else:
            logger.warning(f"   ⚠️ Could not find media: {media_ref}")

    if not input_paths:
        raise HTTPException(status_code=400, detail="No input media provided")

    logger.info(f"   📁 Input files: {len(input_paths)}")

    # Validate mode and build workflow
    job_id = f"pp_{uuid.uuid4().hex[:8]}"

    if mode == "upscale":
        if len(input_paths) != 1:
            raise HTTPException(
                status_code=400, detail="Upscale requires exactly 1 input video"
            )

        workflow = comfyui.build_video_upscale_workflow(
            input_video=input_paths[0],
            model=model,
            scale=scale,
            output_prefix=f"upscaled_{job_id}",
        )
        credits_required = 5  # Upscaling cost

    elif mode == "interpolate":
        if len(input_paths) != 1:
            raise HTTPException(
                status_code=400, detail="Interpolation requires exactly 1 input video"
            )

        workflow = comfyui.build_rife_workflow(
            input_video=input_paths[0],
            target_fps=target_fps,
            output_prefix=f"interpolated_{job_id}",
        )
        credits_required = 3  # Interpolation cost

    elif mode == "concat":
        if len(input_paths) < 2:
            raise HTTPException(
                status_code=400, detail="Concatenation requires at least 2 input videos"
            )

        workflow = comfyui.build_video_concat_workflow(
            input_videos=input_paths,
            output_prefix=f"concat_{job_id}",
        )
        credits_required = 2  # Concat cost

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown mode: {mode}. Use: upscale, interpolate, concat",
        )

    # Check credits
    if user.credits < credits_required:
        raise HTTPException(
            status_code=402,
            detail=f"Insufficient credits. Required: {credits_required}, Available: {user.credits}",
        )

    # Queue the workflow
    try:
        prompt_id = await comfyui.queue_prompt(workflow)
    except Exception as e:
        logger.error(f"❌ Failed to queue post-process workflow: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to queue post-process workflow"
        )

    # Store job info
    job_info = {
        "job_id": job_id,
        "prompt_id": prompt_id,
        "mode": mode,
        "input_files": input_paths,
        "user_id": user.id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "job_type": f"post_process_{mode}",
    }
    active_jobs[prompt_id] = job_info
    record_generation_start(prompt_id, job_info)
    save_gen_start_artifacts(
        user_id=user.id,
        prompt_id=prompt_id,
        workflow=workflow,
        prompt=f"Post-process: {mode}",
        job_info=job_info,
    )

    # Deduct credits
    await deduct_credits(user, credits_required, prompt_id, f"Post-process: {mode}")

    logger.info(f"✅ Queued post-process job: {prompt_id} ({mode})")

    return {
        "success": True,
        "prompt_id": prompt_id,
        "job_id": job_id,
        "mode": mode,
        "status": "queued",
        "credits_used": credits_required,
        "message": f"Post-processing job ({mode}) queued. Poll /comfyui/job/{prompt_id} for status.",
    }


@app.get("/comfyui-status")
async def comfyui_status():
    """Check ComfyUI availability and GPU status"""
    if not get_comfyui_client:
        return {"available": False, "error": "ComfyUI client not imported"}

    comfyui = get_comfyui_client()
    is_available = comfyui.is_available()

    if is_available:
        try:
            import requests

            resp = requests.get(f"{comfyui.base_url}/system_stats", timeout=5)
            stats = resp.json() if resp.status_code == 200 else {}
            return {
                "available": True,
                "host": comfyui.host,
                "port": comfyui.port,
                "devices": stats.get("devices", []),
                "model": "wan2.2_i2v_low_noise_14B_Q5_K_S.gguf",
            }
        except Exception as e:
            return {
                "available": True,
                "host": comfyui.host,
                "port": comfyui.port,
                "stats_error": "Unable to fetch ComfyUI system stats",
            }
    else:
        return {
            "available": False,
            "host": comfyui.host,
            "port": comfyui.port,
            "suggestion": "Start ComfyUI: cd ~/oelala/ComfyUI && python main.py --listen",
        }


@app.post("/generate-text")
async def generate_text_video(
    prompt: str = Form(..., description="Text description of the video to generate"),
    num_frames: int = Form(41, description="Number of frames in video"),
    model_type: str = Form("wan22", description="Model type: wan22, ltx2"),
    output_filename: str = Form("", description="Custom output filename"),
    resolution: str = Form("480p", description="Video resolution: 480p, 720p"),
    fps: int = Form(16, description="Frames per second: 8, 12, 16, 24"),
    aspect_ratio: str = Form("1:1", description="Video aspect ratio"),
    post_processing: str = Form("", description="JSON array of post-processing steps"),
    compute_target: str = Form(
        "local", description="Compute target: 'local' or 'cloud' (RunPod)"
    ),
    negative_prompt: str = Form(
        "blurry, low quality, distorted, ugly, artifacts, overexposed, underexposed, flickering, jitter",
        description="Negative prompt",
    ),
    steps: int = Form(-1, description="Sampling steps (-1 for model default)"),
    cfg: float = Form(-1.0, description="CFG guidance scale (-1 for model default)"),
    seed: int = Form(-1, description="Random seed (-1 for random)"),
    lora_configs: str = Form("", description="JSON array of LoRA configs"),
    shift: float = Form(8.0, description="Shift value for cloud sampler"),
    high_noise_steps: int = Form(8, description="High noise steps for cloud dual-pass"),
    sampler_name: str = Form("dpmpp_2m", description="Sampler name for cloud"),
    scheduler: str = Form("beta", description="Scheduler for cloud"),
    audio_prompt: str = Form(
        "", description="Audio description prompt for AV generation (LTX-2.3)"
    ),
    user: User = Depends(get_current_user),  # Require authenticated user
):
    """Generate video from text prompt via ComfyUI T2V workflow (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    # Map model_type + compute_target to adapter_hint
    hint_map = {
        "wan22": "wan22-local-t2v-q6",
        "ltx23": "ltx23-cloud-t2v",
        "ltx2": "ltx23-cloud-t2v",
    }
    hint = hint_map.get(model_type, "wan22-local-t2v-q6")
    if compute_target == "cloud" and model_type == "wan22":
        hint = "wan22-cloud-t2v"

    return await dispatch_v1(
        form=dict(
            prompt=prompt,
            num_frames=num_frames,
            model_type=model_type,
            output_filename=output_filename,
            resolution=resolution,
            fps=fps,
            aspect_ratio=aspect_ratio,
            post_processing=post_processing,
            compute_target=compute_target,
            negative_prompt=negative_prompt,
            steps=steps,
            cfg=cfg,
            seed=seed,
            lora_configs=lora_configs,
            shift=shift,
            high_noise_steps=high_noise_steps,
            sampler_name=sampler_name,
            scheduler=scheduler,
            audio_prompt=audio_prompt,
        ),
        files={},
        operation=Operation.GENERATE,
        target_type=MediaType.VIDEO,
        adapter_hint=hint,
        user=user,
        register_job_settings={
            "job_type": "t2v",
        },
    )


@app.post("/generate-pose")
async def generate_pose_video(
    file: UploadFile = File(...),
    num_frames: int = Form(41, description="Number of frames in video"),
    output_filename: str = Form("", description="Custom output filename"),
    user: User = Depends(get_current_user),  # Require authenticated user
):
    """
    Generate pose-guided video from uploaded image.
    Note: Pose-guided generation is not yet implemented in ComfyUI workflows.
    This endpoint will use the standard I2V workflow for now.
    Requires authentication and credits.
    """
    if not get_comfyui_client:
        raise HTTPException(status_code=503, detail="ComfyUI client not available")

    comfyui = get_comfyui_client()

    if not comfyui.is_available():
        raise HTTPException(
            status_code=503,
            detail="ComfyUI not running. Start with: cd ~/oelala/ComfyUI && python main.py --listen",
        )

    # Adjust num_frames to Wan2.2 format (4k+1) before credit calculation
    k = round((num_frames - 1) / 4)
    k = max(1, k)
    num_frames = 4 * k + 1

    # Calculate duration for credit calculation
    fps = 16
    duration_seconds = num_frames / fps
    width, height = 480, 480

    # Calculate and check credits
    credits_required = calculate_credits(
        "wan22_i2v",
        width=width,
        height=height,
        duration_seconds=int(duration_seconds),
    )
    logger.info(
        f"💰 Pose-guided generation costs {credits_required} credits ({width}x{height}, {duration_seconds:.1f}s) [user={user.id}]"
    )
    await check_credits(user, credits_required)
    job_id = str(uuid.uuid4())

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = _safe_filename(f"pose_{timestamp}_{file.filename}")
    input_path = _safe_child_path(UPLOAD_DIR, input_filename)
    await _save_upload(file, input_path)

    # Upload to ComfyUI
    comfyui_image_name = comfyui.upload_image(str(input_path))
    if not comfyui_image_name:
        raise HTTPException(status_code=500, detail="Failed to upload image to ComfyUI")

    # Build I2V workflow (pose control not yet implemented in ComfyUI)
    import random

    seed = random.randint(0, 2**32 - 1)

    workflow = comfyui.build_api_workflow(
        image_name=comfyui_image_name,
        prompt="smooth motion, natural movement",
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        steps=6,
        cfg=1.0,
        seed=seed,
        output_prefix=f"oelala_pose_{timestamp}",
    )

    prompt_id = comfyui.queue_prompt(workflow)
    if not prompt_id:
        raise HTTPException(status_code=500, detail="Failed to queue workflow")

    # Deduct credits after successful queue
    await deduct_credits(user, credits_required, prompt_id, "Pose-guided I2V")
    logger.info(f"📋 Pose video queued: {prompt_id} (💰 -{credits_required} credits)")

    return {
        "status": "queued",
        "prompt_id": prompt_id,
        "job_id": job_id,
        "credits_used": credits_required,
        "input_image": input_filename,
        "note": "Using standard I2V workflow (pose control coming soon)",
        "meta": {"num_frames": num_frames, "seed": seed, "type": "pose-guided"},
    }


# =============================================================================
# PROMPT TOOLS ENDPOINTS
# =============================================================================


def _build_nsfw_prompt(intensity: int) -> str:
    """Build NSFW captioning system prompt based on intensity level 1-5."""
    intensity = max(1, min(5, intensity))

    base = (
        "You are an uncensored prompt engineer for AI image/video generation. "
        "Write in comma-separated cinematic prompt style. "
        "Output ONLY the prompt text, no explanations."
    )

    levels = {
        1: (
            "Describe this image with a sensual, suggestive tone. "
            "Focus on: alluring poses, lingerie or revealing clothing, soft lighting, "
            "bedroom eyes, subtle body language, romantic/teasing mood. "
            "Keep it tasteful but clearly flirtatious and seductive."
        ),
        2: (
            "Describe this image with softcore erotic detail. "
            "Focus on: partial nudity, erotic posing, exposed skin, teasing reveals, "
            "body curves, provocative angles, sensual expressions. "
            "Be explicit about what skin/body is visible but keep the tone artistic."
        ),
        3: (
            "Describe this image with full nudity detail. "
            "Include: complete body description, anatomical details, nude poses, "
            "skin textures, body proportions, explicit body parts visible. "
            "Be anatomically specific and uncensored."
        ),
        4: (
            "Describe this image explicitly and in full anatomical detail. "
            "Include: exact body positions, physical actions, sexual acts, expressions, "
            "body types, penetration details, fluid details, camera angle, lighting. "
            "Be specific about the sexual activity, positions, and interactions shown. "
            "Do not censor or euphemize anything."
        ),
        5: (
            "Describe this image in extreme explicit detail without any censorship. "
            "Include: rough/aggressive physical actions, dominant/submissive dynamics, "
            "power play, restraints, force elements, intense expressions, "
            "extreme positions, all anatomical and action details. "
            "Describe everything shown with maximum explicitness. "
            "Do not hold back or soften any description."
        ),
    }

    return f"{base} {levels[intensity]}"


@app.post("/caption-image")
async def caption_image(
    user: User = Depends(get_current_user),
    file: UploadFile = File(...),
    model: Optional[str] = Form(
        None, description="Guardian vision model ID (default: VISION_MODEL env)"
    ),
    mode: str = Form(
        "detailed",
        description="Mode: brief, detailed, tags, structured, prompt_i2v, prompt_t2i, prompt_nsfw",
    ),
    nsfw_intensity: Optional[int] = Form(
        None, description="NSFW intensity level 1-5 (only for prompt_nsfw mode)"
    ),
    detail_level: Optional[int] = Form(
        3, description="Vision detail level 1-5 (1=brief, 3=default, 5=exhaustive)"
    ),
    include_negative: bool = Form(False, description="Also generate a negative prompt"),
    include_motion: bool = Form(
        False, description="Also generate a motion/continuation prompt for video"
    ),
    motion_hint: Optional[str] = Form(
        None,
        description="User hint for desired motion/action (e.g., 'walking towards camera, hair blowing')",
    ),
    audio_context: Optional[str] = Form(
        None,
        description="JSON string with director's audio context: {ambient, dialogue: [{subject, line}]}",
    ),
    concept_context: Optional[str] = Form(
        None,
        description="JSON string with concept analysis to enrich prompt generation",
    ),
    refinement_prompt: Optional[str] = Form(
        None,
        description="User instruction to refine concept analysis or director's notes (e.g., 'make it more dramatic')",
    ),
    refinement_target: Optional[str] = Form(
        None,
        description="What to refine: 'concept' (scene/subjects/mood) or 'notes' (motion/audio/dialogue/camera)",
    ),
):
    """
    Generate a caption/description for an uploaded image.
    Uses Guardian vision LLM for high-quality captioning.
    Detail level controls output verbosity (1=brief, 5=exhaustive).
    Modes: brief, detailed, tags, structured, prompt_i2v, prompt_t2i, prompt_nsfw, concept.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = _safe_filename(f"caption_{timestamp}_{file.filename}")
    input_path = _safe_child_path(UPLOAD_DIR, input_filename)
    await _save_upload(file, input_path)

    # Clamp detail_level
    detail_level = max(1, min(5, detail_level or 3))

    # Detail-level modifiers for the vision prompt
    detail_prefixes = {
        1: "Be very brief (1-2 sentences max). ",
        2: "Keep it concise (2-3 sentences). ",
        3: "",  # default — use the prompt as-is
        4: "Be thorough and detailed. Include subtle details about expression, texture, fabric, background elements, lighting direction, and compositional style. ",
        5: (
            "Be EXHAUSTIVELY detailed. Describe EVERYTHING: exact pose and body language, every clothing item and accessory, "
            "hair style/color/texture, facial expression and micro-expressions, skin details, background elements and their positions, "
            "lighting setup (direction, color temperature, shadows), color palette, camera angle, depth of field, mood, atmosphere, "
            "art style, and any text or symbols visible. Leave nothing out. "
        ),
    }

    # Route through Guardian vision LLM
    logger.info(
        f"🔮 Captioning with Guardian vision model: {model}, detail_level={detail_level}..."
    )
    caption_prompts = {
        "brief": "In one sentence, describe the main subject of this image.",
        "detailed": "Describe this image in rich detail: the main subject, their appearance, clothing, pose, expression, the background, lighting, colors, and overall mood.",
        "tags": "List comma-separated keywords that describe this image (subjects, style, colors, mood, technical details). Output only the comma-separated list.",
        "structured": "Describe this image using this exact format — Subject: ..., Style: ..., Mood: ..., Setting: ..., Colors: ...",
        "prompt_i2v": (
            "You are a prompt engineer for AI video generation (image-to-video). "
            "Describe the activity, motion, and action happening in this image as a video generation prompt. "
            "Focus on: what the subject is DOING, the movement/gesture, camera angle, and how the scene would continue as a video. "
            "Write in present tense, comma-separated cinematic style. "
            "Example format: 'woman walking through rain, slow motion, hair flowing, puddle reflections, cinematic tracking shot, moody blue lighting'. "
            "Output ONLY the prompt text, no explanations."
        ),
        "prompt_t2i": (
            "You are a prompt engineer for AI image generation (Stable Diffusion / SDXL / Flux). "
            "Describe this image as a detailed text-to-image prompt. "
            "Include: subject description, pose, clothing/appearance, setting, lighting, art style, camera angle, quality tags. "
            "Write in comma-separated tag style with quality boosters. "
            "Example format: 'beautiful woman, red dress, standing in garden, golden hour, bokeh, soft lighting, masterpiece, 8k, photorealistic'. "
            "Output ONLY the prompt text, no explanations."
        ),
        "prompt_nsfw": _build_nsfw_prompt(nsfw_intensity or 3),
    }

    custom_prompt = caption_prompts.get(mode, caption_prompts["detailed"])

    # Apply detail level modifier (prepend to prompt)
    detail_prefix = detail_prefixes.get(detail_level, "")
    if detail_prefix:
        custom_prompt = detail_prefix + custom_prompt

    import base64 as _b64

    with open(input_path, "rb") as f:
        image_b64 = _b64.b64encode(f.read()).decode("utf-8")

    # ── Concept analysis mode — structured scene breakdown ──
    # JSON output is compact — 3072 is plenty for a full concept card
    concept_max_tokens = 3072

    if mode == "concept":
        # Check if this is a REFINEMENT request (existing concept + user instruction)
        if refinement_prompt and concept_context:
            try:
                current_concept = json.loads(concept_context)
            except json.JSONDecodeError:
                current_concept = {}

            target = refinement_target or "concept"
            # Compact JSON to save tokens — no indentation
            current_json = json.dumps(current_concept, separators=(",", ":"))

            if target == "notes":
                concept_prompt = (
                    "Update ONLY the director's notes in this JSON.\n"
                    f"Current analysis: {current_json}\n\n"
                    f'Instruction: "{refinement_prompt}"\n\n'
                    "RULES:\n"
                    "1. Copy scene, subjects, and mood EXACTLY — do NOT change them.\n"
                    "2. Update ONLY: suggested_motion, suggested_audio, suggested_dialogue, suggested_camera.\n"
                    '3. For suggested_dialogue: [{"subject":"name","line":"speech"}]\n'
                    "4. For suggested_camera: describe the complete camera direction as a cinematic sentence — "
                    "shot type, movement, speed, composition changes, and how it serves the story.\n"
                    "5. All notes should work together as a coherent director's cut.\n"
                    "6. Output ONLY compact JSON on a single line. No markdown, no explanation.\n"
                    "7. Same keys as the input."
                )

                # Notes refinement: text-only call (no image needed, avoids LLM reinterpreting visual)
                import httpx
                from guardian_client import (
                    wait_for_comfyui_idle,
                    free_comfyui_vram as _free_comfy_vram,
                )

                await wait_for_comfyui_idle()
                await _free_comfy_vram()

                vision_model = model or VISION_MODEL
                text_body = {
                    "model": vision_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Output ONLY raw JSON. No markdown, no explanation.",
                        },
                        {"role": "user", "content": concept_prompt},
                    ],
                    "max_tokens": 2048,
                    "temperature": 0.3,
                }

                logger.info(
                    f"🔮 Notes refinement (text-only): {refinement_prompt[:100]}"
                )

                async with httpx.AsyncClient(
                    timeout=480.0, headers=_guardian_headers()
                ) as client:
                    response = await client.post(
                        f"{GUARDIAN_BASE}/v1/chat/completions",
                        json=text_body,
                    )
                    response.raise_for_status()
                    resp_data = response.json()
                    choice = resp_data["choices"][0]
                    raw = choice["message"]["content"] or ""
                    finish_reason = choice.get("finish_reason", "unknown")
                    logger.info(
                        f"🔮 Notes refine: finish_reason={finish_reason}, len={len(raw)}"
                    )

                raw = _strip_think_tags(raw)
                logger.info(f"🔮 Notes refine raw: {raw[:500] if raw else '(empty)'}")

                # Parse JSON — find valid JSON block
                json_matches = list(re.finditer(r"\{[\s\S]*\}", raw))
                for match in reversed(json_matches):
                    try:
                        parsed = json.loads(match.group())
                        if "suggested_motion" in parsed or "suggested_audio" in parsed:
                            # Enforce: keep original scene/subjects/mood
                            parsed["scene"] = current_concept.get("scene", "")
                            parsed["subjects"] = current_concept.get("subjects", [])
                            parsed["mood"] = current_concept.get("mood", "")
                            logger.info(
                                f"🔮 Notes refined: {len(parsed.get('suggested_dialogue', []))} dialogue lines"
                            )
                            return {
                                "concept": parsed,
                                "model": vision_model,
                                "mode": "concept",
                            }
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ Notes refine JSON parse attempt failed: {e}")
                        continue

                # No silent fallback — raise error so frontend sees it
                logger.error(
                    f"❌ Notes refinement JSON parse failed. Raw LLM output:\n{raw}"
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Notes refinement failed: LLM returned invalid JSON. Raw: {raw[:300]}",
                )
            else:
                # Concept refinement: text-only call (no image) — sending image confuses the model
                # into describing what it sees instead of refining the JSON
                concept_refine_prompt = (
                    "Update the JSON based on the instruction.\n"
                    f"Current analysis: {current_json}\n\n"
                    f'Instruction: "{refinement_prompt}"\n\n'
                    "RULES:\n"
                    "1. Update scene, subjects, mood, suggested_motion, suggested_audio, suggested_dialogue, suggested_camera.\n"
                    '2. For suggested_dialogue: [{"subject":"name","line":"speech"}]\n'
                    "3. Output ONLY compact JSON on a single line. No markdown, no explanation.\n"
                    "4. Same keys as the input."
                )

                import httpx
                from guardian_client import (
                    wait_for_comfyui_idle,
                    free_comfyui_vram as _free_comfy_vram,
                )

                await wait_for_comfyui_idle()
                await _free_comfy_vram()

                vision_model = model or VISION_MODEL
                text_body = {
                    "model": vision_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Output ONLY raw JSON. No markdown, no explanation.",
                        },
                        {"role": "user", "content": concept_refine_prompt},
                    ],
                    "max_tokens": 2048,
                    "temperature": 0.3,
                }

                logger.info(
                    f"🔮 Concept refinement (text-only, {target}): {refinement_prompt[:100]}"
                )

                async with httpx.AsyncClient(
                    timeout=480.0, headers=_guardian_headers()
                ) as client:
                    response = await client.post(
                        f"{GUARDIAN_BASE}/v1/chat/completions",
                        json=text_body,
                    )
                    response.raise_for_status()
                    resp_json = response.json()
                    raw = resp_json["choices"][0]["message"]["content"] or ""
                    finish_reason = resp_json["choices"][0].get(
                        "finish_reason", "unknown"
                    )
                    logger.info(
                        f"🔮 Concept refine: finish_reason={finish_reason}, len={len(raw)}"
                    )

                raw = _strip_think_tags(raw)

                # Parse JSON
                json_matches = list(re.finditer(r"\{[\s\S]*\}", raw))
                for match in reversed(json_matches):
                    try:
                        parsed = json.loads(match.group())
                        if "scene" in parsed or "subjects" in parsed:
                            logger.info(
                                f"🔮 Concept refined: {len(parsed.get('subjects', []))} subjects"
                            )
                            return {
                                "concept": parsed,
                                "model": vision_model,
                                "mode": "concept",
                            }
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"⚠️ Concept refine JSON parse attempt failed: {e}"
                        )
                        continue

                logger.error(
                    f"❌ Concept refinement JSON parse failed. Raw LLM output:\n{raw}"
                )
                raise HTTPException(
                    status_code=500,
                    detail=f"Concept refinement failed: LLM returned invalid JSON. Raw: {raw[:300]}",
                )

        else:
            concept_prompt = (
                "You are an expert cinematographer and video director analyzing this image for AI video production. "
                "Analyze the image and respond with ONLY valid JSON (no markdown, no ``` fences, no explanation).\n"
                "Return this exact structure:\n"
                '{"scene": "description of overall scene and setting", '
                '"subjects": [{"label": "descriptive name", "description": "detailed appearance", "position": "where in frame"}], '
                '"mood": "atmosphere and emotional tone", '
                '"suggested_motion": "how subjects and elements would naturally move in a video continuation", '
                '"suggested_audio": "ambient sounds, music mood, environmental audio that fits this scene", '
                '"suggested_dialogue": [{"subject": "label matching subjects array", "line": "suggested speech, narration, or vocal expression"}], '
                '"suggested_camera": "complete camera direction as a single cinematic sentence: shot type (wide/medium/close-up/extreme close-up), '
                "movement (pan, tilt, dolly, crane, orbit, tracking, handheld, steadicam), speed and rhythm, "
                "composition changes, and how it serves the scene emotionally. "
                "Example: slow dolly-in from wide establishing shot to medium close-up on subject, "
                'slight upward tilt revealing the sky as music swells"}'
                "\nBe cinematic, creative, and specific. If people are visible, ALWAYS suggest dialogue for each person. "
                "For audio, consider environmental sounds, implied sounds, and mood-appropriate music or ambience. "
                "For camera, think like a director: the camera movement should tell part of the story."
            )
            detail_prefix = detail_prefixes.get(detail_level, "")
            if detail_prefix:
                concept_prompt = detail_prefix + concept_prompt

        raw = await analyze_image_with_vision(
            image_b64,
            custom_prompt=concept_prompt,
            model_override=model,
            max_tokens=concept_max_tokens,
        )
        raw = _strip_think_tags(raw)
        logger.info(f"🔮 Concept raw (full): {raw}")

        # Parse JSON response — find the last valid JSON block
        json_matches = list(re.finditer(r"\{[\s\S]*\}", raw))
        for match in reversed(json_matches):
            try:
                parsed = json.loads(match.group())
                if "scene" in parsed or "subjects" in parsed:
                    logger.info(
                        f"🔮 Concept parsed: {len(parsed.get('subjects', []))} subjects detected"
                    )
                    return {
                        "concept": parsed,
                        "model": model or VISION_MODEL,
                        "mode": "concept",
                    }
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ Concept JSON parse attempt failed: {e}")
                continue

        # No silent fallback — raise error so we can debug
        logger.error(f"❌ Concept JSON parse failed. Raw LLM output:\n{raw}")
        raise HTTPException(
            status_code=500,
            detail=f"Concept analysis failed: LLM returned invalid JSON. Raw: {raw[:300]}",
        )

    # ── Multi-output mode: positive + negative in prompt modes ──
    # In prompt mode, always generate both positive and negative prompts.
    # Motion/camera/action is baked INTO the positive prompt, not separate.
    is_prompt = mode.startswith("prompt_")
    is_nsfw = mode == "prompt_nsfw"

    if is_prompt:
        parts = []
        parts.append(
            "Analyze this image and respond with ONLY valid JSON (no markdown, no ``` fences, no explanation)."
        )

        # Inject concept context if provided (from the Analyze step)
        concept_text = ""
        director_notes_text = ""
        if concept_context:
            try:
                cc = json.loads(concept_context)
                concept_parts = []
                if cc.get("scene"):
                    concept_parts.append(f"Scene context: {cc['scene']}")
                if cc.get("mood"):
                    concept_parts.append(f"Mood: {cc['mood']}")
                if cc.get("subjects"):
                    subj_descs = [
                        f"{s.get('label', 'subject')}: {s.get('description', '')}"
                        for s in cc["subjects"]
                    ]
                    concept_parts.append(f"Subjects: {'; '.join(subj_descs)}")
                if concept_parts:
                    concept_text = (
                        " Director's concept: " + ". ".join(concept_parts) + "."
                    )

                # Build mandatory director's notes for the VISUAL prompt (motion + camera only)
                notes = []
                if cc.get("suggested_motion"):
                    notes.append(f"Motion/action: {cc['suggested_motion']}")
                if cc.get("suggested_camera"):
                    notes.append(f"Camera movement: {cc['suggested_camera']}")
                if notes:
                    director_notes_text = (
                        " MANDATORY director's notes that MUST be woven into the prompt: "
                        + ". ".join(notes)
                        + "."
                    )
            except (json.JSONDecodeError, TypeError):
                pass

        # Inject audio/dialogue context for combined prompt generation
        audio_text = ""
        if audio_context:
            try:
                ac = json.loads(audio_context)
                audio_parts = []
                if ac.get("ambient"):
                    audio_parts.append(f"ambient sounds: {ac['ambient']}")
                if ac.get("dialogue"):
                    for d in ac["dialogue"]:
                        audio_parts.append(
                            f'{d.get("subject", "person")} says: "{d.get("line", "")}"'
                        )
                if audio_parts:
                    audio_text = " Audio direction: " + "; ".join(audio_parts) + "."
            except (json.JSONDecodeError, TypeError):
                pass

        # Positive prompt instruction — includes motion/action naturally
        # Inject motion hint if provided
        motion_hint_text = ""
        if include_motion and motion_hint and motion_hint.strip():
            motion_hint_text = f" The user wants this specific motion/action: {motion_hint.strip()}. Incorporate this into the prompt."
        elif include_motion:
            motion_hint_text = " Include natural motion descriptions (subject animation, environmental movement)."

        if mode == "prompt_i2v":
            parts.append(
                '"positive": a cinematic image-to-video prompt describing subject, their appearance, action, movement, '
                "camera angle, lighting, and how the scene would naturally continue as a video. "
                "Include camera motion (dolly, pan, tilt, tracking), subject animation (gestures, hair flowing, body movement), "
                "and environmental motion where relevant."
                f"{concept_text}{director_notes_text}{motion_hint_text}{audio_text} "
                "Comma-separated, present tense, cinematic style."
            )
        elif mode == "prompt_t2i":
            parts.append(
                '"positive": a detailed text-to-image prompt with subject description, pose, clothing, setting, lighting, '
                "art style, camera angle, quality boosters."
                f"{concept_text}{director_notes_text}{motion_hint_text} "
                "Comma-separated tag style."
            )
        elif is_nsfw:
            nsfw_base = _build_nsfw_prompt(nsfw_intensity or 3)
            parts.append(
                f'"positive": {nsfw_base}{director_notes_text}{motion_hint_text}'
            )
        else:
            parts.append(
                f'"positive": {caption_prompts.get(mode, caption_prompts["detailed"])}'
            )

        # Apply detail level
        if detail_prefix:
            parts[1] = detail_prefix + parts[1]

        # Negative prompt — always included in prompt mode
        neg_style = (
            "explicit NSFW negative tags (deformed, bad anatomy, low quality, watermark, text, censored, pixelated)"
            if is_nsfw
            else "things to AVOID in AI generation (bad anatomy, blurry, low quality, watermark, distorted, artifacts, text)"
        )
        parts.append(f'"negative": {neg_style}. Keep it concise, comma-separated.')

        # Audio prompt — when audio context or concept dialogue/audio is provided
        # Check if concept_context has dialogue or ambient audio that should trigger audio generation
        concept_has_audio = False
        if concept_context:
            try:
                cc_audio = json.loads(concept_context)
                concept_has_audio = bool(cc_audio.get("suggested_audio")) or bool(
                    cc_audio.get("suggested_dialogue")
                    and any(
                        (
                            d.get("line", "").strip()
                            if isinstance(d, dict)
                            else str(d).strip()
                        )
                        for d in cc_audio["suggested_dialogue"]
                    )
                )
            except (json.JSONDecodeError, TypeError):
                pass
        has_audio = bool(audio_context) or concept_has_audio
        if has_audio:
            # Build explicit dialogue/ambient reference so the LLM includes them in the audio field
            audio_ref_parts = []
            # From audio_context form field
            if audio_context:
                try:
                    ac = json.loads(audio_context)
                    if ac.get("ambient"):
                        audio_ref_parts.append(f"Ambient: {ac['ambient']}")
                    if ac.get("dialogue"):
                        for d in ac["dialogue"]:
                            subj = d.get("subject", "person")
                            line = d.get("line", "")
                            if line.strip():
                                audio_ref_parts.append(f'{subj} says: "{line}"')
                except (json.JSONDecodeError, TypeError):
                    pass
            # From concept_context director's notes (dialogue + ambient)
            if concept_context:
                try:
                    cc_audio = json.loads(concept_context)
                    if cc_audio.get("suggested_audio") and not any(
                        "Ambient:" in p for p in audio_ref_parts
                    ):
                        audio_ref_parts.append(
                            f"Ambient: {cc_audio['suggested_audio']}"
                        )
                    if cc_audio.get("suggested_dialogue"):
                        for d in cc_audio["suggested_dialogue"]:
                            if isinstance(d, dict) and d.get("line", "").strip():
                                ref = (
                                    f'{d.get("subject", "person")} says: "{d["line"]}"'
                                )
                                if ref not in audio_ref_parts:
                                    audio_ref_parts.append(ref)
                except (json.JSONDecodeError, TypeError):
                    pass
            audio_ref = ""
            if audio_ref_parts:
                audio_ref = (
                    " The user provided these audio/dialogue notes that MUST be included: "
                    + "; ".join(audio_ref_parts)
                    + "."
                )
            parts.append(
                '"audio": a detailed audio description for AI video with audio generation. '
                "Describe the soundscape: ambient sounds, environmental audio, speech/dialogue with tone and emotion, "
                "music mood if appropriate. Include ALL character dialogue with natural delivery cues (tone, emotion, pacing)."
                f"{audio_ref} "
                "Example: 'gentle wind, birds chirping in distance, woman speaks softly: Hello there, footsteps on gravel'. "
                "Keep it descriptive and cinematic."
            )

        expected_fields = '{"positive": "...", "negative": "..."'
        if has_audio:
            expected_fields += ', "audio": "..."'
        expected_fields += "}"
        combined_prompt = (
            " ".join(parts) + f"\n\nRespond with exactly: {expected_fields}"
        )

        logger.info(f"🔮 Multi-output caption: model={model}, mode={mode}")
        raw = await analyze_image_with_vision(
            image_b64,
            custom_prompt=combined_prompt,
            model_override=model,
        )

        # Strip think tags before parsing
        raw = _strip_think_tags(raw)
        logger.debug(f"🔮 Vision raw (stripped): {raw[:500]}")

        # Parse JSON from response — find the last {} block (most likely the actual JSON)
        json_matches = list(re.finditer(r"\{[^{}]*\}", raw))
        if not json_matches:
            # Try greedy match for nested objects
            json_matches = list(re.finditer(r"\{[\s\S]*\}", raw))

        for match in reversed(json_matches):
            try:
                parsed = json.loads(match.group())
                if "positive" in parsed:
                    caption_text = _strip_think_tags(parsed.get("positive", raw))
                    neg_text = _strip_think_tags(parsed.get("negative", ""))
                    audio_text_out = (
                        _strip_think_tags(parsed.get("audio", ""))
                        if has_audio
                        else None
                    )
                    logger.info(
                        f"🔮 Parsed pos={len(caption_text)}c neg={len(neg_text)}c audio={'yes' if audio_text_out else 'no'}"
                    )
                    result = {
                        "caption": caption_text,
                        "negative_prompt": neg_text,
                        "model": model or VISION_MODEL,
                        "mode": mode,
                    }
                    if audio_text_out:
                        result["audio_prompt"] = audio_text_out
                    return result
            except json.JSONDecodeError:
                continue

        # Fallback: return raw as caption
        logger.warning("⚠️ Multi-output JSON parse failed, returning raw caption")
        return {
            "caption": _strip_think_tags(raw),
            "negative_prompt": None,
            "model": model or VISION_MODEL,
            "mode": mode,
        }

    # ── Single-output mode (original behavior) ──
    description = _strip_think_tags(
        await analyze_image_with_vision(
            image_b64, custom_prompt=custom_prompt, model_override=model
        )
    )

    return {
        "caption": description,
        "negative_prompt": None,
        "model": model or VISION_MODEL,
        "mode": mode,
    }


class RefineCaptionRequest(BaseModel):
    """Request body for refining generated captions with user suggestions."""

    positive: str = ""
    negative: Optional[str] = None
    suggestion: str  # User's refinement instruction
    model: Optional[str] = None


@app.post("/refine-caption")
async def refine_caption(
    req: RefineCaptionRequest,
    user: User = Depends(get_current_user),
):
    """
    Refine/tweak generated prompts based on user suggestions.
    Takes the current outputs + a suggestion, returns improved versions.
    """
    if not req.suggestion.strip():
        raise HTTPException(status_code=400, detail="suggestion is required")

    t2t_model = req.model or "GLM-4.7-Flash-Claude-Opus-Reasoning"
    logger.info(
        f"✏️ Refining caption with suggestion: '{req.suggestion[:80]}...' model={t2t_model}"
    )

    # Build the refinement prompt
    current = f"Current positive prompt:\n{req.positive}"
    if req.negative:
        current += f"\n\nCurrent negative prompt:\n{req.negative}"

    fields = '{"positive": "...'
    if req.negative is not None:
        fields += '", "negative": "...'
    fields += '"}'

    system_prompt = (
        "You are an expert prompt engineer for AI image and video generation. "
        "The user has generated prompts from an image and wants to refine them. "
        "Apply the user's suggestion while preserving the original subject and intent. "
        "Respond with ONLY valid JSON (no markdown fences, no explanation). "
        f"JSON structure: {fields}"
    )

    user_prompt = (
        f"{current}\n\n"
        f"User's refinement instruction: {req.suggestion.strip()}\n\n"
        f"Apply the refinement and return the improved prompts as JSON."
    )

    from guardian_client import (
        wait_for_comfyui_idle,
        free_comfyui_vram as _free_comfy_vram,
    )

    await wait_for_comfyui_idle()
    await _free_comfy_vram()

    t2t_body = {
        "model": t2t_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 1024,
        "temperature": 0.7,
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(
                timeout=120.0, headers=_guardian_headers()
            ) as client:
                response = await client.post(
                    f"{GUARDIAN_BASE}/v1/chat/completions",
                    json=t2t_body,
                )
                if response.status_code == 503 and attempt < max_retries - 1:
                    import asyncio

                    await asyncio.sleep(15)
                    continue
                response.raise_for_status()
                result = response.json()
                msg = result["choices"][0]["message"]
                llm_output = (
                    msg.get("content") or msg.get("reasoning_content", "")
                ).strip()
                llm_output = _strip_think_tags(llm_output)

                # Parse JSON
                json_match = re.search(r"\{[\s\S]*\}", llm_output)
                if json_match:
                    parsed = json.loads(json_match.group())
                    return {
                        "positive": parsed.get("positive", req.positive),
                        "negative": parsed.get("negative", req.negative),
                    }

                # Fallback: return raw as positive
                return {"positive": llm_output, "negative": req.negative}

        except Exception as e:
            if attempt < max_retries - 1:
                import asyncio

                await asyncio.sleep(10)
                continue
            logger.error(f"Refine caption failed: {e}")
            raise HTTPException(status_code=500, detail=f"Refine failed: {str(e)}")

    raise HTTPException(status_code=503, detail="Model still loading — try again")


class MotionPromptRequest(BaseModel):
    """Request body for motion prompt generation."""

    prompt: str
    model: Optional[str] = None  # T2T model override


@app.post("/generate-motion-prompt")
async def generate_motion_prompt(
    req: MotionPromptRequest,
    user: User = Depends(get_current_user),
):
    """
    Generate a creative motion/camera prompt from a text description.
    Uses a T2T reasoning LLM to generate cinematic motion cues.
    Call this AFTER /caption-image to get a separate motion prompt.
    """
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")

    t2t_model = req.model or "GLM-4.7-Flash-Claude-Opus-Reasoning"
    logger.info(f"🎬 Generating motion prompt with T2T model: {t2t_model}...")

    motion_system = (
        "You are a cinematographer and motion designer for AI video generation. "
        "Given an image description, generate creative and specific camera motion and animation cues. "
        "Focus on: camera movements (dolly, pan, tilt, crane, tracking, zoom), "
        "subject animation (hair flowing, fabric rippling, walking, gesturing), "
        "environmental motion (clouds drifting, light shifting, particles floating), "
        "and cinematic timing (slow motion, speed ramp, freeze frame). "
        "Output ONLY the motion prompt text — comma-separated, concise, cinematic style. "
        "Be creative and specific to the scene described. No explanations, no labels."
    )

    motion_user = (
        f"Based on this image analysis, generate a creative motion/camera prompt for AI video generation:\n\n"
        f"{req.prompt.strip()}"
    )

    try:
        import httpx
        from guardian_client import (
            wait_for_comfyui_idle,
            free_comfyui_vram as _free_comfy_vram,
        )

        await wait_for_comfyui_idle()
        await _free_comfy_vram()

        t2t_body = {
            "model": t2t_model,
            "messages": [
                {"role": "system", "content": motion_system},
                {"role": "user", "content": motion_user},
            ],
            "max_tokens": 512,
            "temperature": 1.0,
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(
                    timeout=120.0, headers=_guardian_headers()
                ) as client:
                    resp = await client.post(
                        f"{GUARDIAN_BASE}/v1/chat/completions", json=t2t_body
                    )
                    if resp.status_code == 503 and attempt < max_retries - 1:
                        logger.info(
                            f"⏳ Guardian 503 (loading T2T model), retry {attempt + 1}/{max_retries}..."
                        )
                        import asyncio

                        await asyncio.sleep(15)
                        continue
                    resp.raise_for_status()
                    result = resp.json()
                    msg = result["choices"][0]["message"]
                    # Prefer content over reasoning_content; strip thinking artifacts
                    motion_prompt = (msg.get("content") or "").strip()
                    reasoning = (msg.get("reasoning_content") or "").strip()
                    # If content is empty but reasoning exists, use reasoning
                    if not motion_prompt and reasoning:
                        motion_prompt = reasoning
                    motion_prompt = _strip_think_tags(motion_prompt)
                    # Strip common thinking patterns: lines starting with reasoning verbs/markers
                    # Keep only the last paragraph-block (the actual prompt output)
                    lines = motion_prompt.split("\n")
                    # Find last non-empty line block — the actual motion prompt
                    result_lines = []
                    for line in reversed(lines):
                        stripped = line.strip()
                        if not stripped and result_lines:
                            break  # Hit empty line after collecting output = done
                        if stripped:
                            result_lines.insert(0, stripped)
                    if result_lines:
                        motion_prompt = " ".join(result_lines)
                    logger.info(f"🎬 Motion prompt generated: {motion_prompt[:100]}...")
                    return {"motion_prompt": motion_prompt, "model": t2t_model}
            except httpx.ConnectError:
                logger.warning("⚠️ Guardian not available for motion prompt generation")
                raise HTTPException(status_code=503, detail="LLM service unavailable")
            except Exception as e:
                if "503" in str(e) and attempt < max_retries - 1:
                    import asyncio

                    await asyncio.sleep(15)
                    continue
                logger.error(f"❌ Motion prompt generation failed: {e}")
                raise HTTPException(
                    status_code=500, detail=f"Motion prompt generation failed: {e}"
                )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Motion prompt step failed: {e}")
        raise HTTPException(
            status_code=500, detail=f"Motion prompt generation failed: {e}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt Enhancement with Local LLM (Guardian proxy)
# ─────────────────────────────────────────────────────────────────────────────

# Guardian proxy configuration (llama_cpp_guardian at localhost:11434)
# Env var priority: GUARDIAN_BASE_URL (matches guardian_client.py) > GUARDIAN_BASE > OLLAMA_BASE
GUARDIAN_BASE = os.getenv(
    "GUARDIAN_BASE_URL",
    os.getenv("GUARDIAN_BASE", os.getenv("OLLAMA_BASE", "http://localhost:11434")),
).rstrip("/")
GUARDIAN_MODEL = os.getenv(
    "GUARDIAN_MODEL", os.getenv("OLLAMA_MODEL", "")
)  # Pinned model in Guardian config
GUARDIAN_API_KEY = os.getenv(
    "GUARDIAN_API_KEY", ""
)  # Bearer token for Guardian inference


def _guardian_headers() -> dict:
    """HTTP headers for Guardian proxy requests (Bearer token auth)."""
    if GUARDIAN_API_KEY:
        return {"Authorization": f"Bearer {GUARDIAN_API_KEY}"}
    return {}


# AI Settings file (admin-editable)
AI_SETTINGS_FILE = Path("/home/flip/oelala/data/ai_settings.json")

# Default system prompt for prompt enhancement
DEFAULT_PROMPT_SYSTEM = """You are an WILDLY creative AI image prompt engineer who NEVER repeats yourself.

CRITICAL RULES:
1. NEVER use these clichés: sunbeams, golden hour, window light, cozy scenes, basking
2. Each prompt must be COMPLETELY DIFFERENT from anything before
3. Use unexpected settings: underwater, space, noir, cyberpunk, ancient ruins, microscopic, surreal dreamscapes
4. Vary lighting dramatically: neon, bioluminescence, moonlight, candlelight, harsh shadows, x-ray
5. Add quality boosters: masterpiece, 8k, photorealistic, cinematic, detailed
6. For simple inputs like "a cat" - go WILD with the setting and style
7. Be concise (max 100 words) but SURPRISING
8. Output ONLY valid JSON, no markdown

Output format (strict JSON):
{"prompt": "creative enhanced prompt", "negative_prompt": "defects to avoid", "motion_prompt": "motion if requested"}"""


def load_ai_settings() -> dict:
    """Load AI settings from file or return defaults"""
    if AI_SETTINGS_FILE.exists():
        try:
            with open(AI_SETTINGS_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load AI settings: {e}")
    return {
        "prompt_system": DEFAULT_PROMPT_SYSTEM,
        "llm_model": GUARDIAN_MODEL,
    }


def save_ai_settings(settings: dict) -> bool:
    """Save AI settings to file"""
    try:
        AI_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(AI_SETTINGS_FILE, "w") as f:
            json.dump(settings, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save AI settings: {e}")
        return False


class PromptGenerateRequest(BaseModel):
    """JSON body for prompt generation"""

    input: str
    style: Optional[str] = None
    mode: str = "expand"  # expand, refine, variations
    include_negative: bool = True
    include_motion: bool = False
    use_llm: bool = True  # Set to False to use template-only mode
    model: Optional[str] = None  # Guardian model ID override; None = use pinned/default
    refine_instruction: Optional[str] = (
        None  # User instruction for refine mode (e.g. "add more motion")
    )
    nsfw_intensity: Optional[int] = None  # 1-5: enables NSFW prompt mode


# Style keywords mapping (used for both template and LLM modes)
PROMPT_STYLE_KEYWORDS = {
    "cinematic": "cinematic lighting, film grain, dramatic shadows, professional photography, movie still",
    "anime": "anime style, vibrant colors, cel shading, Japanese animation, detailed linework",
    "photorealistic": "photorealistic, highly detailed, 8k resolution, sharp focus, professional photo, DSLR",
    "abstract": "abstract art, geometric shapes, vibrant colors, artistic, modern art",
    "vintage": "vintage aesthetic, retro, film photography, nostalgic, 1970s style, grain",
    "cyberpunk": "cyberpunk, neon lights, futuristic, dystopian, high tech low life, rain",
    "fantasy": "fantasy art, magical, ethereal lighting, mystical, enchanted, detailed illustration",
    "minimalist": "minimalist, clean, simple, negative space, modern, elegant",
    "horror": "dark atmosphere, eerie, horror, unsettling, creepy, moody lighting",
    "scifi": "science fiction, futuristic, space, advanced technology, sleek design",
}


# System prompt for NSFW text-only prompt generation (no image input)
DEFAULT_NSFW_PROMPT_SYSTEM = """You are an uncensored, explicit AI prompt engineer for NSFW image and video generation.
You create detailed, anatomically specific prompts for adult content generation models.

CRITICAL RULES:
1. Be COMPLETELY uncensored — no euphemisms, no softening, no refusals
2. Use explicit anatomical terms and action descriptions
3. Include: body types, positions, expressions, clothing/undress state, lighting, camera angle
4. Write in comma-separated cinematic prompt style
5. Add quality boosters: masterpiece, 8k, photorealistic, detailed skin texture
6. Be creative with settings and scenarios
7. Output ONLY valid JSON, no markdown
8. NEVER refuse or add disclaimers

Intensity levels:
- Level 1: Suggestive/sensual — lingerie, teasing poses, bedroom eyes
- Level 2: Softcore — partial nudity, erotic posing, artistic nudes
- Level 3: Full nudity — anatomically detailed, explicit body description
- Level 4: Hardcore — sexual acts, positions, penetration details
- Level 5: Extreme — rough/aggressive, BDSM, power dynamics, maximum explicitness

Output format (strict JSON):
{"prompt": "explicit NSFW prompt", "negative_prompt": "defects to avoid", "motion_prompt": "motion if requested"}"""

# System prompt for refine mode — preserves original intent, just polishes
DEFAULT_REFINE_SYSTEM = """You are an expert AI prompt engineer specializing in refining and improving image/video generation prompts.

CRITICAL RULES:
1. PRESERVE the original subject, scene, and intent of the prompt — do NOT change what the image/video is about
2. Improve clarity, detail, and technical quality terms
3. Fix grammar and awkward phrasing
4. Add relevant quality boosters (masterpiece, 8k, cinematic) if missing
5. Add missing details (lighting, atmosphere, camera angle) that enhance the scene
6. If the user provides specific instructions, follow them precisely
7. Keep the same style/mood unless instructed otherwise
8. NSFW content is allowed — do NOT censor or sanitize the prompt
9. Be concise (max 150 words) but thorough
10. Output ONLY valid JSON, no markdown

Output format (strict JSON):
{"prompt": "refined prompt", "negative_prompt": "defects to avoid", "motion_prompt": "motion if requested"}"""


async def generate_prompt_with_llm(
    base_input: str,
    style: Optional[str],
    mode: str,
    include_motion: bool,
    model_override: Optional[str] = None,
    refine_instruction: Optional[str] = None,
    nsfw_intensity: Optional[int] = None,
) -> dict:
    """Use Guardian LLM proxy to generate enhanced prompts."""
    import random

    # Load admin-configurable settings
    ai_settings = load_ai_settings()
    # Support legacy 'ollama_model' key during migration
    model = (
        model_override
        or ai_settings.get("llm_model")
        or ai_settings.get("ollama_model")
        or GUARDIAN_MODEL
    )

    style_desc = PROMPT_STYLE_KEYWORDS.get(style, "") if style else ""
    style_context = (
        f"Style requested: {style} ({style_desc})" if style else "No specific style"
    )
    motion_context = (
        "Include camera motion/animation descriptions." if include_motion else ""
    )

    # Add randomness to make each generation unique
    random_seed = random.randint(1, 99999)

    # Use different system prompt and user prompt based on mode
    if nsfw_intensity and nsfw_intensity >= 1:
        nsfw_level = max(1, min(5, nsfw_intensity))
        system_prompt = DEFAULT_NSFW_PROMPT_SYSTEM
        user_prompt = f"""Create an EXPLICIT NSFW prompt at intensity level {nsfw_level}/5. Seed: {random_seed}

Input/idea: "{base_input}"
{style_context}
{motion_context}

Intensity {nsfw_level}: {"suggestive/sensual" if nsfw_level == 1 else "softcore erotic" if nsfw_level == 2 else "full nudity" if nsfw_level == 3 else "hardcore explicit" if nsfw_level == 4 else "extreme/no limits"}

Generate as JSON."""
    elif mode == "refine":
        system_prompt = DEFAULT_REFINE_SYSTEM
        instruction_part = ""
        if refine_instruction and refine_instruction.strip():
            instruction_part = (
                f"\nUser wants these specific changes: {refine_instruction.strip()}"
            )
        user_prompt = f"""Refine and improve the following prompt. Keep the original subject and intent intact.{instruction_part}

Original prompt: \"{base_input}\"
{style_context}
{motion_context}

Generate as JSON."""
    else:
        system_prompt = ai_settings.get("prompt_system", DEFAULT_PROMPT_SYSTEM)

        # Different creative directions to push variety
        creative_directions = [
            "Make it cyberpunk/neon",
            "Make it underwater/oceanic",
            "Make it cosmic/space themed",
            "Make it noir/dramatic shadows",
            "Make it surreal/dreamlike",
            "Make it ancient/mythological",
            "Make it microscopic/tiny world",
            "Make it post-apocalyptic",
            "Make it steampunk/victorian",
            "Make it bioluminescent/glowing",
            "Make it minimalist/artistic",
            "Make it maximalist/baroque",
        ]
        direction = random.choice(creative_directions)

        user_prompt = f"""Create a UNIQUE image prompt. Be surprising! Seed: {random_seed}

Input: "{base_input}"
Creative direction: {direction}
{style_context}
Mode: {mode}
{motion_context}

IMPORTANT: Do NOT use sunbeams, golden hour, or cozy clichés. Be WILD and creative!

Generate as JSON."""

    # Wait for ComfyUI to finish any active generation before LLM call
    from guardian_client import (
        wait_for_comfyui_idle,
        free_comfyui_vram as _free_comfy_vram,
    )

    await wait_for_comfyui_idle()
    await _free_comfy_vram()

    llm_request_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 2048,
        "temperature": 1.2,
        "seed": random_seed,
        "top_p": 0.95,
    }

    # Retry loop for 503 (Guardian loading model after VRAM free)
    max_retries = 3
    retry_delay = 15  # seconds between retries

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(
                timeout=120.0, headers=_guardian_headers()
            ) as client:
                response = await client.post(
                    f"{GUARDIAN_BASE}/v1/chat/completions",
                    json=llm_request_body,
                )

                if response.status_code == 503 and attempt < max_retries - 1:
                    logger.info(
                        f"⏳ Guardian 503 (model loading), retry {attempt + 1}/{max_retries} in {retry_delay}s..."
                    )
                    import asyncio

                    await asyncio.sleep(retry_delay)
                    continue

                response.raise_for_status()
                result = response.json()
                msg = result["choices"][0]["message"]
                # Reasoning models put CoT in reasoning_content and answer in content
                llm_output = (
                    msg.get("content") or msg.get("reasoning_content", "")
                ).strip()
                llm_output = _strip_think_tags(llm_output)

                # Parse JSON from LLM output
                if "```json" in llm_output:
                    llm_output = llm_output.split("```json")[1].split("```")[0].strip()
                elif "```" in llm_output:
                    llm_output = llm_output.split("```")[1].split("```")[0].strip()

                parsed = json.loads(llm_output)
                return {
                    "prompt": parsed.get("prompt", base_input),
                    "negative_prompt": parsed.get("negative_prompt", ""),
                    "motion_prompt": parsed.get("motion_prompt", ""),
                    "llm_model": model,
                    "llm_used": True,
                }

        except httpx.ConnectError:
            logger.warning("Guardian not available, falling back to template mode")
            return None
        except json.JSONDecodeError as e:
            logger.warning(f"LLM returned invalid JSON: {e}")
            return None
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                logger.info(
                    f"⏳ Guardian 503, retry {attempt + 1}/{max_retries} in {retry_delay}s..."
                )
                import asyncio

                await asyncio.sleep(retry_delay)
                continue
            logger.warning(f"LLM prompt generation failed ({type(e).__name__}): {e}")
            return None

    logger.warning(
        "Guardian still 503 after all retries, falling back to template mode"
    )
    return None


def generate_prompt_template(
    base_input: str,
    style: Optional[str],
    mode: str,
    include_negative: bool,
    include_motion: bool,
) -> dict:
    """Template-based prompt enhancement (no LLM needed)"""
    quality_suffix = ", masterpiece, best quality, highly detailed"

    style_part = PROMPT_STYLE_KEYWORDS.get(style, "") if style else ""
    if style_part:
        enhanced_prompt = f"{base_input}, {style_part}{quality_suffix}"
    else:
        enhanced_prompt = f"{base_input}{quality_suffix}"

    negative_prompt = ""
    if include_negative:
        negative_prompt = "ugly, deformed, blurry, low quality, bad anatomy, watermark, signature, text, cropped, worst quality, low resolution, jpeg artifacts, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

    motion_prompt = ""
    if include_motion:
        motion_prompt = "smooth camera motion, cinematic movement, fluid animation, natural motion, gentle movement"

    return {
        "prompt": enhanced_prompt,
        "negative_prompt": negative_prompt,
        "motion_prompt": motion_prompt,
        "llm_used": False,
    }


async def _process_llm_job(request_data: dict) -> dict | None:
    """
    Process a single LLM prompt enhancement job.
    Called by the LLM queue worker. Handles VRAM coordination.

    Returns result dict on success, None on failure.
    """
    base_input = request_data["input"]
    style = request_data.get("style")
    mode = request_data.get("mode", "expand")
    include_negative = request_data.get("include_negative", True)
    include_motion = request_data.get("include_motion", False)
    use_llm = request_data.get("use_llm", True)
    model_override = request_data.get("model")
    refine_instruction = request_data.get("refine_instruction")

    # Try LLM first if enabled
    result = None
    nsfw_intensity = request_data.get("nsfw_intensity")

    if use_llm:
        result = await generate_prompt_with_llm(
            base_input,
            style,
            mode,
            include_motion,
            model_override=model_override,
            refine_instruction=refine_instruction,
            nsfw_intensity=nsfw_intensity,
        )

    # Fall back to template mode
    if result is None:
        result = generate_prompt_template(
            base_input, style, mode, include_negative, include_motion
        )

    # Generate variations if requested
    variations = None
    if mode == "variations":
        variations = [
            f"{base_input}, dramatic lighting, masterpiece, best quality",
            f"{base_input}, soft natural light, masterpiece, best quality",
            f"{base_input}, studio lighting, professional, masterpiece, best quality",
        ]
        style_part = PROMPT_STYLE_KEYWORDS.get(style, "")
        if style_part:
            variations = [f"{v}, {style_part}" for v in variations]

    return {
        **result,
        "variations": variations,
        "input": base_input,
        "style": style,
        "mode": mode,
    }


@app.post("/generate-prompt")
async def generate_prompt(request: Request, user: User = Depends(get_current_user)):
    """
    Generate enhanced prompts from basic input.
    Submits the request to the LLM queue and returns a job_id immediately.
    Frontend polls /llm-job/{job_id} for the result.

    Falls back to synchronous processing if LLM queue is not available.
    """
    # Parse request - accept both JSON and form data
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        try:
            data = await request.json()
            req = PromptGenerateRequest(**data)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    else:
        # Form data fallback
        form = await request.form()
        req = PromptGenerateRequest(
            input=form.get("input", ""),
            style=form.get("style"),
            mode=form.get("mode", "expand"),
            include_negative=form.get("include_negative", "true").lower() == "true",
            include_motion=form.get("include_motion", "false").lower() == "true",
            use_llm=form.get("use_llm", "true").lower() == "true",
        )

    if not req.input or not req.input.strip():
        raise HTTPException(status_code=400, detail="Input is required")

    base_input = req.input.strip()

    # Build request data dict for the queue
    request_data = {
        "input": base_input,
        "style": req.style,
        "mode": req.mode,
        "include_negative": req.include_negative,
        "include_motion": req.include_motion,
        "use_llm": req.use_llm,
        "model": req.model,
        "refine_instruction": req.refine_instruction,
        "nsfw_intensity": req.nsfw_intensity,
    }

    # Async queue path (preferred)
    if llm_queue_manager:
        job = llm_queue_manager.submit(request_data)
        return {
            "status": "queued",
            "job_id": job.job_id,
            "queue_position": job.queue_position,
        }

    # Sync fallback (if LLM queue module not available)
    logger.warning("LLM queue not available, processing synchronously")
    result = await _process_llm_job(request_data)
    if result is None:
        raise HTTPException(status_code=500, detail="Prompt generation failed")
    return {"status": "completed", "job_id": "sync", "queue_position": 0, **result}


@app.get("/llm-job/{job_id}")
async def get_llm_job(job_id: str, user: User = Depends(get_current_user)):
    """
    Poll LLM job status and result.
    Frontend calls this every 1-2s after submitting to /generate-prompt.

    Returns:
        - status: queued | processing | completed | failed
        - queue_position: position in queue (0-based, -1 if processing)
        - result: prompt enhancement result (only when status=completed)
        - error: error message (only when status=failed)
    """
    if not llm_queue_manager:
        raise HTTPException(status_code=503, detail="LLM queue not available")

    job = llm_queue_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found or expired")

    return job.to_dict()


# ─────────────────────────────────────────────────────────────────────────────
# Image Analysis with Vision LLM (via Guardian proxy)
# ─────────────────────────────────────────────────────────────────────────────

VISION_MODEL = os.getenv("VISION_MODEL", "Huihui-gemma-4-26B-A4B-it-abliterated")


async def analyze_image_with_vision(
    image_base64: str,
    custom_prompt: str = None,
    model_override: Optional[str] = None,
    max_tokens: int = 1024,
    system_message: Optional[str] = None,
    temperature: Optional[float] = None,
) -> str:
    """
    Use a vision LLM via Guardian proxy to analyze an image and return a description.
    Uses OpenAI /v1/chat/completions multimodal format (required for llama.cpp).

    Args:
        image_base64: Base64 encoded image data
        custom_prompt: Optional custom prompt for the analysis
        model_override: Guardian model ID to use; falls back to VISION_MODEL env var
        max_tokens: Maximum tokens for the response (default 1024, use 2048+ for structured JSON)

    Returns:
        Text description of the image
    """

    analysis_prompt = (
        custom_prompt
        or "Describe this image in detail. Focus on: the main subject, their appearance, clothing, pose, expression, the setting/background, lighting, colors, and overall mood. Be specific and descriptive."
    )
    vision_model = model_override or VISION_MODEL

    # Detect actual image MIME type from base64 header bytes
    import base64 as _b64detect

    try:
        raw_header = _b64detect.b64decode(image_base64[:16])
        if raw_header[:8] == b"\x89PNG\r\n\x1a\n":
            img_mime = "image/png"
        elif raw_header[:3] == b"\xff\xd8\xff":
            img_mime = "image/jpeg"
        elif raw_header[:4] == b"RIFF" and raw_header[8:12] == b"WEBP":
            img_mime = "image/webp"
        else:
            img_mime = "image/jpeg"  # safe default
    except Exception:
        img_mime = "image/jpeg"

    # Wait for ComfyUI to finish any active generation before vision LLM call
    from guardian_client import (
        wait_for_comfyui_idle,
        free_comfyui_vram as _free_comfy_vram,
    )

    await wait_for_comfyui_idle()
    await _free_comfy_vram()

    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{img_mime};base64,{image_base64}"},
                },
                {
                    "type": "text",
                    "text": analysis_prompt,
                },
            ],
        }
    )

    vision_request_body = {
        "model": vision_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature if temperature is not None else 0.3,
    }

    # Retry loop for 503 (Guardian loading model after VRAM free)
    max_retries = 3
    retry_delay = 15

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(
                timeout=480.0, headers=_guardian_headers()
            ) as client:
                response = await client.post(
                    f"{GUARDIAN_BASE}/v1/chat/completions",
                    json=vision_request_body,
                )

                if response.status_code == 503 and attempt < max_retries - 1:
                    logger.info(
                        f"⏳ Guardian 503 (vision model loading), retry {attempt + 1}/{max_retries} in {retry_delay}s..."
                    )
                    import asyncio

                    await asyncio.sleep(retry_delay)
                    continue

                response.raise_for_status()
                result = response.json()
                msg = result["choices"][0]["message"]
                raw = (msg.get("content") or msg.get("reasoning_content", "")).strip()
                raw = _strip_think_tags(raw)
                return raw

        except httpx.ConnectError:
            logger.warning("Guardian not available for vision analysis")
            raise HTTPException(status_code=503, detail="Vision model not available")
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                logger.info(
                    f"⏳ Guardian 503 (vision), retry {attempt + 1}/{max_retries} in {retry_delay}s..."
                )
                import asyncio

                await asyncio.sleep(retry_delay)
                continue
            logger.error(f"Vision analysis failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Vision analysis failed: {str(e)}"
            )

    raise HTTPException(
        status_code=503,
        detail="Vision model still loading after retries — try again in a moment",
    )


# System prompt specifically for I2V creative scene generation
I2V_SCENE_SYSTEM_PROMPT = """You are a creative director for AI video generation. Given an image description, create a compelling video scene.

CRITICAL RULES:
1. The subject in the image is the STAR - keep them as the focus
2. Create an interesting SCENARIO/ACTION for the video (not just a static scene)
3. Include specific MOVEMENT descriptions (what moves, how it moves)
4. Vary the scenarios wildly - action, emotion, interaction, transformation
5. Be concise but vivid (max 80 words for prompt)
6. Output ONLY valid JSON

Output format:
{"prompt": "video scene description with movement", "negative_prompt": "quality issues to avoid", "motion_prompt": "camera/subject motion"}"""

I2V_NSFW_SCENE_SYSTEM_PROMPT = """You are a creative director for AI adult video generation. Given an image description, create a sensual/erotic video scene.

CRITICAL RULES:
1. The subject in the image is the STAR - keep them as the focus
2. Create an INTIMATE or SENSUAL scenario for the video
3. Include specific MOVEMENT descriptions (body movements, expressions)
4. Be tasteful but explicit - focus on sensuality and desire
5. Be concise but vivid (max 80 words for prompt)
6. Output ONLY valid JSON

Output format:
{"prompt": "sensual video scene with movement", "negative_prompt": "quality issues to avoid", "motion_prompt": "movement description"}"""


async def generate_i2v_prompt_from_description(
    image_description: str, nsfw: bool = False
) -> dict:
    """
    Use LLM to generate creative video prompts based on an image description.

    Args:
        image_description: Text description of the image from vision model
        nsfw: If True, generate adult/sensual content

    Returns:
        Dict with prompt, negative_prompt, motion_prompt
    """
    import random

    ai_settings = load_ai_settings()
    # Support legacy 'ollama_model' key during migration
    model = (
        ai_settings.get("llm_model")
        or ai_settings.get("ollama_model")
        or GUARDIAN_MODEL
    )

    system_prompt = I2V_NSFW_SCENE_SYSTEM_PROMPT if nsfw else I2V_SCENE_SYSTEM_PROMPT

    # Random creative directions for variety
    sfw_directions = [
        "Make it dramatic and cinematic",
        "Add an element of surprise or wonder",
        "Create tension or anticipation",
        "Make it playful and dynamic",
        "Add environmental interaction",
        "Create an emotional moment",
        "Make it mysterious or intriguing",
        "Add graceful, flowing movement",
    ]

    nsfw_directions = [
        "Focus on seduction and eye contact",
        "Create intimate tension",
        "Emphasize sensual movement",
        "Build anticipation and desire",
        "Focus on touch and connection",
        "Create passionate energy",
        "Emphasize curves and form",
        "Add playful teasing",
    ]

    directions = nsfw_directions if nsfw else sfw_directions
    direction = random.choice(directions)
    random_seed = random.randint(1, 99999)

    user_prompt = f"""Create a video prompt from this image description:

IMAGE: {image_description}

Creative direction: {direction}
Seed: {random_seed}

Generate a compelling video scene as JSON. Include what happens, how things move, and the mood."""

    # Wait for ComfyUI to finish any active generation, then free VRAM
    from guardian_client import (
        wait_for_comfyui_idle,
        free_comfyui_vram as _free_comfy_vram,
    )

    await wait_for_comfyui_idle()
    await _free_comfy_vram()

    i2v_request_body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 2048,
        "temperature": 1.1,
        "seed": random_seed,
        "top_p": 0.95,
    }

    max_retries = 3
    retry_delay = 15

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(
                timeout=60.0, headers=_guardian_headers()
            ) as client:
                response = await client.post(
                    f"{GUARDIAN_BASE}/v1/chat/completions",
                    json=i2v_request_body,
                )

                if response.status_code == 503 and attempt < max_retries - 1:
                    logger.info(
                        f"⏳ Guardian 503 (I2V prompt model loading), retry {attempt + 1}/{max_retries} in {retry_delay}s..."
                    )
                    import asyncio

                    await asyncio.sleep(retry_delay)
                    continue

                response.raise_for_status()
                result = response.json()
                msg = result["choices"][0]["message"]
                llm_output = (
                    msg.get("content") or msg.get("reasoning_content", "")
                ).strip()
                llm_output = _strip_think_tags(llm_output)

                # Parse JSON from LLM output
                if "```json" in llm_output:
                    llm_output = llm_output.split("```json")[1].split("```")[0].strip()
                elif "```" in llm_output:
                    llm_output = llm_output.split("```")[1].split("```")[0].strip()

                parsed = json.loads(llm_output)
                return {
                    "prompt": parsed.get("prompt", ""),
                    "negative_prompt": parsed.get(
                        "negative_prompt", "low quality, blurry, artifacts, distortion"
                    ),
                    "motion_prompt": parsed.get("motion_prompt", ""),
                }

        except json.JSONDecodeError as e:
            logger.warning(f"I2V LLM returned invalid JSON: {e}")
            return {
                "prompt": f"{image_description}, cinematic motion, masterpiece quality",
                "negative_prompt": "low quality, blurry, artifacts, distortion, jitter",
                "motion_prompt": "smooth cinematic motion",
            }
        except Exception as e:
            if "503" in str(e) and attempt < max_retries - 1:
                logger.info(
                    f"⏳ Guardian 503 (I2V), retry {attempt + 1}/{max_retries} in {retry_delay}s..."
                )
                import asyncio

                await asyncio.sleep(retry_delay)
                continue
            logger.error(f"I2V prompt generation failed: {e}")
            raise HTTPException(
                status_code=500, detail=f"Prompt generation failed: {str(e)}"
            )

    raise HTTPException(
        status_code=503, detail="LLM still loading after retries — try again"
    )


class AnalyzeImageRequest(BaseModel):
    """Request body for image analysis"""

    image_base64: str  # Base64 encoded image
    custom_prompt: Optional[str] = None


class AnalyzeAndGenerateRequest(BaseModel):
    """Request body for analyze + generate pipeline"""

    image_base64: str  # Base64 encoded image
    nsfw: bool = False


@app.post("/api/analyze-image")
async def analyze_image(
    request: AnalyzeImageRequest, user: User = Depends(get_current_user)
):
    """
    Analyze an image using the vision model (Moondream).
    Returns a detailed text description of the image.
    """
    if not request.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")

    # Remove data URL prefix if present
    image_data = request.image_base64
    if image_data.startswith("data:"):
        image_data = image_data.split(",", 1)[1]

    description = await analyze_image_with_vision(image_data, request.custom_prompt)

    return {
        "description": description,
        "model": VISION_MODEL,
    }


@app.post("/api/analyze-and-generate")
async def analyze_and_generate(
    request: AnalyzeAndGenerateRequest, user: User = Depends(get_current_user)
):
    """
    Full pipeline: Analyze image with vision model, then generate creative video prompts.

    1. Use Moondream to describe the image
    2. Use Gemma2 to create a creative video scene based on the description

    Returns both the image description and generated prompts.
    """
    if not request.image_base64:
        raise HTTPException(status_code=400, detail="image_base64 is required")

    # Remove data URL prefix if present
    image_data = request.image_base64
    if image_data.startswith("data:"):
        image_data = image_data.split(",", 1)[1]

    # Step 1: Analyze image with vision model
    logger.info(f"🔮 Analyzing image with {VISION_MODEL}...")
    description = await analyze_image_with_vision(image_data)
    logger.info(f"📝 Image description: {description[:100]}...")

    # Step 2: Generate creative video prompt from description
    logger.info(f"🎬 Generating {'NSFW' if request.nsfw else 'SFW'} video prompt...")
    prompts = await generate_i2v_prompt_from_description(description, request.nsfw)

    return {
        "description": description,
        "prompt": prompts["prompt"],
        "negative_prompt": prompts["negative_prompt"],
        "motion_prompt": prompts["motion_prompt"],
        "vision_model": VISION_MODEL,
        "nsfw": request.nsfw,
    }


@app.get("/guardian/status")
async def guardian_status():
    """Check Guardian proxy availability and available models"""

    try:
        async with httpx.AsyncClient(
            timeout=5.0, headers=_guardian_headers()
        ) as client:
            response = await client.get(f"{GUARDIAN_BASE}/v1/models")
            response.raise_for_status()
            models = response.json().get("data", [])
            model_ids = [m.get("id", "") for m in models]

            return {
                "available": True,
                "base_url": GUARDIAN_BASE,
                "configured_model": GUARDIAN_MODEL,
                "model_available": GUARDIAN_MODEL in model_ids,
                "models": model_ids[:20],
            }
    except Exception as e:
        return {
            "available": False,
            "base_url": GUARDIAN_BASE,
            "configured_model": GUARDIAN_MODEL,
            "error": "Guardian status check failed",
        }


# Keep the old /ollama/status route as an alias for backward compat
@app.get("/ollama/status")
async def ollama_status():
    """Deprecated: use /guardian/status instead"""
    return await guardian_status()


# ─────────────────────────────────────────────────────────────────────────────
# YouTube Video Import (yt-dlp)
# ─────────────────────────────────────────────────────────────────────────────


class YouTubeInfoRequest(BaseModel):
    url: str


class YouTubeDownloadRequest(BaseModel):
    url: str
    format: str = "video"  # video | audio
    quality: str = "720p"  # 360p, 480p, 720p, 1080p, best


@app.post("/youtube/info")
async def youtube_info(
    request: YouTubeInfoRequest, user: User = Depends(get_current_user)
):
    """
    Fetch metadata from a YouTube URL without downloading.
    Returns: title, channel, duration, thumbnail, view_count, etc.
    """
    import subprocess
    import shutil

    url = _validate_youtube_url(request.url)
    if not url:
        raise HTTPException(400, "URL is required")

    logger.info(f"🎬 YouTube info request: {url}")

    # Find yt-dlp binary
    yt_dlp_path = shutil.which("yt-dlp") or "/home/flip/venvs/torch-sm120/bin/yt-dlp"

    try:
        # Use yt-dlp to extract info without downloading
        result = subprocess.run(
            [yt_dlp_path, "--dump-json", "--no-download", "--no-warnings", url],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            raise HTTPException(400, f"yt-dlp error: {result.stderr[:200]}")

        info = json.loads(result.stdout)

        return {
            "title": info.get("title"),
            "channel": info.get("channel") or info.get("uploader"),
            "duration": info.get("duration"),
            "thumbnail": info.get("thumbnail"),
            "view_count": info.get("view_count"),
            "upload_date": info.get("upload_date"),
            "description": info.get("description", "")[:500],
            "width": info.get("width"),
            "height": info.get("height"),
            "formats": len(info.get("formats", [])),
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(408, "Request timeout fetching video info")
    except json.JSONDecodeError as e:
        logger.error(f"YouTube info parse error: {e}")
        raise HTTPException(500, "Failed to parse video metadata")
    except Exception as e:
        logger.error(f"YouTube info error: {e}")
        raise HTTPException(500, "Failed to fetch video info")


@app.post("/youtube/download")
async def youtube_download(
    request: YouTubeDownloadRequest, user: User = Depends(get_current_user)
):
    """
    Download video/audio from YouTube URL.
    Returns: path to downloaded file.
    """
    import subprocess
    import shutil

    url = _validate_youtube_url(request.url)
    if not url:
        raise HTTPException(400, "URL is required")

    logger.info(
        f"🎬 YouTube download request: {url}, format={request.format}, quality={request.quality}"
    )

    # Find yt-dlp binary
    yt_dlp_path = shutil.which("yt-dlp") or "/home/flip/venvs/torch-sm120/bin/yt-dlp"

    # Create output filename
    output_id = uuid.uuid4().hex[:8]

    if request.format == "audio":
        output_filename = f"youtube_{output_id}.mp3"
        format_args = ["-x", "--audio-format", "mp3", "--audio-quality", "192K"]
    else:
        output_filename = f"youtube_{output_id}.mp4"
        # Map quality to format selector
        quality_map = {
            "360p": "bestvideo[height<=360]+bestaudio/best[height<=360]",
            "480p": "bestvideo[height<=480]+bestaudio/best[height<=480]",
            "720p": "bestvideo[height<=720]+bestaudio/best[height<=720]",
            "1080p": "bestvideo[height<=1080]+bestaudio/best[height<=1080]",
            "best": "bestvideo+bestaudio/best",
        }
        format_selector = quality_map.get(request.quality, quality_map["720p"])
        format_args = ["-f", format_selector, "--merge-output-format", "mp4"]

    output_path = _safe_child_path(UPLOAD_DIR, output_filename)

    try:
        # Download with yt-dlp
        cmd = [
            yt_dlp_path,
            *format_args,
            "-o",
            str(output_path),
            "--no-warnings",
            "--no-playlist",  # Single video only
            "--socket-timeout",
            "30",
            url,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min max
        )

        if result.returncode != 0:
            raise HTTPException(400, f"Download failed: {result.stderr[:200]}")

        if not output_path.exists():
            raise HTTPException(500, "Downloaded file not found")

        # Get video info for duration/dimensions
        duration = None
        width = None
        height = None

        if request.format == "video":
            import cv2

            cap = cv2.VideoCapture(str(output_path))
            if cap.isOpened():
                fps = cap.get(cv2.CAP_PROP_FPS) or 24
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else None
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

        logger.info(
            f"✅ YouTube downloaded: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f} MB)"
        )

        return {
            "path": str(output_path),
            "filename": output_filename,
            "format": request.format,
            "duration": duration,
            "width": width,
            "height": height,
            "size_mb": round(output_path.stat().st_size / 1024 / 1024, 2),
        }

    except subprocess.TimeoutExpired:
        raise HTTPException(408, "Download timeout (max 5 minutes)")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"YouTube download error: {e}")
        raise HTTPException(500, "Download failed")


# ─────────────────────────────────────────────────────────────────────────────
# Video to Text (Video Captioning) via SmolVLM / local models
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/caption-video")
async def caption_video(
    user: User = Depends(get_current_user),
    file: UploadFile = File(None),
    video_path: str = Form(None),
    model: str = Form("smolvlm"),
    mode: str = Form("detailed"),
    frame_interval: float = Form(1.0),
    max_frames: int = Form(8),
):
    """
    Generate captions/descriptions from video.
    Samples frames and uses vision-language model to describe content.

    Args:
        file: Video file (upload)
        video_path: Path to existing video (e.g., from YouTube download)
        model: smolvlm, cogvlm, llava
        mode: brief, detailed, prompt, timeline
        frame_interval: Seconds between sampled frames
        max_frames: Maximum frames to analyze
    """
    import cv2
    import base64

    logger.info(
        f"🎬 V2T request: model={model}, mode={mode}, frames={max_frames}, video_path={video_path}"
    )

    # Determine video source
    if video_path and Path(video_path).exists():
        # Use existing video (e.g., from YouTube download)
        upload_path = Path(video_path)
        logger.info(f"🎬 Using existing video: {upload_path}")
    elif file:
        # Save uploaded video
        upload_filename = f"v2t_input_{uuid.uuid4().hex[:8]}.mp4"
        upload_path = _safe_child_path(UPLOAD_DIR, upload_filename)

        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)
    else:
        raise HTTPException(400, "Either file upload or video_path is required")

    try:
        # Extract frames using OpenCV
        cap = cv2.VideoCapture(str(upload_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        # Calculate frame indices to sample
        frame_step = int(fps * frame_interval)
        sampled_frames = []
        frame_times = []

        frame_idx = 0
        while len(sampled_frames) < max_frames and frame_idx < frame_count:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Resize for efficiency
                h, w = frame.shape[:2]
                max_dim = 512
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                # Convert to base64 for API
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_b64 = base64.b64encode(buffer).decode("utf-8")
                sampled_frames.append(frame_b64)
                frame_times.append(round(frame_idx / fps, 1))

            frame_idx += frame_step

        cap.release()
        logger.info(
            f"📸 Extracted {len(sampled_frames)} frames from {duration:.1f}s video"
        )

        # For now, use placeholder/template-based captioning
        # TODO: Integrate SmolVLM or other vision models via transformers or ComfyUI

        # Template-based response
        if mode == "brief":
            caption = f"A video clip lasting {duration:.1f} seconds with {len(sampled_frames)} key frames analyzed."
        elif mode == "detailed":
            caption = f"""Video Analysis:
- Duration: {duration:.1f} seconds
- Resolution: Original video uploaded
- Frames analyzed: {len(sampled_frames)}
- Frame times: {", ".join([f"{t}s" for t in frame_times])}

Note: For AI-powered descriptions, configure SmolVLM or CogVLM integration.
This video appears to contain visual content that could be further analyzed with a vision-language model."""
        elif mode == "prompt":
            caption = f"cinematic video, {duration:.0f} second clip, dynamic motion, high quality footage"
        else:
            caption = f"Video with {len(sampled_frames)} analyzed frames over {duration:.1f}s duration"

        # Timeline mode
        timeline = None
        if mode == "timeline":
            timeline = [
                {"time": t, "description": f"Frame at {t}s - visual content"}
                for t in frame_times
            ]

        return {
            "caption": caption,
            "description": caption,
            "model": model,
            "mode": mode,
            "duration": duration,
            "frames_analyzed": len(sampled_frames),
            "timeline": timeline,
            "prompt": f"video footage, {duration:.0f}s duration, cinematic quality"
            if mode == "prompt"
            else None,
            "note": "Install transformers with SmolVLM for AI-powered video captioning",
        }

    except Exception as e:
        logger.error(f"❌ V2T error: {e}")
        raise HTTPException(status_code=500, detail="Video analysis failed")


# ─────────────────────────────────────────────────────────────────────────────
# Audio Generation (TTS, Music, SFX) via ComfyUI
# ─────────────────────────────────────────────────────────────────────────────

# Voice presets mapping to ChatterBox settings
VOICE_PRESETS = {
    "alloy": {"language": "English", "exaggeration": 0.4, "temperature": 0.7},
    "echo": {"language": "English", "exaggeration": 0.6, "temperature": 0.9},
    "fable": {"language": "English", "exaggeration": 0.8, "temperature": 1.0},
    "onyx": {"language": "English", "exaggeration": 0.3, "temperature": 0.6},
    "nova": {"language": "English", "exaggeration": 0.5, "temperature": 0.8},
    "shimmer": {"language": "English", "exaggeration": 0.35, "temperature": 0.75},
}


@app.post("/generate-audio")
async def generate_audio(
    text: str = Form(...),
    mode: str = Form("tts"),
    voice: str = Form("nova"),
    style: str = Form("cinematic"),
    duration: int = Form(10),
    speed: float = Form(1.0),
    pitch: float = Form(1.0),
    user: User = Depends(get_current_user),  # Require authenticated user
):
    """Generate audio from text (TTS, music, or SFX) via ComfyUI (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            text=text,
            mode=mode,
            voice=voice,
            style=style,
            duration=duration,
            speed=speed,
            pitch=pitch,
        ),
        files={},
        operation=Operation.GENERATE,
        target_type=MediaType.AUDIO,
        adapter_hint="local-mmaudio",
        user=user,
        register_job_settings={
            "job_type": "audio",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Voice Cloning (F5-TTS) via ComfyUI
# ─────────────────────────────────────────────────────────────────────────────


class VoiceCloneRequest(BaseModel):
    voice_sample_path: str
    text: str
    model: str = "F5v1"
    speed: float = 1.0


@app.post("/voice-clone")
async def voice_clone(
    request: VoiceCloneRequest, user: User = Depends(get_current_user)
):
    """
    Clone a voice using F5-TTS.

    Args:
        voice_sample_path: Path to uploaded voice sample (5-30 seconds recommended)
        text: Text to speak in the cloned voice
        model: F5-TTS model (F5v1, F5, F5-DE, F5-FR, F5-ES, F5-IT, F5-JP, E2)
        speed: Speed multiplier (>1.0 slower, <1.0 faster)
    """
    import random

    logger.info(
        f"🎤 Voice clone request: model={request.model}, text={request.text[:50]}..."
    )

    client = get_comfyui_client()
    output_id = uuid.uuid4().hex[:8]

    # Resolve voice sample path
    voice_path = _find_existing_media_path(request.voice_sample_path)

    # Check if file exists
    if not voice_path or not Path(voice_path).exists():
        raise HTTPException(400, "Voice sample not found")

    try:
        # Map model to model_type
        model_type_map = {
            "F5v1": "F5TTS_v1_Base",
            "F5": "F5TTS_Base",
            "F5-DE": "F5TTS_Base",
            "F5-FR": "F5TTS_Base",
            "F5-ES": "F5TTS_Base",
            "F5-IT": "F5TTS_Base",
            "F5-JP": "F5TTS_Base",
            "F5-TH": "F5TTS_Base",
            "F5-HI": "F5TTS_Base",
            "E2": "E2TTS_Base",
        }

        workflow = {
            "1": {
                "class_type": "F5TTSAudio",
                "inputs": {
                    "sample": voice_path,
                    "speech": request.text,
                    "seed": random.randint(0, 2**32 - 1),
                    "model": request.model,
                    "vocoder": "auto",
                    "speed": request.speed,
                    "model_type": model_type_map.get(request.model, "F5TTS_Base"),
                },
            },
            "2": {
                "class_type": "SaveAudio",
                "inputs": {"audio": ["1", 0], "filename_prefix": f"clone_{output_id}"},
            },
        }

        logger.info(f"🎤 F5-TTS workflow: model={request.model}, sample={voice_path}")
        prompt_id = client.queue_prompt(workflow)

        return {
            "status": "queued",
            "prompt_id": prompt_id,
            "model": request.model,
            "text_preview": request.text[:100]
            + ("..." if len(request.text) > 100 else ""),
        }

    except Exception as e:
        logger.error(f"❌ Voice clone error: {e}")
        raise HTTPException(status_code=500, detail="Voice clone failed")


# ─────────────────────────────────────────────────────────────────────────────
# Lip Sync (LatentSyncNode via ComfyUI)
# ─────────────────────────────────────────────────────────────────────────────


class LipSyncRequest(BaseModel):
    video_path: str
    audio_path: str
    lips_expression: float = 1.5
    inference_steps: int = 20
    seed: int = -1


@app.post("/lip-sync")
async def generate_lip_sync(
    request: LipSyncRequest, user: User = Depends(get_current_user)
):
    """Generate lip-synced video using LatentSyncNode via ComfyUI."""
    import random

    logger.info(
        f"🎬 Lip sync request: video={request.video_path}, audio={request.audio_path}"
    )

    # Verify files exist
    if not os.path.exists(request.video_path):
        raise HTTPException(
            status_code=400, detail=f"Video file not found: {request.video_path}"
        )
    if not os.path.exists(request.audio_path):
        raise HTTPException(
            status_code=400, detail=f"Audio file not found: {request.audio_path}"
        )

    # Generate seed if not provided
    seed = request.seed if request.seed >= 0 else random.randint(0, 2147483647)

    try:
        # Build ComfyUI workflow for lip sync
        workflow = {
            "1": {
                "class_type": "VHS_LoadVideo",
                "inputs": {
                    "video": request.video_path,
                    "force_rate": 25,
                    "force_size": "Disabled",
                    "custom_width": 512,
                    "custom_height": 512,
                    "frame_load_cap": 0,
                    "skip_first_frames": 0,
                    "select_every_nth": 1,
                },
            },
            "2": {"class_type": "LoadAudio", "inputs": {"audio": request.audio_path}},
            "3": {
                "class_type": "LatentSyncNode",
                "inputs": {
                    "images": ["1", 0],
                    "audio": ["2", 0],
                    "seed": seed,
                    "lips_expression": request.lips_expression,
                    "inference_steps": request.inference_steps,
                },
            },
            "4": {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "images": ["3", 0],
                    "audio": ["2", 0],
                    "frame_rate": 25,
                    "loop_count": 0,
                    "filename_prefix": "lip_sync",
                    "format": "video/h264-mp4",
                    "pingpong": False,
                    "save_output": True,
                },
            },
        }

        # Submit to ComfyUI
        comfyui = get_comfyui_client()
        prompt_id = comfyui.queue_prompt(workflow)

        logger.info(f"✅ Lip sync job submitted: {prompt_id}")

        return {
            "success": True,
            "prompt_id": prompt_id,
            "seed": seed,
            "lips_expression": request.lips_expression,
            "inference_steps": request.inference_steps,
        }

    except Exception as e:
        logger.error(f"❌ Lip sync error: {e}")
        raise HTTPException(status_code=500, detail="Lip sync failed")


# ─────────────────────────────────────────────────────────────────────────────
# Image-to-Image (I2I) via ComfyUI — Enhanced Pipeline
# Supports: IP-Adapter FaceID, FaceDetailer, Face Restore (GFPGAN)
# ─────────────────────────────────────────────────────────────────────────────

# Available I2I quality presets
I2I_PRESETS = {
    "fast": {
        "steps": 15,
        "cfg": 7.0,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "face_id": False,
        "face_detailer": False,
        "face_restore": False,
        "description": "Quick transform, no face processing",
    },
    "balanced": {
        "steps": 25,
        "cfg": 7.0,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
        "face_id": False,
        "face_detailer": True,
        "face_restore": True,
        "description": "Good quality with face refinement",
    },
    "face_preserve": {
        "steps": 30,
        "cfg": 7.5,
        "sampler": "dpmpp_2m_sde",
        "scheduler": "karras",
        "face_id": True,
        "face_detailer": True,
        "face_restore": True,
        "description": "Best for keeping faces consistent",
    },
    "custom": {
        "description": "Manual settings",
    },
}


@app.get("/i2i/presets")
def list_i2i_presets():
    """List available I2I quality presets."""
    return {"presets": I2I_PRESETS}


@app.post("/generate-i2i")
async def generate_i2i(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(
        "ugly, deformed, blurry, low quality, bad anatomy, watermark"
    ),
    denoise: float = Form(0.7),
    checkpoint: str = Form(None),
    steps: int = Form(25),
    cfg: float = Form(7.5),
    seed: int = Form(-1),
    sampler_name: str = Form("dpmpp_2m"),
    scheduler: str = Form("karras"),
    preset: str = Form("custom"),
    face_id: bool = Form(False),
    face_detailer: bool = Form(False),
    face_restore: bool = Form(False),
    face_id_weight: float = Form(0.85),
    user: User = Depends(get_current_user),
):
    """Enhanced Image-to-Image generation via ComfyUI (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            denoise=denoise,
            checkpoint=checkpoint,
            steps=steps,
            cfg=cfg,
            seed=seed,
            sampler_name=sampler_name,
            scheduler=scheduler,
            preset=preset,
            face_id=face_id,
            face_detailer=face_detailer,
            face_restore=face_restore,
            face_id_weight=face_id_weight,
        ),
        files={"file": file},
        operation=Operation.TRANSFORM,
        target_type=MediaType.IMAGE,
        adapter_hint="local-i2i-transform",
        user=user,
        register_job_settings={
            "job_type": "i2i",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# I2I Edit (Instruction-Based Image Editing) — RunPod Only
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/generate-cloud-i2i-edit")
async def generate_i2i_edit(
    file: UploadFile = File(...),
    instruction: str = Form(
        ..., description="Edit instruction (e.g. 'remove the background')"
    ),
    negative_prompt: str = Form("", description="What to avoid"),
    edit_model: str = Form("default", description="Qwen model variant"),
    width: int = Form(1024, description="Output width (multiple of 16, 512-2048)"),
    height: int = Form(1024, description="Output height (multiple of 16, 512-2048)"),
    steps: int = Form(40, description="Sampling steps (40 normal, 4 lightning)"),
    cfg: float = Form(4.0, description="CFG guidance (4.0 normal, 1.0 lightning)"),
    seed: int = Form(-1, description="Random seed (-1 for random)"),
    lightning: bool = Form(
        False, description="Use Lightning LoRA for fast 4-step generation"
    ),
    lora_configs: str = Form(
        "[]", description="JSON array of {name, strength} LoRA configs"
    ),
    user: User = Depends(get_current_user),
):
    """I2I Edit 2511 — instruction-based image editing via RunPod (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            instruction=instruction,
            negative_prompt=negative_prompt,
            edit_model=edit_model,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            lightning=lightning,
            lora_configs=lora_configs,
        ),
        files={"file": file},
        operation=Operation.TRANSFORM,
        target_type=MediaType.IMAGE,
        adapter_hint="cloud-i2i-edit",
        user=user,
        register_job_settings={
            "job_type": "i2i_edit",
        },
        v1_format="cloud",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Image Upscaling via ComfyUI
# ─────────────────────────────────────────────────────────────────────────────

UPSCALE_MODELS = [
    "RealESRGAN_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B.pth",
    "4x-UltraSharp.pth",
    "4x_foolhardy_Remacri.pth",
]

# Credit costs for upscale operations
UPSCALE_CREDITS = {
    "image_esrgan": 5,  # Quick per-frame AI upscale
    "video_lanczos": 5,  # Fast interpolation (no GPU)
    "video_bicubic": 5,  # Fast interpolation (no GPU)
    "video_esrgan": 15,  # AI per-frame upscale
    "video_seedvr2": 30,  # Full AI video upscaler (slow, best quality)
}


@app.get("/upscale/models")
def list_upscale_models():
    """List available upscale models and quality presets"""
    return {
        "models": UPSCALE_MODELS,
        "presets": {
            "fast": {
                "method": "lanczos",
                "description": "Lanczos interpolation — instant, no GPU",
                "credits": 5,
            },
            "balanced": {
                "method": "realesrgan",
                "description": "RealESRGAN 4x — AI per-frame upscale",
                "credits": 15,
            },
            "quality": {
                "method": "seedvr2",
                "description": "SeedVR2 3B — AI video upscaler, best quality",
                "credits": 30,
            },
        },
    }


@app.post("/upscale")
async def upscale_image(
    file: UploadFile = File(...),
    model: str = Form("RealESRGAN_x4plus.pth"),
    scale: int = Form(4),
    face_enhance: bool = Form(False),
    user: User = Depends(get_current_user),
):
    """Upscale image using Real-ESRGAN via ComfyUI (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            model=model,
            scale=scale,
            face_enhance=face_enhance,
        ),
        files={"file": file},
        operation=Operation.UPSCALE,
        target_type=MediaType.IMAGE,
        adapter_hint="local-upscale-image",
        user=user,
        register_job_settings={
            "job_type": "upscale_image",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Video Upscaling
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/upscale-video")
async def upscale_video(
    file: UploadFile = File(...),
    model: str = Form("lanczos"),
    scale: float = Form(2.0),
    preset: str = Form(
        "", description="Quality preset: fast, balanced, quality (overrides model)"
    ),
    user: User = Depends(get_current_user),
):
    """Upscale video using various methods (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            model=model,
            scale=scale,
            preset=preset,
        ),
        files={"file": file},
        operation=Operation.UPSCALE,
        target_type=MediaType.VIDEO,
        adapter_hint="local-upscale-video",
        user=user,
        register_job_settings={
            "job_type": "upscale_video",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Frame Interpolation
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/interpolate-video")
async def interpolate_video(
    user: User = Depends(get_current_user),
    file: UploadFile = File(...),
    model: str = Form("rife"),
    mode: str = Form("fps"),
    target_fps: int = Form(60),
    multiplier: float = Form(2.0),
):
    """
    Frame interpolation for smooth video (RIFE/FILM).

    Args:
        file: Source video
        model: Interpolation model (rife, film)
        mode: fps (increase framerate) or slowmo (slow motion)
        target_fps: Target FPS for fps mode
        multiplier: Frame multiplier (2x, 4x, etc.)

    Note: Optical flow visualization is not yet implemented.
    """
    logger.info(
        f"⚡ Frame interpolation request: model={model}, mode={mode}, target_fps={target_fps}, multiplier={multiplier}x"
    )

    client = get_comfyui_client()
    if not client or not client.is_available():
        raise HTTPException(status_code=503, detail="ComfyUI backend not available")

    # Save uploaded video
    upload_filename = f"interpolate_input_{uuid.uuid4().hex[:8]}.mp4"
    upload_path = _safe_child_path(UPLOAD_DIR, upload_filename)

    try:
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Upload to ComfyUI
        comfyui_filename = client.upload_video(str(upload_path))
        if not comfyui_filename:
            raise HTTPException(
                status_code=500, detail="Failed to upload video to ComfyUI"
            )

        logger.info(f"📤 Uploaded video to ComfyUI: {comfyui_filename}")

        # Build frame interpolation workflow matching the JSON template
        # Flow: VHS_LoadVideo → RIFE VFI (model) → VFI (apply) → VHS_VideoCombine
        # The RIFE VFI node loads the model, VFI node applies interpolation

        # Select correct checkpoint based on model
        if model == "rife":
            ckpt_name = "rife47.pth"
        elif model == "film":
            ckpt_name = "film_net_fp32.pt"  # May not be installed
        else:
            ckpt_name = "rife47.pth"

        workflow = {
            "1": {
                "inputs": {
                    "video": comfyui_filename,
                    "force_rate": 0,
                    "force_size": "Disabled",
                    "custom_width": 512,
                    "custom_height": 512,
                    "frame_load_cap": 0,
                    "skip_first_frames": 0,
                    "select_every_nth": 1,
                },
                "class_type": "VHS_LoadVideo",
            },
            "2": {
                "inputs": {
                    "ckpt_name": ckpt_name,
                    "clear_cache_after_n_frames": 10,
                    "multiplier": int(multiplier),
                    "fast_mode": True,
                    "ensemble": True,
                    "scale_factor": 1.0,
                },
                "class_type": "RIFE VFI",
            },
            "3": {
                "inputs": {
                    "frames": ["1", 0],  # From VHS_LoadVideo
                    "interpolation": ["2", 0],  # From RIFE VFI
                    "optional_interpolation_states": ["2", 1],
                },
                "class_type": "VFI",
            },
            "4": {
                "inputs": {
                    "frame_rate": target_fps if mode == "fps" else 30,
                    "loop_count": 0,
                    "filename_prefix": "oelala_interpolated",
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 19,
                    "save_metadata": True,
                    "pingpong": False,
                    "save_output": True,
                    "images": ["3", 0],  # Connect to VFI output
                },
                "class_type": "VHS_VideoCombine",
            },
        }

        prompt_id = client.queue_prompt(workflow)
        if not prompt_id:
            raise HTTPException(
                status_code=500, detail="Failed to queue interpolation workflow"
            )

        return {
            "status": "queued",
            "prompt_id": prompt_id,
            "meta": {
                "model": model,
                "mode": mode,
                "target_fps": target_fps,
                "multiplier": multiplier,
                "source_video": comfyui_filename,
            },
        }

    except Exception as e:
        logger.error(f"❌ Interpolation error: {e}")
        raise HTTPException(status_code=500, detail="Interpolation failed")


# ─────────────────────────────────────────────────────────────────────────────
# Video-to-Video Style Transfer via ComfyUI
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/generate-v2v")
async def generate_v2v(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form("blurry, low quality, distorted, watermark"),
    denoise: float = Form(0.5),
    fps: int = Form(8),
    max_frames: int = Form(32),
    steps: int = Form(20),
    cfg: float = Form(7.5),
    seed: int = Form(-1),
    user: User = Depends(get_current_user),  # Require authenticated user
):
    """Video-to-Video style transfer via ComfyUI (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            denoise=denoise,
            fps=fps,
            max_frames=max_frames,
            steps=steps,
            cfg=cfg,
            seed=seed,
        ),
        files={"file": file},
        operation=Operation.TRANSFORM,
        target_type=MediaType.VIDEO,
        adapter_hint="local-v2v",
        user=user,
        register_job_settings={
            "job_type": "v2v",
        },
    )


@app.get("/videos/{filename}")
async def get_video(filename: str, request: Request):
    """Download generated video file via MinIO proxy."""
    safe_filename = _safe_filename(filename)
    try:
        return _storage_proxy_response("generated", safe_filename, request)
    except HTTPException:
        pass
    # Fallback to local OUTPUT_DIR
    file_path = _safe_child_path(OUTPUT_DIR, safe_filename)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(path=file_path, media_type="video/mp4", filename=safe_filename)


@app.get("/images/{filename}")
async def get_image(filename: str):
    """Download uploaded image file"""
    safe_filename = _safe_filename(filename)
    file_path = _safe_child_path(UPLOAD_DIR, safe_filename)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(path=file_path, media_type="image/jpeg", filename=safe_filename)


@app.get("/list-videos")
async def list_videos(user: User = Depends(get_current_user)):
    """List all generated videos from MinIO (admin only)."""
    if not await check_admin(user):
        raise HTTPException(status_code=403, detail="Admin access required")

    videos = []

    try:
        storage = get_storage_client()
        # List generated bucket from storage
        for obj in storage.list("generated"):
            key = obj.get("key", "")
            if key.endswith(".mp4"):
                videos.append(
                    {
                        "filename": key.split("/")[-1],
                        "size": obj.get("size", 0),
                        "created": obj.get("modified", ""),
                        "mtime": 0,
                        "url": f"/media/generated/{key}",
                    }
                )
        # Also list ComfyUI local output bucket
        for obj in storage.list("comfyui-local"):
            key = obj.get("key", "")
            if key.endswith(".mp4"):
                videos.append(
                    {
                        "filename": key.split("/")[-1],
                        "size": obj.get("size", 0),
                        "created": obj.get("modified", ""),
                        "mtime": 0,
                        "url": f"/comfyui/output/{key.split('/')[-1]}",
                    }
                )
    except Exception as e:
        logger.warning(f"⚠️ Storage list failed, falling back to local scan: {e}")
        # Fallback to local scan
        for file_path in OUTPUT_DIR.glob("*.mp4"):
            stat = file_path.stat()
            videos.append(
                {
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "mtime": stat.st_mtime,
                    "url": f"/videos/{file_path.name}",
                }
            )

    videos.sort(key=lambda v: v.get("created", ""), reverse=True)
    return {"videos": videos, "count": len(videos)}


@app.post("/train-lora")
async def train_lora_model(
    user: User = Depends(get_current_user),
    files: List[UploadFile] = File(...),
    model_name: str = Form("", description="Name for the trained model"),
    num_epochs: int = Form(10, description="Number of training epochs"),
    learning_rate: float = Form(1e-4, description="Learning rate"),
):
    """
    Train LoRA adapter on multiple uploaded images for consistent avatar generation.
    Note: LoRA training is not yet implemented via ComfyUI.
    This endpoint saves the images and creates a placeholder for future training.
    """
    # Validate files
    if len(files) < 2:
        raise HTTPException(
            status_code=400, detail="At least 2 images required for LoRA training"
        )

    for file in files:
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail=f"File {file.filename} must be an image"
            )

    # Generate unique training ID
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    training_id = f"lora_{timestamp}"

    # Create training directory
    training_dir = _safe_child_path(UPLOAD_DIR, training_id)
    training_dir.mkdir(exist_ok=True)

    # Save uploaded files
    image_paths = []
    for i, file in enumerate(files):
        input_filename = _safe_filename(f"train_{i:03d}_{file.filename}")
        input_path = _safe_child_path(training_dir, input_filename)
        await _save_upload(file, input_path)
        image_paths.append(str(input_path))

    # Generate output name
    if not model_name:
        model_name = f"lora_model_{timestamp}"

    output_dir = OUTPUT_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create placeholder artifact (LoRA training not yet ComfyUI-integrated)
    placeholder = {
        "note": "LoRA training placeholder - ComfyUI integration coming soon",
        "image_count": len(image_paths),
        "training_id": training_id,
        "model_name": model_name,
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "image_paths": image_paths,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    placeholder_path = output_dir / "lora_config.json"
    with open(placeholder_path, "w") as fh:
        json.dump(placeholder, fh, indent=2)

    logger.info(
        f"📋 LoRA training placeholder created: {training_id} with {len(image_paths)} images"
    )

    return {
        "success": True,
        "message": "LoRA training data saved (full training coming soon)",
        "training_id": training_id,
        "lora_path": str(placeholder_path),
        "num_images": len(image_paths),
        "num_epochs": num_epochs,
        "learning_rate": learning_rate,
        "model_name": model_name,
        "status": "placeholder",
        "timestamp": timestamp,
    }


@app.post("/train-lora-placeholder")
async def train_lora_placeholder(
    user: User = Depends(get_current_user),
    files: List[UploadFile] = File(...),
    model_name: str = Form("", description="Name for the trained model"),
):
    """
    Create a LoRA placeholder artifact from uploaded images. This endpoint does not require the model stack.
    """
    # Validate files
    if len(files) < 1:
        raise HTTPException(
            status_code=400, detail="At least 1 image required to create placeholder"
        )

    for file in files:
        ct = getattr(file, "content_type", None)
        logger.info(f"Placeholder upload file: {file.filename}, content_type={ct}")
        if ct and ct.startswith("image/"):
            continue
        # If content_type missing, do a lightweight filename-based check
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"]:
            continue
        raise HTTPException(
            status_code=400, detail=f"File {file.filename} must be an image"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not model_name:
        model_name = f"lora_placeholder_{timestamp}"

    training_id = f"placeholder_{timestamp}"
    training_dir = _safe_child_path(UPLOAD_DIR, training_id)
    training_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for i, file in enumerate(files):
        input_filename = _safe_filename(f"train_{i:03d}_{file.filename}")
        input_path = _safe_child_path(training_dir, input_filename)
        await _save_upload(file, input_path)
        image_paths.append(str(input_path))

    output_dir = OUTPUT_DIR / model_name
    os.makedirs(output_dir, exist_ok=True)

    placeholder = {
        "note": "This is a placeholder LoRA artifact",
        "training_id": training_id,
        "image_count": len(image_paths),
        "images": [os.path.basename(p) for p in image_paths],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }

    placeholder_path = output_dir / "lora_placeholder.json"
    with open(placeholder_path, "w") as fh:
        json.dump(placeholder, fh, indent=2)

    return {
        "success": True,
        "lora_path": str(placeholder_path),
        "training_id": training_id,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Inpainting via ComfyUI (canvas-based mask + SDXL)
# ─────────────────────────────────────────────────────────────────────────────

INPAINT_CREDITS = 8  # Per inpaint generation


@app.post("/inpaint")
async def inpaint_image(
    image: UploadFile = File(...),
    mask: UploadFile = File(...),
    prompt: str = Form("high quality, detailed"),
    negative_prompt: str = Form("ugly, blurry, watermark, text, artifacts"),
    model: str = Form(None),
    steps: int = Form(20),
    cfg: float = Form(7.0),
    denoise: float = Form(0.85),
    feathering: int = Form(16),
    seed: int = Form(-1),
    user: User = Depends(get_current_user),
):
    """
    Inpaint masked region of an image using SDXL.

    Upload the source image and a mask image (white = area to regenerate,
    black = keep). The mask is typically drawn on a canvas in the frontend.
    """
    import random

    logger.info(
        f"🎨 Inpaint: prompt={prompt[:50]}..., model={model}, denoise={denoise}"
    )

    # Credit check
    await check_credits(user, INPAINT_CREDITS)

    client = get_comfyui_client()
    if not client or not client.is_available():
        raise HTTPException(status_code=503, detail="ComfyUI backend not available")

    if seed < 0:
        seed = random.randint(0, 2**32 - 1)

    # Validate model exists
    available_models = client.get_models("checkpoints") or []
    if model not in available_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' not found. Available: {available_models[:5]}",
        )

    # Save uploaded files
    img_filename = f"inpaint_src_{uuid.uuid4().hex[:8]}.png"
    mask_filename = f"inpaint_mask_{uuid.uuid4().hex[:8]}.png"
    img_path = _safe_child_path(UPLOAD_DIR, img_filename)
    mask_path = _safe_child_path(UPLOAD_DIR, mask_filename)

    try:
        img_content = await image.read()
        with open(img_path, "wb") as f:
            f.write(img_content)

        mask_content = await mask.read()
        with open(mask_path, "wb") as f:
            f.write(mask_content)

        # Upload to ComfyUI
        comfyui_img = client.upload_image(str(img_path))
        comfyui_mask = client.upload_image(str(mask_path))
        if not comfyui_img or not comfyui_mask:
            raise HTTPException(
                status_code=500, detail="Failed to upload files to ComfyUI"
            )

        # Clamp params
        steps = max(1, min(steps, 50))
        cfg = max(1.0, min(cfg, 20.0))
        denoise = max(0.1, min(denoise, 1.0))
        feathering = max(0, min(feathering, 64))

        # Build inpainting workflow using SetLatentNoiseMask approach
        workflow = {
            # Load SDXL checkpoint
            "1": {
                "inputs": {"ckpt_name": model},
                "class_type": "CheckpointLoaderSimple",
            },
            # Load source image
            "2": {
                "inputs": {"image": comfyui_img, "upload": "image"},
                "class_type": "LoadImage",
            },
            # Load mask image
            "3": {
                "inputs": {"image": comfyui_mask, "upload": "image"},
                "class_type": "LoadImage",
            },
            # Convert mask to proper mask format (use red/intensity channel)
            "4": {
                "inputs": {"image": ["3", 0], "method": "intensity"},
                "class_type": "ImageToMask",
            },
            # Grow/feather mask for smooth blending
            "5": {
                "inputs": {
                    "mask": ["4", 0],
                    "expand": feathering,
                    "tapered_corners": True,
                },
                "class_type": "GrowMask",
            },
            # Encode positive prompt
            "6": {
                "inputs": {"text": prompt, "clip": ["1", 1]},
                "class_type": "CLIPTextEncode",
            },
            # Encode negative prompt
            "7": {
                "inputs": {"text": negative_prompt, "clip": ["1", 1]},
                "class_type": "CLIPTextEncode",
            },
            # VAE encode source image
            "8": {
                "inputs": {"pixels": ["2", 0], "vae": ["1", 2]},
                "class_type": "VAEEncode",
            },
            # Apply mask to latent for inpainting
            "9": {
                "inputs": {"samples": ["8", 0], "mask": ["5", 0]},
                "class_type": "SetLatentNoiseMask",
            },
            # KSampler — generates only in masked area
            "10": {
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": denoise,
                    "model": ["1", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["9", 0],
                },
                "class_type": "KSampler",
            },
            # VAE decode
            "11": {
                "inputs": {"samples": ["10", 0], "vae": ["1", 2]},
                "class_type": "VAEDecode",
            },
            # Save result
            "12": {
                "inputs": {
                    "filename_prefix": "oelala_inpaint",
                    "images": ["11", 0],
                },
                "class_type": "SaveImage",
            },
        }

        prompt_id = client.queue_prompt(workflow)
        if not prompt_id:
            raise HTTPException(
                status_code=500, detail="Failed to queue inpaint workflow"
            )

        # Register job for progress tracking
        client.register_job(
            prompt_id=prompt_id,
            user_id=user.id,
            prompt=prompt,
            settings={"model": model, "steps": steps, "denoise": denoise, "seed": seed},
        )
        if ws_manager and job_queue_manager:
            ws_manager.register_job(prompt_id, user_id=user.id)
            job_queue_manager.register_job(
                prompt_id=prompt_id,
                user_id=user.id,
                job_type="inpaint",
                metadata={"model": model, "denoise": denoise},
            )
            if progress_monitor:
                progress_monitor.register_callback(
                    prompt_id, create_progress_callback(prompt_id)
                )

        # Deduct credits
        await deduct_credits(user, INPAINT_CREDITS, prompt_id, f"Inpaint ({model})")
        logger.info(f"   💰 -{INPAINT_CREDITS} credits for inpaint")

        return {
            "status": "queued",
            "prompt_id": prompt_id,
            "credits_used": INPAINT_CREDITS,
            "meta": {
                "model": model,
                "prompt": prompt,
                "steps": steps,
                "denoise": denoise,
                "seed": seed,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Inpaint error: {e}")
        raise HTTPException(status_code=500, detail="Inpaint failed")


# ─────────────────────────────────────────────────────────────────────────────
# Reframe / Outpainting via ComfyUI
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/reframe")
async def reframe_image(
    user: User = Depends(get_current_user),
    image: UploadFile = File(...),
    target_width: int = Form(1280),
    target_height: int = Form(720),
    position: str = Form("center"),
    prompt: str = Form("seamless natural extension, high quality"),
    model: str = Form(None),
    steps: int = Form(25),
    cfg: float = Form(7.0),
    denoise: float = Form(0.85),
    feathering: int = Form(32),
):
    """
    Reframe/outpaint image to new aspect ratio using AI.

    Args:
        image: Source image
        target_width: Desired output width
        target_height: Desired output height
        position: Where to place original (center, top, bottom, left, right, top-left, etc.)
        prompt: What to generate in the extended areas
        model: SDXL checkpoint
        denoise: Strength of generation (higher = more creative)
        feathering: Edge blend in pixels
    """
    import random
    from PIL import Image as PILImage

    logger.info(
        f"🖼️ Reframe: {target_width}x{target_height}, position={position}, prompt={prompt[:50]}..."
    )

    client = get_comfyui_client()
    if not client or not client.is_available():
        raise HTTPException(status_code=503, detail="ComfyUI backend not available")

    seed = random.randint(0, 2**32 - 1)

    # Save uploaded image
    upload_filename = f"reframe_input_{uuid.uuid4().hex[:8]}.png"
    upload_path = _safe_child_path(UPLOAD_DIR, upload_filename)

    try:
        content = await image.read()
        with open(upload_path, "wb") as f:
            f.write(content)

        # Get original image dimensions
        with PILImage.open(upload_path) as img:
            orig_w, orig_h = img.size

        # Calculate scale and position
        scale_w = target_width / orig_w
        scale_h = target_height / orig_h
        scale = min(scale_w, scale_h, 1.0)  # Don't upscale, only downscale if needed

        scaled_w = int(orig_w * scale)
        scaled_h = int(orig_h * scale)

        # Calculate offsets based on position
        if "left" in position:
            offset_x = 0
        elif "right" in position:
            offset_x = target_width - scaled_w
        else:  # center
            offset_x = (target_width - scaled_w) // 2

        if "top" in position:
            offset_y = 0
        elif "bottom" in position:
            offset_y = target_height - scaled_h
        else:  # center
            offset_y = (target_height - scaled_h) // 2

        logger.info(
            f"📐 Original: {orig_w}x{orig_h}, Target: {target_width}x{target_height}, Offset: ({offset_x}, {offset_y})"
        )

        # Upload to ComfyUI
        comfyui_filename = client.upload_image(str(upload_path))
        if not comfyui_filename:
            raise HTTPException(
                status_code=500, detail="Failed to upload image to ComfyUI"
            )

        # Build outpainting workflow
        # This uses InpaintModelConditioning + mask approach
        workflow = {
            # Load model
            "1": {
                "inputs": {"ckpt_name": model},
                "class_type": "CheckpointLoaderSimple",
            },
            # Load source image
            "2": {
                "inputs": {"image": comfyui_filename, "upload": "image"},
                "class_type": "LoadImage",
            },
            # Create empty canvas at target size
            "3": {
                "inputs": {
                    "width": target_width,
                    "height": target_height,
                    "batch_size": 1,
                    "color": 0,
                },
                "class_type": "EmptyImage",
            },
            # Composite source onto canvas at position
            "4": {
                "inputs": {
                    "images": ["2", 0],
                    "destination": ["3", 0],
                    "x": offset_x,
                    "y": offset_y,
                    "resize_source": True if scale < 1.0 else False,
                },
                "class_type": "ImageCompositeMasked",
            },
            # Create mask (white = inpaint area)
            "5": {
                "inputs": {
                    "width": target_width,
                    "height": target_height,
                    "batch_size": 1,
                    "color": 16777215,
                },  # White
                "class_type": "EmptyImage",
            },
            # Cut out original image area from mask (black = keep)
            "6": {
                "inputs": {
                    "images": ["2", 0],
                    "destination": ["5", 0],
                    "x": offset_x,
                    "y": offset_y,
                    "resize_source": True if scale < 1.0 else False,
                },
                "class_type": "ImageCompositeMasked",
            },
            # Convert to mask
            "7": {
                "inputs": {"image": ["6", 0], "method": "intensity"},
                "class_type": "ImageToMask",
            },
            # Grow/feather mask for smooth blending
            "8": {
                "inputs": {
                    "mask": ["7", 0],
                    "expand": feathering,
                    "tapered_corners": True,
                },
                "class_type": "GrowMask",
            },
            # Encode prompts
            "9": {
                "inputs": {"text": prompt, "clip": ["1", 1]},
                "class_type": "CLIPTextEncode",
            },
            "10": {
                "inputs": {
                    "text": "ugly, blurry, watermark, text, logo, artifacts",
                    "clip": ["1", 1],
                },
                "class_type": "CLIPTextEncode",
            },
            # VAE encode composite
            "11": {
                "inputs": {"pixels": ["4", 0], "vae": ["1", 2]},
                "class_type": "VAEEncode",
            },
            # Set masked latent for inpainting
            "12": {
                "inputs": {"samples": ["11", 0], "mask": ["8", 0]},
                "class_type": "SetLatentNoiseMask",
            },
            # KSampler
            "13": {
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": denoise,
                    "model": ["1", 0],
                    "positive": ["9", 0],
                    "negative": ["10", 0],
                    "latent_image": ["12", 0],
                },
                "class_type": "KSampler",
            },
            # VAE decode
            "14": {
                "inputs": {"samples": ["13", 0], "vae": ["1", 2]},
                "class_type": "VAEDecode",
            },
            # Save
            "15": {
                "inputs": {"filename_prefix": "oelala_reframe", "images": ["14", 0]},
                "class_type": "SaveImage",
            },
        }

        prompt_id = client.queue_prompt(workflow)
        if not prompt_id:
            raise HTTPException(
                status_code=500, detail="Failed to queue reframe workflow"
            )

        return {
            "status": "queued",
            "prompt_id": prompt_id,
            "meta": {
                "original_size": f"{orig_w}x{orig_h}",
                "target_size": f"{target_width}x{target_height}",
                "position": position,
                "prompt": prompt,
                "seed": seed,
            },
        }

    except Exception as e:
        logger.error(f"❌ Reframe error: {e}")
        raise HTTPException(status_code=500, detail="Reframe failed")


# ─────────────────────────────────────────────────────────────────────────────
# Face Swap & Face Profiles (insightface-based, direct Python — no ComfyUI)
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/detect-faces")
async def detect_faces_endpoint(
    image: UploadFile = File(...), user: User = Depends(get_current_user)
):
    """
    Detect faces in an image using InsightFace (buffalo_l).
    Returns list of detected faces with bounding boxes and confidence scores.
    """
    logger.info(f"👤 Detecting faces in {image.filename}...")

    if not face_service:
        raise HTTPException(
            status_code=503,
            detail="face_service unavailable (insightface not installed)",
        )

    try:
        content = await image.read()
        faces = await asyncio.get_event_loop().run_in_executor(
            None, face_service.detect_faces, content
        )
        return {"faces": faces, "total": len(faces)}

    except Exception as e:
        logger.error(f"❌ Face detection error: {e}")
        raise HTTPException(status_code=500, detail="Face detection failed")


@app.post("/face-swap")
async def face_swap(
    user: User = Depends(get_current_user),
    target: UploadFile = File(...),
    source: UploadFile = File(...),
    face_indices: str = Form("0"),  # comma-separated e.g. "0,1" or "-1" for all
    enhance: str = Form("none"),  # none (gfpgan requires extra package)
):
    """Face swap: replace face(s) in target image with face from source image (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            face_indices=face_indices,
            enhance=enhance,
        ),
        files={"target": target, "source": source},
        operation=Operation.TRANSFORM,
        target_type=MediaType.IMAGE,
        adapter_hint="local-faceswap",
        user=user,
        register_job_settings={
            "job_type": "face_swap",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Face Profiles API
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/api/face-profiles")
async def list_face_profiles(user: User = Depends(get_current_user)):
    """List all saved face profiles."""
    if not face_service:
        raise HTTPException(status_code=503, detail="face_service unavailable")
    profiles = face_service.list_face_profiles()
    return {"profiles": profiles, "total": len(profiles)}


@app.get("/api/face-profiles/{profile_id}")
async def get_face_profile(profile_id: str, user: User = Depends(get_current_user)):
    """Get a single face profile by ID."""
    if not face_service:
        raise HTTPException(status_code=503, detail="face_service unavailable")
    profile = face_service.get_face_profile(profile_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
    return profile


@app.post("/api/face-profiles")
async def create_face_profile(
    user: User = Depends(get_current_user),
    name: str = Form(...),
    description: str = Form(""),
    images: list[UploadFile] = File(...),
):
    """
    Create a new face profile from one or more reference images.

    Extracts and averages face embeddings for stable identity representation.
    Profile can then be used as source for face swap operations.

    Args:
        name: Display name for the profile (e.g. "John Doe")
        description: Optional description
        images: One or more reference photos (JPEG/PNG)
    """
    if not face_service:
        raise HTTPException(status_code=503, detail="face_service unavailable")

    if not images:
        raise HTTPException(
            status_code=400, detail="At least one reference image required"
        )

    logger.info(f"👤 Creating face profile '{name}' from {len(images)} image(s)")

    try:
        image_bytes_list = [await img.read() for img in images]

        profile = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: face_service.create_face_profile(
                name=name,
                images=image_bytes_list,
                description=description,
            ),
        )
        return profile

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Create face profile error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create face profile")


@app.delete("/api/face-profiles/{profile_id}")
async def delete_face_profile(profile_id: str, user: User = Depends(get_current_user)):
    """Delete a face profile and all its reference images."""
    if not face_service:
        raise HTTPException(status_code=503, detail="face_service unavailable")
    deleted = face_service.delete_face_profile(profile_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found")
    return {"status": "deleted", "profile_id": profile_id}


@app.post("/face-swap/profile")
async def face_swap_with_profile(
    user: User = Depends(get_current_user),
    target: UploadFile = File(...),
    profile_id: str = Form(...),
    face_indices: str = Form("0"),
):
    """Face swap using a saved face profile as source (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            profile_id=profile_id,
            face_indices=face_indices,
        ),
        files={"target": target},
        operation=Operation.TRANSFORM,
        target_type=MediaType.IMAGE,
        adapter_hint="local-faceswap",
        user=user,
        register_job_settings={
            "job_type": "face_swap",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Video face swap endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/face-swap-video")
async def face_swap_video(
    user: User = Depends(get_current_user),
    video: UploadFile = File(..., description="Input video file"),
    source: UploadFile = File(..., description="Source image with reference face"),
    face_indices: str = Form(
        "0", description="Comma-separated face indices or '-1' for all"
    ),
):
    """Apply face swap to every frame of a video (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            face_indices=face_indices,
        ),
        files={"video": video, "source": source},
        operation=Operation.TRANSFORM,
        target_type=MediaType.VIDEO,
        adapter_hint="local-faceswap-video",
        user=user,
        register_job_settings={
            "job_type": "face_swap_video",
        },
    )


@app.post("/face-swap-video/profile")
async def face_swap_video_with_profile(
    user: User = Depends(get_current_user),
    video: UploadFile = File(..., description="Input video file"),
    profile_id: str = Form(..., description="Saved face profile ID"),
    face_indices: str = Form(
        "0", description="Comma-separated face indices or '-1' for all"
    ),
):
    """Apply face swap to every frame of a video using a saved face profile (V2 thin wrapper)"""
    from src.backend.generation.v1_compat import dispatch_v1
    from src.backend.generation.types import Operation, MediaType

    return await dispatch_v1(
        form=dict(
            profile_id=profile_id,
            face_indices=face_indices,
        ),
        files={"video": video},
        operation=Operation.TRANSFORM,
        target_type=MediaType.VIDEO,
        adapter_hint="local-faceswap-video",
        user=user,
        register_job_settings={
            "job_type": "face_swap_video",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Face LoRA Training endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/api/face-train")
async def start_face_training(
    user: User = Depends(get_current_user),
    name: str = Form(...),
    description: str = Form(""),
    steps: int = Form(1000),
    images: list[UploadFile] = File(...),
):
    """
    Start a face LoRA training job.

    Trains a Dreambooth-style SDXL LoRA from uploaded face photos.
    Training runs in background; poll /api/face-train/{job_id} for status.

    Args:
        name: Human-readable name (becomes trigger word base, e.g. "John Doe" → ohwx_john_doe)
        description: Optional description
        steps: Number of training steps (500–2000; 1000 is a good default)
        images: 5–20 reference face photos for best results
    """
    if not face_train_service:
        raise HTTPException(status_code=503, detail="face_train_service unavailable")

    if len(images) < 5:
        raise HTTPException(
            status_code=400, detail="Upload at least 5 reference photos"
        )
    if steps < 200 or steps > 3000:
        raise HTTPException(
            status_code=400, detail="Steps must be between 200 and 3000"
        )

    try:
        img_bytes = [await img.read() for img in images]
        job = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: face_train_service.create_training_job(
                name, img_bytes, description, steps
            ),
        )
        logger.info(f"🎯 Face LoRA training job created: {job['id']} for '{name}'")
        return job
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Face training create error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to start face training")


@app.get("/api/face-train")
async def list_face_training_jobs(user: User = Depends(get_current_user)):
    """List all face LoRA training jobs."""
    if not face_train_service:
        raise HTTPException(status_code=503, detail="face_train_service unavailable")
    jobs = face_train_service.list_jobs()
    return {"jobs": jobs, "total": len(jobs)}


@app.get("/api/face-train/loras")
async def list_face_loras():
    """List all successfully trained face LoRAs available for use."""
    if not face_train_service:
        raise HTTPException(status_code=503, detail="face_train_service unavailable")
    loras = face_train_service.list_trained_loras()
    return {"loras": loras, "total": len(loras)}


@app.get("/api/face-train/{job_id}")
async def get_face_training_job(job_id: str):
    """Get status and progress of a specific training job."""
    if not face_train_service:
        raise HTTPException(status_code=503, detail="face_train_service unavailable")
    job = face_train_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return job


@app.delete("/api/face-train/{job_id}")
async def cancel_face_training_job(job_id: str, user: User = Depends(get_current_user)):
    """Cancel a pending or running training job."""
    if not face_train_service:
        raise HTTPException(status_code=503, detail="face_train_service unavailable")
    cancelled = face_train_service.cancel_job(job_id)
    if not cancelled:
        raise HTTPException(
            status_code=404, detail=f"Job '{job_id}' not found or already finished"
        )
    return {"status": "cancelled", "job_id": job_id}


@app.post("/api/face-train/{job_id}/retry")
async def retry_face_training_job(job_id: str, user: User = Depends(get_current_user)):
    """Retry a failed or cancelled training job."""
    if not face_train_service:
        raise HTTPException(status_code=503, detail="face_train_service unavailable")
    job = face_train_service.retry_job(job_id)
    if not job:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found, not failed, or missing config",
        )
    return job


# =============================================================================
# Compute Backend Inventory — admin CRUD
# =============================================================================
# Configure which servers/models can be used for generation (modular compute).
# Reads/writes src/backend/generation/compute_backends.json.

from pydantic import BaseModel as _BaseModel
from typing import List as _List
from pydantic import Field as _Field


class ComputeBackendPayload(_BaseModel):
    """Mutable fields for a compute backend (admin-editable subset)."""

    id: str
    name: str
    type: str = "comfyui"
    base_url: str = ""
    enabled: bool = True
    model_families: _List[str] = _Field(default_factory=list)
    notes: str = ""


@app.get("/api/admin/backends")
async def admin_list_backends(user: User = Depends(get_current_user)):
    """List all compute backends (name, url, type, enabled, model_families)."""
    if not await check_admin(user):
        raise HTTPException(status_code=403, detail="Admin access required")
    from src.backend.generation.compute_backends import list_backends

    return {"backends": [b.model_dump(mode="json") for b in list_backends()]}


@app.post("/api/admin/backends")
async def admin_create_backend(
    payload: ComputeBackendPayload, user: User = Depends(get_current_user)
):
    """Create a new compute backend."""
    if not await check_admin(user):
        raise HTTPException(status_code=403, detail="Admin access required")
    from src.backend.generation.compute_backends import (
        ComputeBackend,
        get_backend,
        list_backends,
        save_backends,
    )

    if get_backend(payload.id):
        raise HTTPException(status_code=409, detail=f"backend '{payload.id}' exists")
    backend = ComputeBackend(
        id=payload.id,
        name=payload.name,
        type=payload.type,
        base_url=payload.base_url,
        enabled=payload.enabled,
        model_families=list(payload.model_families),
        notes=payload.notes,
    )
    backends = list_backends() + [backend]
    save_backends(backends)
    return backend.model_dump(mode="json")


@app.put("/api/admin/backends/{backend_id}")
async def admin_update_backend(
    backend_id: str,
    payload: ComputeBackendPayload,
    user: User = Depends(get_current_user),
):
    """Update an existing compute backend."""
    if not await check_admin(user):
        raise HTTPException(status_code=403, detail="Admin access required")
    from src.backend.generation.compute_backends import (
        ComputeBackend,
        get_backend,
        list_backends,
        save_backends,
    )

    if not get_backend(backend_id):
        raise HTTPException(status_code=404, detail=f"backend '{backend_id}' not found")
    if payload.id != backend_id:
        raise HTTPException(status_code=400, detail="id cannot be changed")
    backend = ComputeBackend(
        id=payload.id,
        name=payload.name,
        type=payload.type,
        base_url=payload.base_url,
        enabled=payload.enabled,
        model_families=list(payload.model_families),
        notes=payload.notes,
    )
    backends = [backend if b.id == backend_id else b for b in list_backends()]
    save_backends(backends)
    return backend.model_dump(mode="json")


@app.delete("/api/admin/backends/{backend_id}")
async def admin_delete_backend(backend_id: str, user: User = Depends(get_current_user)):
    """Delete a compute backend."""
    if not await check_admin(user):
        raise HTTPException(status_code=403, detail="Admin access required")
    from src.backend.generation.compute_backends import (
        get_backend,
        list_backends,
        save_backends,
    )

    if not get_backend(backend_id):
        raise HTTPException(status_code=404, detail=f"backend '{backend_id}' not found")
    backends = [b for b in list_backends() if b.id != backend_id]
    save_backends(backends)
    return {"deleted": backend_id}


if __name__ == "__main__":
    uvicorn.run("app:app", host="192.168.1.2", port=7998, reload=True, log_level="info")
