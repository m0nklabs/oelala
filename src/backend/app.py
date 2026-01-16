#!/usr/bin/env python3
"""
Oelala Web Interface Backend
FastAPI application for AI Video Generation Pipeline
"""

import os
import sys

# Load environment variables from .env file BEFORE any other imports
# This must happen early so other modules get the env vars
from dotenv import load_dotenv

load_dotenv(dotenv_path="/home/flip/oelala/.env")

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
)
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import shutil
from pathlib import Path
import logging
from datetime import datetime
import json
from collections import deque
import uuid
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("/home/flip/oelala")  # Add oelala root directory

# Authentication
from auth import get_current_user, User, decode_jwt_with_secret, decode_jwt_with_jwks

# Storage client for user media (legacy sync client)
from storage_client import get_client as get_storage_client

# MediaService for oelala-storage + Supabase integration (new async client)
from media_service import MediaService, MediaRecord

# Credits system
from credits import calculate_credits
from credits_api import (
    router as credits_router,
    stripe_router,
    check_credits,
    deduct_credits,
)

# Gallery system
from gallery_api import router as gallery_router

# Profile system
from profile_api import router as profile_router

# Admin system
from admin_api import router as admin_router, check_admin

# Webhooks system
from webhooks_api import router as webhooks_router
from webhook_service import webhook_service

# ComfyUI Client for all image/video generation
try:
    from src.backend.comfyui_client import ComfyUIClient, get_comfyui_client

    print("✅ ComfyUIClient imported successfully")
except ImportError as e:
    print(f"❌ Failed to import ComfyUIClient: {e}")
    ComfyUIClient = None
    get_comfyui_client = None

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    for ws in log_subscribers:
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

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
app.include_router(webhooks_router)  # Webhooks at /webhooks/*

# Create directories
UPLOAD_DIR = Path("/home/flip/oelala/uploads")
OUTPUT_DIR = Path("/home/flip/oelala/generated")
FRONTEND_DIR = Path("/home/flip/oelala/src/frontend")
COMFYUI_OUTPUT_DIR = Path("/home/flip/oelala/ComfyUI/output")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Mount static files after CORS
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Mount ComfyUI output directory
if COMFYUI_OUTPUT_DIR.exists():
    app.mount(
        "/comfyui-output",
        StaticFiles(directory=str(COMFYUI_OUTPUT_DIR)),
        name="comfyui_output",
    )


# =============================================================================
# Helper Functions
# =============================================================================


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
    Upload a generated media file to oelala-storage and sync metadata to Supabase.
    
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

    # Start webhook retry worker
    logger.info("🪝 Starting webhook retry worker...")
    await webhook_service.start_retry_worker(interval=30.0)
    logger.info("✅ Webhook retry worker started!")


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
):
    """List media files from ComfyUI output directory

    Args:
        type: Filter by media type ('all', 'video', 'image', 'audio')
        grouped: Group videos with source images (not implemented yet)
        include_metadata: Include PNG metadata in response
        hide_start_images: Hide images that are start frames for videos (default True)
    """
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
                item["metadata"] = {"has_metadata": False, "error": str(e)}

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
    """Delete media files from ComfyUI output directory"""
    comfyui_output = Path("/home/flip/oelala/ComfyUI/output")

    if not comfyui_output.exists():
        raise HTTPException(status_code=404, detail="Output directory not found")

    deleted = []
    errors = []

    for filename in request.filenames:
        file_path = comfyui_output / filename

        # Security: prevent path traversal
        if not str(file_path.resolve()).startswith(str(comfyui_output.resolve())):
            errors.append({"filename": filename, "error": "Invalid path"})
            continue

        if not file_path.exists():
            errors.append({"filename": filename, "error": "File not found"})
            continue

        try:
            file_path.unlink()
            deleted.append(filename)
        except Exception as e:
            errors.append({"filename": filename, "error": str(e)})

    return {"deleted": deleted, "errors": errors, "count": len(deleted)}


@app.get("/loras")
async def list_loras():
    """
    List available LoRA models from ComfyUI/models/loras folder.
    Returns LoRAs grouped by noise type (high/low) for Wan2.2 dual-pass workflow.
    Includes NSFW detection based on filename patterns.
    """
    loras_dir = Path("/home/flip/oelala/ComfyUI/models/loras")

    if not loras_dir.exists():
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

    def is_nsfw(name: str, path: str) -> bool:
        """Check if a LoRA is NSFW based on name/path."""
        check_str = f"{name} {path}".lower()
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

    for lora_path in loras_dir.rglob("*.safetensors"):
        # Get relative path from loras folder
        rel_path = str(lora_path.relative_to(loras_dir))
        name = lora_path.stem

        # Get category (subdirectory name, or "root" for top-level files)
        parent = lora_path.parent.relative_to(loras_dir)
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


@app.get("/comfyui/queue")
async def get_comfyui_queue():
    """
    Get ComfyUI queue status including running and pending jobs.
    Enriches with Oelala job metadata where available.
    """
    import requests

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
                job_info = {
                    "prompt_id": prompt_id,
                    "status": "running",
                    "queue_position": 0,
                }
                # Enrich with Oelala metadata if available
                if prompt_id in active_jobs:
                    job_info.update(active_jobs[prompt_id])
                running.append(job_info)

        pending = []
        for idx, item in enumerate(data.get("queue_pending", [])):
            if len(item) >= 2:
                prompt_id = item[1]
                job_info = {
                    "prompt_id": prompt_id,
                    "status": "pending",
                    "queue_position": idx + 1,
                }
                if prompt_id in active_jobs:
                    job_info.update(active_jobs[prompt_id])
                pending.append(job_info)

        return {
            "running": running,
            "pending": pending,
            "total_running": len(running),
            "total_pending": len(pending),
        }
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to get ComfyUI queue: {e}")
        raise HTTPException(
            status_code=502, detail=f"ComfyUI connection failed: {str(e)}"
        )


@app.get("/comfyui/job/{prompt_id}")
async def get_job_status(prompt_id: str):
    """
    Get status of a specific job by prompt_id.
    Returns status (queued/running/completed/failed) and output if available.
    """
    import requests

    # Check in our active jobs store
    job_info = active_jobs.get(prompt_id, {})

    # Check ComfyUI history for completion status
    try:
        history_resp = requests.get(
            f"http://localhost:8188/history/{prompt_id}", timeout=5
        )
        if history_resp.status_code == 200:
            history = history_resp.json()
            if prompt_id in history:
                job_data = history[prompt_id]
                outputs = job_data.get("outputs", {})

                # Find video, image, or audio output
                output_video = None
                output_image = None
                output_audio = None
                for node_id, node_output in outputs.items():
                    # Video output (VHS_VideoCombine)
                    if "gifs" in node_output:
                        for gif in node_output["gifs"]:
                            if gif.get("type") == "output":
                                output_video = f"/comfyui/output/{gif['filename']}"
                                break
                    # Image output (SaveImage)
                    if "images" in node_output:
                        for img in node_output["images"]:
                            if img.get("type") == "output":
                                output_image = f"/comfyui/output/{img['filename']}"
                                break
                    # Audio output (SaveAudio, SaveAudioMP3, SaveAudioOpus)
                    if "audio" in node_output:
                        for audio in node_output["audio"]:
                            if audio.get("type") == "output":
                                output_audio = f"/comfyui/output/{audio['filename']}"
                                break

                # Auto-upload to user storage if this is a registered job
                comfyui = get_comfyui_client()
                if comfyui and (output_video or output_image or output_audio):
                    # Determine output type and path
                    if output_video:
                        output_type = "video"
                        output_filename = output_video.split("/")[-1]
                    elif output_image:
                        output_type = "image"
                        output_filename = output_image.split("/")[-1]
                    else:
                        output_type = "audio"
                        output_filename = output_audio.split("/")[-1]

                    output_path = COMFYUI_OUTPUT_DIR / output_filename

                    # Trigger async auto-upload with MediaService (storage + Supabase sync)
                    storage_path = None
                    signed_url = None
                    if output_path.exists():
                        storage_path = await comfyui.on_job_complete_async(
                            prompt_id, str(output_path), output_type
                        )
                        if storage_path:
                            logger.info(
                                f"✅ Auto-uploaded {output_type} for job {prompt_id}: {storage_path}"
                            )
                            # Generate signed URL for the uploaded content
                            signed_url = get_signed_media_url(storage_path, expires_in=86400)  # 24h

                return {
                    "prompt_id": prompt_id,
                    "status": "completed",
                    "output_video": output_video,
                    "output_image": output_image,
                    "output_audio": output_audio,
                    "url": signed_url or output_image or output_video or output_audio,  # Prefer signed URL
                    "signed_url": signed_url,
                    "storage_path": storage_path,
                    **job_info,
                }
    except Exception as e:
        logger.warning(f"Error checking history for {prompt_id}: {e}")

    # Check if it's in the queue
    try:
        queue_resp = requests.get("http://localhost:8188/queue", timeout=5)
        if queue_resp.status_code == 200:
            queue_data = queue_resp.json()

            # Check running
            for item in queue_data.get("queue_running", []):
                if len(item) >= 2 and item[1] == prompt_id:
                    return {"prompt_id": prompt_id, "status": "running", **job_info}

            # Check pending
            for idx, item in enumerate(queue_data.get("queue_pending", [])):
                if len(item) >= 2 and item[1] == prompt_id:
                    return {
                        "prompt_id": prompt_id,
                        "status": "pending",
                        "queue_position": idx + 1,
                        **job_info,
                    }
    except Exception as e:
        logger.warning(f"Error checking queue for {prompt_id}: {e}")

    # Not found anywhere - might have failed or been cancelled
    return {"prompt_id": prompt_id, "status": "unknown", **job_info}


@app.get("/comfyui/output/{filename}")
async def get_comfyui_output(filename: str):
    """Serve ComfyUI output files (videos/images)"""
    output_path = Path("/home/flip/oelala/ComfyUI/output") / filename
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(output_path)


@app.get("/media/generated/{filename}")
async def get_generated_media(filename: str):
    """Serve files from media/generated/ directory (public for now, will be auth-gated later)"""
    # TODO: Add proper auth once user-scoped storage is implemented
    media_path = Path("/home/flip/oelala/media/generated") / filename
    if not media_path.exists():
        raise HTTPException(status_code=404, detail="Media file not found")
    return FileResponse(media_path)


@app.get("/comfyui-metadata/{filename}")
async def get_comfyui_metadata(filename: str):
    """
    Extract and return the ComfyUI workflow/metadata from an output file.
    Works with videos (mp4, webm, mov) and images (png).
    Searches in both ComfyUI/output/ and media/generated/ directories.
    """
    import subprocess

    # Search in multiple directories
    search_dirs = [
        Path("/home/flip/oelala/ComfyUI/output"),
        Path("/home/flip/oelala/media/generated"),
    ]
    
    output_path = None
    for search_dir in search_dirs:
        candidate = search_dir / filename
        if candidate.exists():
            output_path = candidate
            break
    
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
            return {"metadata": metadata, "filename": filename}
        else:
            raise HTTPException(status_code=404, detail="No metadata found in file")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to extract metadata from {filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/comfyui/queue/{prompt_id}")
async def cancel_job(prompt_id: str):
    """Cancel a queued or running job"""
    import requests

    try:
        # ComfyUI interrupt endpoint
        resp = requests.post(
            "http://localhost:8188/interrupt", json={"prompt_id": prompt_id}, timeout=5
        )
        resp.raise_for_status()

        # Also try to delete from queue
        requests.post(
            "http://localhost:8188/queue", json={"delete": [prompt_id]}, timeout=5
        )

        # Remove from our tracking
        if prompt_id in active_jobs:
            del active_jobs[prompt_id]

        return {"success": True, "prompt_id": prompt_id}
    except Exception as e:
        logger.error(f"Failed to cancel job {prompt_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    import tempfile

    # Save uploaded file temporarily
    suffix = Path(file.filename).suffix if file.filename else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    metadata = {}
    try:
        # Try to read PNG metadata
        img = Image.open(tmp_path)

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
    finally:
        # Cleanup temp file
        try:
            Path(tmp_path).unlink()
        except FileNotFoundError:
            logger.debug(
                f"🐛 Temp file already removed or missing during cleanup: {tmp_path}"
            )
        except OSError as e:
            logger.warning(f"⚠️ Failed to remove temp file {tmp_path}: {e}")

    return metadata


class ExtractMetadataURLRequest(BaseModel):
    image_url: str


@app.post("/extract-metadata-url")
async def extract_metadata_from_url(request: ExtractMetadataURLRequest):
    """
    Extract workflow/prompt metadata from an image URL.
    Supports ComfyUI output URLs and local backend URLs.
    """
    import tempfile
    import httpx

    image_url = request.image_url
    metadata = {}
    tmp_path = None

    try:
        # Download image from URL
        async with httpx.AsyncClient() as client:
            response = await client.get(image_url, timeout=30.0)
            response.raise_for_status()
            content = response.content

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Extract metadata (same logic as file upload)
        img = Image.open(tmp_path)

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

    except Exception as e:
        logger.warning(f"⚠️ Failed to extract metadata from URL: {e}")
        metadata["error"] = str(e)
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink()
            except Exception:
                # Failed to cleanup temp file, not critical
                pass

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
        "upload_dir": str(UPLOAD_DIR),
        "output_dir": str(OUTPUT_DIR),
    }


# =============================================================================
# USER MEDIA API (Storage-backed, user-scoped)
# =============================================================================


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
                        filename = obj.get("filename", key.split("/")[-1] if "/" in key else key)
                        obj_type = obj.get("media_type", "")

                        if obj_type == "videos" or obj.get("content_type", "").startswith("video/"):
                            item_type = "video"
                        elif obj_type == "audio" or obj.get("content_type", "").startswith("audio/"):
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
                            "mtime": obj.get("modified_at").timestamp() if hasattr(obj.get("modified_at"), "timestamp") else 0,
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
                        # Get all user directories from storage
                        users_path = Path("/home/flip/oelala/media/users")
                        if users_path.exists():
                            for user_dir in users_path.iterdir():
                                if user_dir.is_dir():
                                    uid = user_dir.name
                                    try:
                                        objects = storage.list_user_media(uid, media_type)
                                        for obj in objects:
                                            key = obj.get("key", "")
                                            filename = obj.get("filename", key.split("/")[-1] if "/" in key else key)
                                            obj_type = obj.get("media_type", "")

                                            if obj_type == "videos" or obj.get("content_type", "").startswith("video/"):
                                                item_type = "video"
                                            elif obj_type == "audio" or obj.get("content_type", "").startswith("audio/"):
                                                item_type = "audio"
                                            else:
                                                item_type = "image"

                                            all_media.append({
                                                "name": filename,
                                                "filename": filename,
                                                "type": item_type,
                                                "url": f"/admin/user-media/{uid}/{obj_type}/{filename}",
                                                "size": obj.get("size", 0),
                                                "modified": obj.get("modified_at", ""),
                                                "mtime": obj.get("modified_at").timestamp() if hasattr(obj.get("modified_at"), "timestamp") else 0,
                                                "source": "user",
                                                "visibility": "private",
                                                "owner_id": uid,
                                            })
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
                    filename = key.split("/")[-1] if "/" in key else key

                    # Determine type from extension
                    ext = filename.lower().split(".")[-1] if "." in filename else ""
                    if ext in ("mp4", "webm", "mov", "avi"):
                        item_type = "video"
                    elif ext in ("mp3", "wav", "flac", "ogg"):
                        item_type = "audio"
                    else:
                        item_type = "image"

                    # Filter by type if specified
                    if type != "all" and item_type != type:
                        continue

                    all_media.append({
                        "name": filename,
                        "filename": filename,
                        "type": item_type,
                        "url": f"/media/generated/{key}",
                        "size": obj.get("size", 0),
                        "modified": obj.get("modified_at", ""),
                        "mtime": 0,  # Storage API returns string, not timestamp
                        "source": "generated",
                        "visibility": "dev",  # Generated = dev visibility
                    })
            except Exception as e:
                logger.debug(f"Generated media error: {e}")

        # ComfyUI local output (admin only)
        if is_admin and source in ("all", "comfyui-local"):
            try:
                # List directly from filesystem since symlink listing is broken in oelala-storage
                import os
                comfyui_path = Path("/home/flip/oelala/ComfyUI/output")
                if comfyui_path.exists():
                    for file in comfyui_path.iterdir():
                        if file.is_file() and not file.name.startswith("."):
                            filename = file.name
                            ext = filename.lower().split(".")[-1] if "." in filename else ""

                            if ext in ("mp4", "webm", "mov", "avi"):
                                item_type = "video"
                            elif ext in ("mp3", "wav", "flac", "ogg"):
                                item_type = "audio"
                            elif ext in ("png", "jpg", "jpeg", "webp", "gif"):
                                item_type = "image"
                            else:
                                continue

                            # Filter by type if specified
                            if type != "all" and item_type != type:
                                continue

                            stat = file.stat()
                            all_media.append({
                                "name": filename,
                                "filename": filename,
                                "type": item_type,
                                "url": f"/comfyui/output/{filename}",
                                "size": stat.st_size,
                                "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                "mtime": stat.st_mtime,
                                "source": "comfyui-local",
                                "visibility": "dev",  # ComfyUI local = dev visibility
                            })
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
                        
                        all_media.append({
                            "id": item.get("id"),
                            "name": item.get("title", "Untitled"),
                            "filename": storage_path.split("/")[-1] if "/" in storage_path else storage_path,
                            "type": item.get("media_type", "image"),
                            "url": url,
                            "thumbnail_url": item.get("thumbnail_url"),
                            "source": "public",
                            "visibility": "public",
                            "is_nsfw": item.get("is_nsfw", False),
                            "owner_id": item.get("user_id"),
                            "mtime": 0,  # Will sort by created_at
                        })
            except Exception as e:
                logger.debug(f"Public gallery error: {e}")

        # Sort by mtime (newest first)
        all_media.sort(key=lambda x: x.get("mtime", 0), reverse=True)

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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user/media")
async def list_user_media(type: str = "all", user: User = Depends(get_current_user)):
    """
    List media files for the authenticated user from oelala-storage.

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
            type_map = {"video": "videos", "image": "images", "audio": "audio"}
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
                    "mtime": obj.get("modified_at", datetime.now()).timestamp()
                    if isinstance(obj.get("modified_at"), datetime)
                    else 0,
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

    except Exception as e:
        # 404 means user has no storage yet - return empty list
        import httpx

        if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 404:
            logger.info(f"User {user.id} has no storage bucket yet (404)")
            return {"media": [], "stats": {"videos": 0, "images": 0, "audio": 0}}
        logger.error(f"Failed to list user media: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/user/media/{media_type}/{filename:path}")
async def get_user_media(
    media_type: str, filename: str, user: User = Depends(get_current_user)
):
    """
    Serve a user's media file from storage.
    """
    try:
        storage = get_storage_client()
        stream = storage.iter_user_media(user.id, media_type, filename)
        debug_log(f"🔍 streaming media {media_type}/{filename} for user {user.id}")

        # Determine content type
        ext = Path(filename).suffix.lower()
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

        return StreamingResponse(
            stream,
            media_type=content_type,
            headers={"Content-Disposition": f'inline; filename="{filename}"'},
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
        storage = get_storage_client()
        data = storage.get_user_media(user.id, media_type, filename)

        ext = Path(filename).suffix.lower()

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
        raise HTTPException(status_code=500, detail=str(e))


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
    if media_type not in ("images", "videos", "audio"):
        raise HTTPException(status_code=400, detail="Invalid media type")

    try:
        storage = get_storage_client()
        data = await file.read()

        # Generate unique filename if needed
        filename = file.filename or f"{uuid.uuid4()}{Path(file.filename or '').suffix}"

        result = storage.put_user_media(
            user.id, media_type, filename, data, file.content_type
        )

        return {
            "success": True,
            "filename": filename,
            "url": f"/user/media/{media_type}/{filename}",
            "size": len(data),
            **result,
        }

    except Exception as e:
        logger.error(f"Failed to upload user media: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/user/media/{media_type}/{filename:path}")
async def delete_user_media(
    media_type: str, filename: str, user: User = Depends(get_current_user)
):
    """
    Delete a user's media file.
    """
    try:
        storage = get_storage_client()
        success = storage.delete_user_media(user.id, media_type, filename)

        if not success:
            raise HTTPException(status_code=404, detail="Media not found")

        return {"success": True, "deleted": filename}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete user media: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
            "error": str(e),
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
        return {"presets": [], "error": f"Invalid JSON: {str(e)}"}
    except Exception as e:
        logger.error(f"Error loading presets: {e}")
        return {"presets": [], "error": str(e)}


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
        raise HTTPException(status_code=500, detail=str(e))


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
async def get_file(filename: str):
    """Serve generated video files"""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    # Determine media type based on extension so the browser can play it inline
    ext = file_path.suffix.lower()
    media_type = "application/octet-stream"
    if ext == ".mp4":
        media_type = "video/mp4"
    elif ext == ".gif":
        media_type = "image/gif"
    elif ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
        media_type = "image/jpeg"
    elif ext == ".json":
        media_type = "application/json"
    logger.info(f"Serving file {file_path} with media_type={media_type}")
    return FileResponse(file_path, media_type=media_type, filename=filename)


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
    """
    Legacy endpoint - redirects to SDXL via ComfyUI.
    Use /generate-sdxl for direct SDXL generation.
    Requires authentication and credits.
    """
    logger.info(
        f"🔄 Legacy generate-image request redirected to SDXL: {prompt[:50]}..."
    )

    # Build response using SDXL workflow
    client = get_comfyui_client()
    if not client or not client.is_available():
        raise HTTPException(status_code=503, detail="ComfyUI backend not available")

    import random

    seed = random.randint(0, 2**32 - 1)

    # Map aspect ratios to SDXL-optimal resolutions (1MP)
    resolutions = {
        "1:1": (1024, 1024),
        "16:9": (1344, 768),
        "9:16": (768, 1344),
        "4:3": (1152, 864),
        "3:4": (864, 1152),
        "2:3": (832, 1216),
        "3:2": (1216, 832),
        "21:9": (1536, 640),
        "9:21": (640, 1536),
    }
    width, height = resolutions.get(aspect_ratio, (1024, 1024))

    # Calculate and check credits
    credits_required = calculate_credits("generate_sdxl", width=width, height=height)
    logger.info(
        f"💰 Legacy T2I generation costs {credits_required} credits ({width}x{height}) [user={user.id}]"
    )
    await check_credits(user, credits_required)
    if not job_id:
        job_id = str(uuid.uuid4())

    # Build simple SDXL workflow
    workflow = {
        "1": {
            "inputs": {"ckpt_name": "CyberRealistic_Pony_v14.1_FP16.safetensors"},
            "class_type": "CheckpointLoaderSimple",
        },
        "2": {
            "inputs": {"text": prompt, "clip": ["1", 1]},
            "class_type": "CLIPTextEncode",
        },
        "3": {
            "inputs": {"text": "ugly, blurry, low quality", "clip": ["1", 1]},
            "class_type": "CLIPTextEncode",
        },
        "4": {
            "inputs": {"width": width, "height": height, "batch_size": 1},
            "class_type": "EmptyLatentImage",
        },
        "5": {
            "inputs": {
                "seed": seed,
                "steps": 25,
                "cfg": 7.5,
                "sampler_name": "dpmpp_2m",
                "scheduler": "karras",
                "denoise": 1,
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
            },
            "class_type": "KSampler",
        },
        "6": {
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
            "class_type": "VAEDecode",
        },
        "7": {
            "inputs": {"filename_prefix": "oelala_t2i", "images": ["6", 0]},
            "class_type": "SaveImage",
        },
    }

    prompt_id = client.queue_prompt(workflow)
    if not prompt_id:
        raise HTTPException(status_code=500, detail="Failed to queue workflow")

    # Deduct credits after successful queue
    await deduct_credits(user, credits_required, prompt_id, "SDXL T2I (legacy)")
    logger.info(f"📋 Legacy T2I queued: {prompt_id} (💰 -{credits_required} credits)")

    return {
        "status": "queued",
        "prompt_id": prompt_id,
        "job_id": job_id,
        "credits_used": credits_required,
        "meta": {"prompt": prompt, "width": width, "height": height, "seed": seed},
    }


# ─────────────────────────────────────────────────────────────────────────────
# SDXL Text-to-Image via ComfyUI
# ─────────────────────────────────────────────────────────────────────────────

# Available SDXL checkpoints (auto-detected from ComfyUI models folder)
SDXL_CHECKPOINTS = [
    "CyberRealistic_Pony_v14.1_FP16.safetensors",
    "dreamshaperXL_lightningDPMSDE.safetensors",
    "illustriousRealismBy_v10VAE.safetensors",
    "juggernautXL_ragnarok.safetensors",
    "novaAnimeXL_ilV150.safetensors",
    "ponyDiffusionV6XL_v6StartWithThisOne.safetensors",
    "reapony_v90.safetensors",
    "ultraRealisticByStable_v20FP16.safetensors",
    "waiIllustriousSDXL_v160.safetensors",
]


@app.get("/sdxl/checkpoints")
def list_sdxl_checkpoints():
    """List available SDXL checkpoints"""
    return {"checkpoints": SDXL_CHECKPOINTS}


@app.post("/generate-sdxl")
async def generate_sdxl_image(
    prompt: str = Form(...),
    negative_prompt: str = Form(
        "ugly, deformed, blurry, low quality, bad anatomy, watermark, signature, text"
    ),
    checkpoint: str = Form("CyberRealistic_Pony_v14.1_FP16.safetensors"),
    aspect_ratio: str = Form("1:1"),
    steps: int = Form(30),
    cfg: float = Form(7.5),
    seed: int = Form(-1),
    sampler_name: str = Form("dpmpp_2m"),
    scheduler: str = Form("karras"),
    lora_configs: str = Form("[]"),  # JSON string of [{name, strength}]
    user: User = Depends(get_current_user),  # Require authenticated user
):
    """
    Queue SDXL image generation via ComfyUI.
    Returns immediately with prompt_id - use /comfyui/status/{prompt_id} to poll.
    Requires authentication and credits.
    """
    import json as json_lib
    import random

    logger.info(
        f"🎨 SDXL T2I request: {prompt[:50]}... (checkpoint={checkpoint}) [user={user.id}]"
    )

    # Parse LoRA configs
    try:
        loras = json_lib.loads(lora_configs) if lora_configs else []
    except json_lib.JSONDecodeError:
        loras = []

    # Map aspect ratios to SDXL-optimal resolutions (1MP)
    resolutions = {
        "1:1": (1024, 1024),
        "16:9": (1344, 768),
        "9:16": (768, 1344),
        "4:3": (1152, 864),
        "3:4": (864, 1152),
        "2:3": (832, 1216),
        "3:2": (1216, 832),
        "21:9": (1536, 640),
        "9:21": (640, 1536),
    }
    width, height = resolutions.get(aspect_ratio, (1024, 1024))

    # Calculate and check credits
    credits_required = calculate_credits("generate_sdxl", width=width, height=height)
    logger.info(f"💰 SDXL generation costs {credits_required} credits")
    await check_credits(user, credits_required)

    # Generate seed if random
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    # Generate a job ID for credit tracking
    job_id = str(uuid.uuid4())

    try:
        client = get_comfyui_client()

        # Build SDXL workflow inline (same as ComfyUIClient.generate_sdxl_image)
        workflow = {
            "1": {
                "inputs": {"ckpt_name": checkpoint},
                "class_type": "CheckpointLoaderSimple",
            },
            "2": {
                "inputs": {"text": prompt, "clip": ["9", 1]},
                "class_type": "CLIPTextEncode",
            },
            "3": {
                "inputs": {"text": negative_prompt, "clip": ["9", 1]},
                "class_type": "CLIPTextEncode",
            },
            "4": {
                "inputs": {"width": width, "height": height, "batch_size": 1},
                "class_type": "EmptyLatentImage",
            },
            "5": {
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "denoise": 1,
                    "model": ["9", 0],
                    "positive": ["2", 0],
                    "negative": ["3", 0],
                    "latent_image": ["4", 0],
                },
                "class_type": "KSampler",
            },
            "6": {
                "inputs": {"samples": ["5", 0], "vae": ["1", 2]},
                "class_type": "VAEDecode",
            },
            "8": {
                "inputs": {"filename_prefix": "oelala_t2i", "images": ["6", 0]},
                "class_type": "SaveImage",
            },
            "9": {
                "inputs": {
                    "PowerLoraLoaderHeaderWidget": {
                        "type": "PowerLoraLoaderHeaderWidget"
                    },
                    "lora_1": {"on": False, "lora": "None", "strength": 1},
                    "lora_2": {"on": False, "lora": "None", "strength": 1},
                    "lora_3": {"on": False, "lora": "None", "strength": 1},
                    "➕ Add Lora": "",
                    "model": ["1", 0],
                    "clip": ["1", 1],
                },
                "class_type": "Power Lora Loader (rgthree)",
            },
        }

        # Apply LoRA configs
        if loras:
            for i, lora_cfg in enumerate(loras[:3], 1):
                if lora_cfg.get("name") and lora_cfg.get("name") != "None":
                    workflow["9"]["inputs"][f"lora_{i}"] = {
                        "on": True,
                        "lora": lora_cfg["name"],
                        "strength": lora_cfg.get("strength", 1.0),
                    }

        # Queue to ComfyUI (non-blocking)
        prompt_id = client.queue_prompt(workflow)

        if not prompt_id:
            raise HTTPException(status_code=500, detail="Failed to queue workflow")

        # Deduct credits after successful queue
        await deduct_credits(user, credits_required, prompt_id, "SDXL T2I")
        logger.info(f"📋 SDXL queued: {prompt_id} (💰 -{credits_required} credits)")

        return {
            "status": "queued",
            "prompt_id": prompt_id,
            "job_id": job_id,
            "credits_used": credits_required,
            "meta": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "checkpoint": checkpoint,
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SDXL queue failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    """
    Generate image using Flux Dev via ComfyUI.
    Flux doesn't use negative prompts - uses guidance instead.
    Requires authentication and credits.
    """
    import json as json_lib

    logger.info(f"⚡ Flux T2I request: {prompt[:50]}... [user={user.id}]")

    # Parse LoRA configs
    try:
        loras = json_lib.loads(lora_configs) if lora_configs else []
    except json_lib.JSONDecodeError:
        loras = []

    # Map aspect ratios to Flux-optimal resolutions
    resolutions = {
        "1:1": (1024, 1024),
        "16:9": (1344, 768),
        "9:16": (768, 1344),
        "4:3": (1152, 864),
        "3:4": (864, 1152),
        "2:3": (832, 1216),
        "3:2": (1216, 832),
        "21:9": (1536, 640),
        "9:21": (640, 1536),
    }
    width, height = resolutions.get(aspect_ratio, (1024, 1024))

    # Calculate and check credits
    credits_required = calculate_credits("generate_flux", width=width, height=height)
    logger.info(f"💰 Flux generation costs {credits_required} credits")
    await check_credits(user, credits_required)
    job_id = str(uuid.uuid4())

    try:
        client = get_comfyui_client()

        output_path = client.generate_flux_image(
            prompt=prompt,
            output_dir=str(OUTPUT_DIR),
            width=width,
            height=height,
            steps=steps,
            guidance=guidance,
            seed=seed,
            lora_configs=loras,
        )

        if not output_path:
            raise HTTPException(status_code=500, detail="Flux generation failed")

        # Deduct credits after successful generation
        await deduct_credits(user, credits_required, job_id, "Flux T2I")
        logger.info(f"⚡ Flux generated successfully (💰 -{credits_required} credits)")

        filename = Path(output_path).name

        # Upload to storage with metadata sync
        media_record = await upload_generated_media(
            user_id=user.id,
            file_path=Path(output_path),
            generation_type="t2i",
            prompt=prompt,
            workflow_id=job_id,
            extra_metadata={
                "model_name": "flux1-dev-fp8",
                "width": width,
                "height": height,
                "steps": steps,
                "guidance": guidance,
                "seed": seed,
            },
        )

        # Generate signed URL if upload succeeded
        url = f"/files/{filename}"
        signed_url = None
        if media_record:
            signed_url = get_signed_media_url(media_record.storage_path, expires_in=86400)
            url = signed_url

        return {
            "status": "success",
            "url": url,
            "signed_url": signed_url,
            "storage_path": media_record.storage_path if media_record else None,
            "filename": filename,
            "job_id": job_id,
            "credits_used": credits_required,
            "meta": {
                "prompt": prompt,
                "model": "flux1-dev-fp8",
                "width": width,
                "height": height,
                "steps": steps,
                "guidance": guidance,
                "seed": seed,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Flux generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# SD 1.5 Text-to-Image via ComfyUI
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/generate-sd15")
async def generate_sd15_image(
    prompt: str = Form(...),
    negative_prompt: str = Form(
        "(deformed, blurry, bad anatomy, extra fingers, mutated hands, poorly drawn face, low quality:1.4)"
    ),
    aspect_ratio: str = Form("2:3"),
    steps: int = Form(25),
    cfg: float = Form(7.0),
    seed: int = Form(-1),
    sampler_name: str = Form("dpmpp_sde"),
    scheduler: str = Form("karras"),
    lora_configs: str = Form("[]"),  # JSON string of [{name, strength}]
    user: User = Depends(get_current_user),  # Require authenticated user
):
    """
    Generate image using SD 1.5 (Realistic Vision V5.1) via ComfyUI.
    Requires authentication and credits.
    """
    import json as json_lib

    logger.info(f"🖼️ SD1.5 T2I request: {prompt[:50]}...")

    # Parse LoRA configs
    try:
        loras = json_lib.loads(lora_configs) if lora_configs else []
    except json_lib.JSONDecodeError:
        loras = []

    # Map aspect ratios to SD1.5-optimal resolutions (512-768 range)
    resolutions = {
        "1:1": (512, 512),
        "16:9": (768, 432),
        "9:16": (432, 768),
        "4:3": (640, 480),
        "3:4": (480, 640),
        "2:3": (512, 768),
        "3:2": (768, 512),
    }
    width, height = resolutions.get(aspect_ratio, (512, 768))

    # Calculate and check credits
    credits_required = calculate_credits(
        "sd15", width=width, height=height, steps=steps
    )
    logger.info(
        f"💰 SD1.5 generation costs {credits_required} credits ({width}x{height}) [user={user.id}]"
    )
    await check_credits(user, credits_required)
    job_id = str(uuid.uuid4())

    try:
        client = get_comfyui_client()

        output_path = client.generate_sd15_image(
            prompt=prompt,
            output_dir=str(OUTPUT_DIR),
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            seed=seed,
            sampler_name=sampler_name,
            scheduler=scheduler,
            lora_configs=loras,
        )

        if not output_path:
            raise HTTPException(status_code=500, detail="SD1.5 generation failed")

        filename = Path(output_path).name

        # Deduct credits after successful generation
        await deduct_credits(user, credits_required, job_id, "SD1.5 T2I")
        logger.info(f"🎨 SD1.5 image generated (💰 -{credits_required} credits)")

        # Upload to storage with metadata sync
        media_record = await upload_generated_media(
            user_id=user.id,
            file_path=Path(output_path),
            generation_type="t2i",
            prompt=prompt,
            workflow_id=job_id,
            extra_metadata={
                "model_name": "Realistic_Vision_V5.1",
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
                "negative_prompt": negative_prompt,
            },
        )

        # Generate signed URL if upload succeeded
        url = f"/files/{filename}"
        signed_url = None
        if media_record:
            signed_url = get_signed_media_url(media_record.storage_path, expires_in=86400)
            url = signed_url

        return {
            "status": "success",
            "url": url,
            "signed_url": signed_url,
            "storage_path": media_record.storage_path if media_record else None,
            "filename": filename,
            "job_id": job_id,
            "credits_used": credits_required,
            "meta": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "model": "Realistic_Vision_V5.1",
                "width": width,
                "height": height,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SD1.5 generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Wan2.2 Text-to-Image via ComfyUI (DisTorch2 Multi-GPU)
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/generate-wan22-t2i")
async def generate_wan22_t2i(
    prompt: str = Form(...),
    aspect_ratio: str = Form("1:1"),
    steps: int = Form(8),
    seed: int = Form(-1),
    user: User = Depends(get_current_user),  # Require authenticated user
):
    """
    Generate image using Wan2.2 T2V model in T2I mode via ComfyUI.
    Uses DisTorch2 multi-GPU setup with high/low noise models.
    Very high quality but slower than other T2I models.
    Requires authentication and credits.
    """

    logger.info(f"🎬 Wan2.2 T2I request: {prompt[:50]}...")

    # Map aspect ratios to Wan2.2-compatible resolutions
    resolutions = {
        "1:1": (512, 512),
        "16:9": (832, 480),
        "9:16": (480, 832),
        "4:3": (640, 480),
        "3:4": (480, 640),
        "2:3": (512, 768),
        "3:2": (768, 512),
    }
    width, height = resolutions.get(aspect_ratio, (512, 512))

    # Calculate and check credits
    credits_required = calculate_credits("wan22_t2i", width=width, height=height)
    logger.info(
        f"💰 Wan2.2 T2I generation costs {credits_required} credits ({width}x{height}) [user={user.id}]"
    )
    await check_credits(user, credits_required)
    job_id = str(uuid.uuid4())

    try:
        client = get_comfyui_client()

        output_path = client.generate_wan22_t2i(
            prompt=prompt,
            output_dir=str(OUTPUT_DIR),
            width=width,
            height=height,
            steps=steps,
            seed=seed,
        )

        if not output_path:
            raise HTTPException(status_code=500, detail="Wan2.2 T2I generation failed")

        filename = Path(output_path).name

        # Deduct credits after successful generation
        await deduct_credits(user, credits_required, job_id, "Wan2.2 T2I")
        logger.info(f"🎨 Wan2.2 T2I image generated (💰 -{credits_required} credits)")

        # Upload to storage with metadata sync
        media_record = await upload_generated_media(
            user_id=user.id,
            file_path=Path(output_path),
            generation_type="t2i",
            prompt=prompt,
            workflow_id=job_id,
            extra_metadata={
                "model_name": "wan2.2-t2i-distorch2",
                "width": width,
                "height": height,
                "steps": steps,
                "seed": seed,
            },
        )

        # Generate signed URL if upload succeeded
        url = f"/files/{filename}"
        signed_url = None
        if media_record:
            signed_url = get_signed_media_url(media_record.storage_path, expires_in=86400)
            url = signed_url

        return {
            "status": "success",
            "url": url,
            "signed_url": signed_url,
            "storage_path": media_record.storage_path if media_record else None,
            "filename": filename,
            "job_id": job_id,
            "credits_used": credits_required,
            "meta": {
                "prompt": prompt,
                "model": "wan2.2-t2i-distorch2",
                "width": width,
                "height": height,
                "steps": steps,
                "seed": seed,
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Wan2.2 T2I generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
    """
    Generate video from uploaded image via ComfyUI.
    This endpoint wraps the ComfyUI Wan2.2 I2V workflow.
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

    # Calculate duration for credit calculation
    duration_seconds = num_frames / fps if fps > 0 else 3
    # Get resolution dimensions for credit calculation
    width, height = comfyui.get_resolution_dimensions(resolution, aspect_ratio)

    # Calculate and check credits
    credits_required = calculate_credits(
        "wan22_i2v",
        width=width,
        height=height,
        duration_seconds=int(duration_seconds),
    )
    logger.info(
        f"💰 I2V generation costs {credits_required} credits ({resolution}, {duration_seconds:.1f}s) [user={user.id}]"
    )
    await check_credits(user, credits_required)
    job_id = str(uuid.uuid4())

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = f"i2v_{timestamp}_{file.filename}"
    input_path = UPLOAD_DIR / input_filename

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Upload to ComfyUI
    comfyui_image_name = comfyui.upload_image(str(input_path))
    if not comfyui_image_name:
        raise HTTPException(status_code=500, detail="Failed to upload image to ComfyUI")

    # Get resolution dimensions
    width, height = comfyui.get_resolution_dimensions(resolution, aspect_ratio)

    # Build I2V workflow
    import random

    seed = random.randint(0, 2**32 - 1)

    # Adjust num_frames to Wan2.2 format (4k+1)
    k = round((num_frames - 1) / 4)
    k = max(1, k)
    num_frames = 4 * k + 1

    workflow = comfyui.build_api_workflow(
        image_name=comfyui_image_name,
        prompt=prompt or "smooth motion, cinematic",
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        steps=6,
        cfg=1.0,
        seed=seed,
        output_prefix=f"oelala_i2v_{timestamp}",
    )

    # Queue workflow
    prompt_id = comfyui.queue_prompt(workflow)
    if not prompt_id:
        raise HTTPException(status_code=500, detail="Failed to queue workflow")

    # Deduct credits after successful queue
    await deduct_credits(user, credits_required, prompt_id, "Wan2.2 I2V")
    logger.info(f"📋 I2V queued: {prompt_id} (💰 -{credits_required} credits)")

    return {
        "status": "queued",
        "prompt_id": prompt_id,
        "job_id": job_id,
        "input_image": input_filename,
        "credits_used": credits_required,
        "meta": {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "seed": seed,
        },
    }


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
    style_prompt: str = Form(..., description="Style description (e.g., 'anime style, vibrant colors')"),
    mode: str = Form("style_transfer", description="V2V mode: style_transfer, anime, enhance"),
    strength: float = Form(0.5, description="Style strength (0.0-1.0, higher = more style change)"),
    num_frames: int = Form(41, description="Output frames (4k+1 format for Wan2.2)"),
    resolution: str = Form("480p", description="Output resolution: 480p, 720p"),
    fps: int = Form(16, description="Output FPS"),
    preserve_motion: bool = Form(True, description="Try to preserve original motion"),
    seed: int = Form(-1, description="Random seed (-1 for random)"),
    generation_mode: str = Form("standard", description="I2V generation mode to use"),
    user: User = Depends(get_current_user),
):
    """
    Video-to-Video style transfer using AI.
    
    Process:
    1. Extract first frame from input video
    2. Apply style via I2V workflow
    3. Generate new video with transferred style
    
    Use cases:
    - Turn real footage into anime style
    - Apply artistic filters
    - AI enhancement of video quality
    
    Note: This uses I2V pipeline under the hood with the first frame as input.
    For best results, use short clips (2-5 seconds) with clear subjects.
    """
    import cv2
    
    if not get_comfyui_client:
        raise HTTPException(status_code=503, detail="ComfyUI client not available")
    
    comfyui = get_comfyui_client()
    if not comfyui.is_available():
        raise HTTPException(
            status_code=503,
            detail="ComfyUI not running. Start with: cd ~/oelala/ComfyUI && python main.py --listen",
        )
    
    # Validate strength
    strength = max(0.0, min(1.0, strength))
    
    # Wan2.2 requires num_frames in format 4k+1
    k = round((num_frames - 1) / 4)
    k = max(1, k)
    num_frames = 4 * k + 1
    
    # Get resolution dimensions
    width, height = comfyui.get_resolution_dimensions(resolution, "16:9")  # Default to 16:9 for video
    
    # Calculate credits (V2V costs same as I2V)
    duration_seconds = num_frames / fps if fps > 0 else 3
    credits_required = calculate_credits(
        "generate_wan22_comfyui",
        width=width,
        height=height,
        duration_seconds=duration_seconds,
    )
    
    logger.info(f"💰 V2V generation costs {credits_required} credits ({width}x{height}, {num_frames} frames)")
    await check_credits(user, credits_required)
    
    job_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save uploaded video
    input_filename = f"v2v_input_{timestamp}_{file.filename}"
    input_path = UPLOAD_DIR / input_filename
    
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    logger.info(f"📥 V2V input saved: {input_path}")
    
    try:
        # Extract first frame using OpenCV
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")
        
        # Get video info
        original_fps = cap.get(cv2.CAP_PROP_FPS) or 24
        original_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_duration = original_frames / original_fps
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"🎬 Input video: {original_width}x{original_height}, {original_fps}fps, {original_duration:.1f}s")
        
        # Read first frame
        ret, first_frame = cap.read()
        cap.release()
        
        if not ret or first_frame is None:
            raise HTTPException(status_code=400, detail="Could not extract first frame from video")
        
        # Save first frame as input image for I2V
        frame_filename = f"v2v_frame_{timestamp}.png"
        frame_path = UPLOAD_DIR / frame_filename
        
        # Resize to target resolution if needed
        if first_frame.shape[1] != width or first_frame.shape[0] != height:
            first_frame = cv2.resize(first_frame, (width, height), interpolation=cv2.INTER_LANCZOS4)
        
        cv2.imwrite(str(frame_path), first_frame)
        logger.info(f"📸 First frame extracted: {frame_path} ({width}x{height})")
        
        # Build the style-enhanced prompt
        base_prompt = style_prompt.strip()
        
        # Add motion preservation hints based on mode
        if preserve_motion:
            if mode == "anime":
                full_prompt = f"{base_prompt}, anime style, smooth animation, consistent character design, fluid motion"
            elif mode == "enhance":
                full_prompt = f"{base_prompt}, high quality, sharp details, natural movement, enhanced clarity"
            else:  # style_transfer
                full_prompt = f"{base_prompt}, artistic style transfer, preserve motion, consistent style throughout"
        else:
            full_prompt = base_prompt
        
        logger.info(f"🎨 V2V prompt: {full_prompt}")
        
        # Generate video using DisTorch2 I2V workflow with the extracted frame
        output_prefix = f"oelala_v2v_{timestamp}"
        
        # Call the DisTorch2 generation method from comfyui_client
        output_path = comfyui.generate_distorch2_video(
            image_path=str(frame_path),
            prompt=full_prompt,
            output_dir=str(OUTPUT_DIR),
            output_prefix=output_prefix,
            resolution=resolution,
            aspect_ratio="16:9",  # Default for video
            num_frames=num_frames,
            fps=fps,
            steps=6,  # Default for DisTorch2
            cfg=1.0,  # Default for DisTorch2
            seed=seed if seed >= 0 else -1,
        )
        
        if not output_path:
            raise HTTPException(status_code=500, detail="V2V generation failed - no output")
        
        output_filename = Path(output_path).name
        
        # Deduct credits
        await deduct_credits(user, credits_required, job_id, f"V2V {mode}")
        logger.info(f"🎬 V2V generated: {output_filename} (💰 -{credits_required} credits)")
        
        # Upload to storage
        media_record = await upload_generated_media(
            user_id=user.id,
            file_path=Path(output_path),
            generation_type="v2v",
            prompt=full_prompt,
            workflow_id=job_id,
            extra_metadata={
                "model_name": "wan2.2-v2v-distorch2",
                "mode": mode,
                "strength": strength,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": fps,
                "seed": seed,
                "original_video": input_filename,
                "original_duration": original_duration,
            },
        )
        
        # Get signed URL
        url = f"/files/{output_filename}"
        signed_url = None
        if media_record:
            signed_url = get_signed_media_url(media_record.storage_path, expires_in=86400)
        
        return {
            "status": "success",
            "job_id": job_id,
            "url": url,
            "signed_url": signed_url,
            "filename": output_filename,
            "credits_used": credits_required,
            "meta": {
                "mode": mode,
                "style_prompt": style_prompt,
                "strength": strength,
                "width": width,
                "height": height,
                "num_frames": num_frames,
                "fps": fps,
                "seed": seed,
                "original_video": {
                    "filename": file.filename,
                    "duration": original_duration,
                    "resolution": f"{original_width}x{original_height}",
                },
            },
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ V2V generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"V2V generation failed: {str(e)}")
    finally:
        # Cleanup input video (keep extracted frame for debugging)
        if input_path.exists():
            try:
                input_path.unlink()
            except Exception:
                pass


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
    generation_mode: str = Form("standard", description="Generation mode: standard, nsfw_lora"),
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
    """
    Generate Wan2.2 I2V video via ComfyUI with DisTorch2 Dual-Pass workflow.
    Requires authentication and credits.

    This endpoint uses ComfyUI with:
    - Dual-Pass: High Noise model (steps 0-3) → Low Noise model (steps 3+)
    - DisTorch2 expert_mode_allocations for optimal memory scaling
    - CONVERTED T5: umt5-xxl-enc-bf16-uncensored-CONVERTED.safetensors
    - SageAttention (sageattn_qk_int8_pv_fp16_triton)

    Note: num_frames will be adjusted to nearest valid Wan2.2 value (4k+1)
    """
    if not get_comfyui_client:
        raise HTTPException(status_code=503, detail="ComfyUI client not available")

    comfyui = get_comfyui_client()

    if not comfyui.is_available():
        raise HTTPException(
            status_code=503,
            detail="ComfyUI not running. Start with: cd ~/oelala/ComfyUI && python main.py --listen",
        )

    # Wan2.2 requires num_frames in format 4k+1 (5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, ...)
    # Round to nearest valid value
    k = round((num_frames - 1) / 4)
    k = max(1, k)  # Minimum k=1 gives 5 frames
    num_frames = 4 * k + 1
    logger.info(f"🎞️ Adjusted num_frames to Wan2.2 format: {num_frames} (4*{k}+1)")

    # Calculate duration for credit calculation
    duration_seconds = num_frames / fps if fps > 0 else 3
    # Get resolution dimensions for credit calculation
    _comfyui_temp = get_comfyui_client()
    width, height = _comfyui_temp.get_resolution_dimensions(resolution, aspect_ratio)

    # Calculate and check credits
    credits_required = calculate_credits(
        "generate_wan22_comfyui",
        width=width,
        height=height,
        duration_seconds=int(duration_seconds),
    )
    logger.info(
        f"💰 Wan2.2 I2V generation costs {credits_required} credits ({resolution}, {duration_seconds:.1f}s) [user={user.id}]"
    )
    await check_credits(user, credits_required)
    job_id = str(uuid.uuid4())

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = f"comfyui_{timestamp}_{file.filename}"
    input_path = UPLOAD_DIR / input_filename

    # Save uploaded file
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"📤 Saved input image: {input_path}")
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Generate output filename
    if not output_filename:
        output_filename = f"wan22_comfyui_{timestamp}.mp4"
    elif not output_filename.endswith(".mp4"):
        output_filename += ".mp4"

    output_prefix = f"oelala_{timestamp}"

    # Build workflow and inject metadata into input image
    comfyui = get_comfyui_client()
    width, height = comfyui.get_resolution_dimensions(resolution, aspect_ratio)

    # Build the workflow that will be used
    workflow = comfyui.build_workflow(
        image_name=input_filename,
        prompt=prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps,
        steps=steps,
        cfg=cfg,
        seed=seed if seed >= 0 else 42,
        output_prefix=output_prefix,
    )

    # Inject workflow metadata into the input PNG
    # Parse lora_configs JSON
    parsed_lora_configs = []
    if lora_configs:
        try:
            parsed_lora_configs = json.loads(lora_configs)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse lora_configs JSON: {lora_configs}")

    prompt_params = {
        "prompt": prompt,
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "num_frames": num_frames,
        "fps": fps,
        "steps": steps,
        "cfg": cfg,
        "seed": seed,
        "timestamp": timestamp,
        "unet_high_noise": unet_high_noise,
        "unet_low_noise": unet_low_noise,
        "lora_configs": parsed_lora_configs,
    }
    inject_png_workflow_metadata(str(input_path), workflow, prompt_params)

    # Check if sequential/extend mode is enabled
    is_extend_mode = extend_mode.lower() in ("true", "1", "yes")
    actual_clip_count = max(1, min(5, clip_count)) if is_extend_mode else 1

    try:
        if is_extend_mode and actual_clip_count > 1:
            # Sequential generation mode
            total_frames = num_frames * actual_clip_count
            logger.info(
                f"🎬 Starting Sequential Wan2.2 generation ({actual_clip_count} clips)"
            )
            logger.info(f"   📐 Resolution: {resolution}, Aspect: {aspect_ratio}")
            logger.info(
                f"   🎞️ Frames per clip: {num_frames}, Total: {total_frames}, FPS: {fps}"
            )
            logger.info(f"   ⚙️ Steps: {steps}, CFG: {cfg}, Seed: {seed}")
            logger.info(f"   🔧 Unet: H={unet_high_noise}, L={unet_low_noise}")
            if parsed_lora_configs:
                logger.info(f"   🎨 LoRAs: {len(parsed_lora_configs)} configured")
            logger.info(f"   📝 Prompt: {prompt[:100]}...")

            # Generate sequential video via ComfyUI
            loop = asyncio.get_event_loop()
            result_path = await loop.run_in_executor(
                None,
                lambda: comfyui.generate_sequential_video(
                    image_path=str(input_path),
                    prompt=prompt,
                    output_dir=str(OUTPUT_DIR),
                    clip_count=actual_clip_count,
                    resolution=resolution,
                    aspect_ratio=aspect_ratio,
                    num_frames=num_frames,
                    fps=fps,
                    steps=steps,
                    cfg=cfg,
                    seed=seed,
                    output_prefix=output_prefix,
                    unet_high_noise=unet_high_noise,
                    unet_low_noise=unet_low_noise,
                    lora_configs=parsed_lora_configs,
                ),
            )
        else:
            # Standard single-clip generation
            logger.info("🎬 Starting Wan2.2 ComfyUI generation")
            logger.info(f"   📐 Resolution: {resolution}, Aspect: {aspect_ratio}")
            logger.info(f"   🎞️ Frames: {num_frames}, FPS: {fps}")
            logger.info(f"   ⚙️ Steps: {steps}, CFG: {cfg}, Seed: {seed}")
            logger.info(f"   🔧 Unet: H={unet_high_noise}, L={unet_low_noise}")
            logger.info(f"   🎯 Mode: {generation_mode}")
            if parsed_lora_configs:
                logger.info(f"   🎨 LoRAs: {len(parsed_lora_configs)} configured")
                for i, lc in enumerate(parsed_lora_configs):
                    logger.info(
                        f"      [{i + 1}] H={lc.get('high') or 'none'}, L={lc.get('low') or 'none'} @ {lc.get('strength', 1.0)}"
                    )
            logger.info(f"   📝 Prompt: {prompt[:100]}...")

            # Generate video via ComfyUI in threadpool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            result_path = await loop.run_in_executor(
                None,  # Default threadpool
                lambda: comfyui.generate_video(
                    image_path=str(input_path),
                    prompt=prompt,
                    output_dir=str(OUTPUT_DIR),
                    resolution=resolution,
                    aspect_ratio=aspect_ratio,
                    num_frames=num_frames,
                    fps=fps,
                    steps=steps,
                    cfg=cfg,
                    seed=seed,
                    output_prefix=output_prefix,
                    unet_high_noise=unet_high_noise,
                    unet_low_noise=unet_low_noise,
                    lora_configs=parsed_lora_configs,
                    generation_mode=generation_mode,
                ),
            )

        if result_path and Path(result_path).exists():
            # Copy to expected output path if different
            final_output = OUTPUT_DIR / output_filename
            if str(result_path) != str(final_output):
                shutil.copy(result_path, final_output)
                result_path = str(final_output)

            total_frames = (
                num_frames * actual_clip_count if is_extend_mode else num_frames
            )

            # Deduct credits after successful generation
            await deduct_credits(user, credits_required, job_id, "Wan2.2 I2V")
            logger.info(f"🎬 Wan2.2 video generated (💰 -{credits_required} credits)")

            # Upload to oelala-storage and sync metadata to Supabase
            media_record = await upload_generated_media(
                user_id=user.id,
                file_path=final_output,
                generation_type="i2v",
                prompt=prompt,
                workflow_id=job_id,
                extra_metadata={
                    "model_name": "wan2.2_i2v_14B_Q6",
                    "resolution": resolution,
                    "aspect_ratio": aspect_ratio,
                    "num_frames": total_frames,
                    "fps": fps,
                    "steps": steps,
                    "cfg": cfg,
                    "seed": seed,
                    "input_image": input_filename,
                    "extend_mode": is_extend_mode,
                    "clip_count": actual_clip_count,
                },
            )

            # Generate signed URL if upload succeeded
            video_url = f"/files/{output_filename}"  # Fallback to local
            signed_url = None
            if media_record:
                signed_url = get_signed_media_url(media_record.storage_path, expires_in=86400)  # 24h
                video_url = signed_url

            return {
                "success": True,
                "message": f"Wan2.2 video generated via ComfyUI{' (sequential)' if actual_clip_count > 1 else ''}",
                "input_image": input_filename,
                "output_video": output_filename,
                "video_url": video_url,
                "signed_url": signed_url,
                "storage_path": media_record.storage_path if media_record else None,
                "video_path": result_path,
                "prompt": prompt,
                "num_frames": total_frames,
                "frames_per_clip": num_frames,
                "clip_count": actual_clip_count,
                "extend_mode": is_extend_mode,
                "fps": fps,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
                "timestamp": timestamp,
                "backend": "comfyui",
                "model": "wan2.2_i2v_14B_Q6",
                "job_id": job_id,
                "credits_used": credits_required,
            }
        else:
            raise HTTPException(
                status_code=500, detail="ComfyUI video generation returned no output"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ ComfyUI generation error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Wan2.2 ComfyUI generation failed: {str(e)}"
        )


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
    user: User = Depends(get_current_user),  # Require authenticated user
):
    """
    Queue Wan2.2 I2V video generation and return immediately.
    Requires authentication and credits.

    Unlike /generate-wan22-comfyui, this endpoint returns immediately with a prompt_id.
    Use /comfyui/job/{prompt_id} to poll for completion status.

    This allows queueing multiple jobs without waiting.
    """
    if not get_comfyui_client:
        raise HTTPException(status_code=503, detail="ComfyUI client not available")

    comfyui = get_comfyui_client()

    if not comfyui.is_available():
        raise HTTPException(
            status_code=503,
            detail="ComfyUI not running. Start with: cd ~/oelala/ComfyUI && python main.py --listen",
        )

    # Wan2.2 requires num_frames in format 4k+1
    k = round((num_frames - 1) / 4)
    k = max(1, k)
    num_frames = 4 * k + 1

    # Calculate and check credits
    _comfyui_temp = get_comfyui_client()
    width, height = _comfyui_temp.get_resolution_dimensions(resolution, aspect_ratio)
    duration_seconds = num_frames / fps if fps > 0 else 3
    credits_required = calculate_credits(
        "generate_wan22_comfyui",
        width=width,
        height=height,
        duration_seconds=int(duration_seconds),
    )
    logger.info(
        f"💰 Wan2.2 async costs {credits_required} credits ({resolution}, {duration_seconds:.1f}s) [user={user.id}]"
    )
    await check_credits(user, credits_required)
    job_id = str(uuid.uuid4())

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = f"comfyui_{timestamp}_{file.filename}"
    input_path = UPLOAD_DIR / input_filename

    # Save uploaded file
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"📤 Saved input image: {input_path}")
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Upload to ComfyUI
    image_name = comfyui.upload_image(str(input_path))
    if not image_name:
        raise HTTPException(status_code=500, detail="Failed to upload image to ComfyUI")

    # Parse lora_configs
    parsed_lora_configs = []
    if lora_configs:
        try:
            parsed_lora_configs = json.loads(lora_configs)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse lora_configs JSON: {lora_configs}")

    # Generate output prefix
    if not output_filename:
        output_filename = f"wan22_async_{timestamp}.mp4"
    output_prefix = f"oelala_{timestamp}"

    # Get actual seed
    actual_seed = (
        seed if seed >= 0 else int(datetime.now().timestamp() * 1000) % 2147483647
    )

    # Map resolution to long_edge
    resolution_map = {"480p": 480, "576p": 576, "720p": 720, "1080p": 1080}
    long_edge = resolution_map.get(resolution, 480)

    # Check if sequential/extend mode is enabled
    is_extend_mode = extend_mode.lower() in ("true", "1", "yes")
    actual_clip_count = max(1, min(5, clip_count)) if is_extend_mode else 1

    # Build workflow
    if is_extend_mode and actual_clip_count > 1:
        # Build sequential workflow for multiple clips
        logger.info(
            f"🎬 Building sequential workflow: {actual_clip_count} clips × {num_frames} frames"
        )
        comfyui.get_resolution_dimensions(resolution, aspect_ratio)
        workflow = comfyui._build_sequential_workflow(
            image_name=image_name,
            prompt=prompt,
            clip_count=actual_clip_count,
            resolution=resolution,
            aspect_ratio=aspect_ratio,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            cfg=cfg,
            seed=actual_seed,
            output_prefix=output_prefix,
            unet_high_noise=unet_high_noise,
            unet_low_noise=unet_low_noise,
            lora_configs=parsed_lora_configs,
        )
    else:
        # Build standard single-clip workflow
        workflow = comfyui.build_q6_workflow(
            image_name=image_name,
            prompt=prompt,
            num_frames=num_frames,
            fps=fps,
            steps=steps,
            cfg=cfg,
            seed=actual_seed,
            output_prefix=output_prefix,
            aspect_ratio=aspect_ratio,
            long_edge=long_edge,
            unet_high_noise=unet_high_noise,
            unet_low_noise=unet_low_noise,
            lora_configs=parsed_lora_configs,
        )

    # Queue the workflow (non-blocking)
    prompt_id = comfyui.queue_prompt(workflow)

    if not prompt_id:
        raise HTTPException(
            status_code=500, detail="Failed to queue workflow to ComfyUI"
        )

    # Calculate total frames for tracking
    total_frames = num_frames * actual_clip_count if is_extend_mode else num_frames

    # Register job with ComfyUI client for auto-upload on completion
    comfyui.register_job(
        prompt_id=prompt_id,
        user_id=user.id,
        prompt=prompt,
        settings={
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "num_frames": num_frames,
            "fps": fps,
            "extend_mode": is_extend_mode,
            "clip_count": actual_clip_count if is_extend_mode else 1,
        },
    )

    # Register job with WebSocket manager for progress tracking
    if ws_manager and job_queue_manager:
        ws_manager.register_job(prompt_id, user_id=user.id)
        job_queue_manager.register_job(
            prompt_id=prompt_id,
            user_id=user.id,
            job_type="wan22_i2v",
            metadata={
                "prompt": prompt[:100],
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "num_frames": total_frames,
                "fps": fps,
            },
        )

        # Register progress callback to broadcast real-time progress
        if progress_monitor:
            progress_monitor.register_callback(
                prompt_id, create_progress_callback(prompt_id)
            )

    # Store job info for tracking
    job_info = {
        "prompt": prompt[:100],
        "resolution": resolution,
        "aspect_ratio": aspect_ratio,
        "num_frames": total_frames,
        "frames_per_clip": num_frames,
        "clip_count": actual_clip_count,
        "extend_mode": is_extend_mode,
        "fps": fps,
        "steps": steps,
        "seed": actual_seed,
        "output_prefix": output_prefix,
        "output_filename": output_filename,
        "input_image": input_filename,
        "created_at": timestamp,
        "lora_count": len(parsed_lora_configs),
    }
    active_jobs[prompt_id] = job_info

    if is_extend_mode and actual_clip_count > 1:
        logger.info(
            f"🚀 Queued sequential job: {prompt_id} ({actual_clip_count} clips)"
        )
    else:
        logger.info(f"🚀 Queued async job: {prompt_id}")
    logger.info(f"   📐 {resolution} {aspect_ratio}, {num_frames}f @ {fps}fps")
    logger.info(f"   📝 {prompt[:50]}...")

    # Deduct credits after successful queue
    await deduct_credits(user, credits_required, prompt_id, "Wan2.2 I2V (async)")
    logger.info(f"   💰 -{credits_required} credits")

    return {
        "success": True,
        "prompt_id": prompt_id,
        "job_id": job_id,
        "status": "queued",
        "credits_used": credits_required,
        "message": "Job queued successfully. Poll /comfyui/job/{prompt_id} for status.",
        **job_info,
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
                "stats_error": str(e),
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
    user: User = Depends(get_current_user),  # Require authenticated user
):
    """
    Generate video from text prompt via ComfyUI T2V workflow.
    Supports multiple models: wan22 (Wan2.2 14B), ltx2 (LTX-2 19B).
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

    if not prompt or len(prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt is required")

    # Validate model_type
    from comfyui_client import T2V_GENERATION_MODES, build_ltx2_t2v_workflow
    
    if model_type not in T2V_GENERATION_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_type. Available: {list(T2V_GENERATION_MODES.keys())}"
        )
    
    mode_config = T2V_GENERATION_MODES[model_type]
    logger.info(f"🎬 T2V generation with model: {mode_config['name']}")

    # Get resolution dimensions
    width, height = comfyui.get_resolution_dimensions(resolution, aspect_ratio)

    # Calculate duration for credit calculation
    duration_seconds = num_frames / fps if fps > 0 else 3

    # Calculate and check credits (using model_type for pricing)
    credit_type = f"{model_type}_t2v"
    credits_required = calculate_credits(
        credit_type,
        width=width,
        height=height,
        duration_seconds=int(duration_seconds),
    )
    logger.info(
        f"💰 T2V generation costs {credits_required} credits ({resolution}, {duration_seconds:.1f}s) [user={user.id}]"
    )
    await check_credits(user, credits_required)
    job_id = str(uuid.uuid4())

    # Generate unique timestamp
    import random

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = random.randint(0, 2**32 - 1)

    # Build workflow based on model type
    if model_type == "ltx2":
        # LTX-2 doesn't need frame adjustment
        workflow = build_ltx2_t2v_workflow(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            steps=mode_config["default_steps"],
            cfg=mode_config["default_cfg"],
            seed=seed,
            filename_prefix=f"oelala_ltx2_t2v_{timestamp}",
        )
        if not workflow:
            raise HTTPException(status_code=500, detail="Failed to build LTX-2 workflow")
    else:
        # Wan2.2: Adjust num_frames to 4k+1 format
        k = round((num_frames - 1) / 4)
        k = max(1, k)
        num_frames = 4 * k + 1
        
        workflow = comfyui.build_t2v_workflow(
            prompt=prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            fps=fps,
            steps=mode_config["default_steps"],
            cfg=mode_config["default_cfg"],
            seed=seed,
            output_prefix=f"oelala_t2v_{timestamp}",
        )

    # Queue workflow
    prompt_id = comfyui.queue_prompt(workflow)
    if not prompt_id:
        raise HTTPException(status_code=500, detail="Failed to queue workflow")

    # Register job with ComfyUI client for auto-upload on completion
    comfyui.register_job(
        prompt_id=prompt_id,
        user_id=user.id,
        prompt=prompt,
        settings={
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "num_frames": num_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "model_type": model_type,
        },
    )

    # Register job with WebSocket manager for progress tracking
    job_type = f"{model_type}_t2v"
    if ws_manager and job_queue_manager:
        ws_manager.register_job(prompt_id, user_id=user.id)
        job_queue_manager.register_job(
            prompt_id=prompt_id,
            user_id=user.id,
            job_type=job_type,
            metadata={
                "prompt": prompt[:100],
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "num_frames": num_frames,
                "fps": fps,
                "model_type": model_type,
            },
        )

        # Register progress callback to broadcast real-time progress
        if progress_monitor:
            progress_monitor.register_callback(
                prompt_id, create_progress_callback(prompt_id)
            )

    # Deduct credits after successful queue
    model_display = mode_config['name']
    await deduct_credits(user, credits_required, prompt_id, f"{model_display} T2V")
    logger.info(f"📋 T2V ({model_type}) queued: {prompt_id} (💰 -{credits_required} credits)")

    return {
        "status": "queued",
        "prompt_id": prompt_id,
        "job_id": job_id,
        "credits_used": credits_required,
        "meta": {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "fps": fps,
            "seed": seed,
            "type": "text-to-video",
            "model_type": model_type,
            "model_name": mode_config['name'],
        },
    }


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
    input_filename = f"pose_{timestamp}_{file.filename}"
    input_path = UPLOAD_DIR / input_filename

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

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


@app.post("/caption-image")
async def caption_image(
    file: UploadFile = File(...),
    model: str = Form("florence2", description="Model: florence2, blip2, cogvlm"),
    mode: str = Form("detailed", description="Mode: brief, detailed, tags, structured"),
):
    """
    Generate a caption/description for an uploaded image.
    Uses ComfyUI Florence2 node or falls back to template response.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Save uploaded file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = f"caption_{timestamp}_{file.filename}"
    input_path = UPLOAD_DIR / input_filename

    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Try ComfyUI Florence2 if available
    if get_comfyui_client:
        comfyui = get_comfyui_client()
        if comfyui.is_available():
            try:
                # Upload image to ComfyUI
                comfyui_image = comfyui.upload_image(str(input_path))
                if comfyui_image:
                    # Build Florence2 workflow
                    workflow = {
                        "1": {
                            "class_type": "LoadImage",
                            "inputs": {"image": comfyui_image},
                        },
                        "2": {
                            "class_type": "Florence2Run",
                            "inputs": {
                                "image": ["1", 0],
                                "florence2_model": ["3", 0],
                                "text_input": "",
                                "task": "detailed_caption"
                                if mode == "detailed"
                                else "caption",
                                "fill_mask": False,
                                "keep_model_loaded": True,
                                "max_new_tokens": 1024,
                                "num_beams": 3,
                                "do_sample": False,
                                "output_mask_select": "",
                            },
                        },
                        "3": {
                            "class_type": "DownloadAndLoadFlorence2Model",
                            "inputs": {
                                "model": "microsoft/Florence-2-large",
                                "precision": "fp16",
                                "attention": "sdpa",
                            },
                        },
                    }

                    # Queue and wait for result
                    prompt_id = comfyui.queue_prompt(workflow)
                    if prompt_id:
                        # Wait for completion (max 60s)
                        import time

                        for _ in range(60):
                            history = comfyui.get_history(prompt_id)
                            if history and prompt_id in history:
                                outputs = history[prompt_id].get("outputs", {})
                                # Florence2Run outputs text
                                for node_id, output in outputs.items():
                                    if "text" in output:
                                        caption = output["text"]
                                        if isinstance(caption, list):
                                            caption = caption[0] if caption else ""
                                        return {
                                            "caption": caption,
                                            "model": "florence2",
                                            "mode": mode,
                                        }
                                break
                            time.sleep(1)
            except Exception as e:
                logger.warning(f"Florence2 captioning failed: {e}")

    # Fallback: return placeholder (no vision model available)
    # In production, you'd integrate with local transformers or external API
    logger.info("Using placeholder caption (no vision model available)")

    placeholder_captions = {
        "brief": "An image uploaded by the user.",
        "detailed": "This is an uploaded image. To get accurate captions, install Florence2 in ComfyUI: ComfyUI-Florence2 custom node.",
        "tags": "image, uploaded, user content",
        "structured": "Subject: unknown, Style: photograph, Mood: neutral, Setting: unidentified",
    }

    return {
        "caption": placeholder_captions.get(mode, placeholder_captions["detailed"]),
        "model": "placeholder",
        "mode": mode,
        "note": "Install ComfyUI-Florence2 for real image captioning",
    }


@app.post("/generate-prompt")
async def generate_prompt(
    input: str = Form(..., description="Basic idea or keywords"),
    style: str = Form(None, description="Style preset"),
    mode: str = Form("expand", description="Mode: expand, refine, variations"),
    include_negative: bool = Form(True),
    include_motion: bool = Form(False),
):
    """
    Generate enhanced prompts from basic input.
    Uses templates by default, can integrate with LLM for smarter enhancement.
    """
    if not input or not input.strip():
        raise HTTPException(status_code=400, detail="Input is required")

    base_input = input.strip()

    # Style keywords mapping
    STYLE_KEYWORDS = {
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

    # Quality boosters
    quality_suffix = ", masterpiece, best quality, highly detailed"

    # Build enhanced prompt
    style_part = STYLE_KEYWORDS.get(style, "") if style else ""
    if style_part:
        enhanced_prompt = f"{base_input}, {style_part}{quality_suffix}"
    else:
        enhanced_prompt = f"{base_input}{quality_suffix}"

    # Generate negative prompt
    negative_prompt = ""
    if include_negative:
        negative_prompt = "ugly, deformed, blurry, low quality, bad anatomy, watermark, signature, text, cropped, worst quality, low resolution, jpeg artifacts, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck"

    # Generate motion prompt for video
    motion_prompt = ""
    if include_motion:
        motion_keywords = [
            "smooth camera motion",
            "cinematic movement",
            "fluid animation",
            "natural motion",
            "gentle movement",
        ]
        motion_prompt = ", ".join(motion_keywords)

    # Generate variations if requested
    variations = None
    if mode == "variations":
        variations = [
            f"{base_input}, dramatic lighting{quality_suffix}",
            f"{base_input}, soft natural light{quality_suffix}",
            f"{base_input}, studio lighting, professional{quality_suffix}",
        ]
        if style_part:
            variations = [f"{v}, {style_part}" for v in variations]

    return {
        "prompt": enhanced_prompt,
        "negative_prompt": negative_prompt,
        "motion_prompt": motion_prompt,
        "variations": variations,
        "input": base_input,
        "style": style,
        "mode": mode,
    }


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
async def youtube_info(request: YouTubeInfoRequest):
    """
    Fetch metadata from a YouTube URL without downloading.
    Returns: title, channel, duration, thumbnail, view_count, etc.
    """
    import subprocess
    import shutil

    url = request.url.strip()
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
        raise HTTPException(500, f"Failed to parse yt-dlp output: {e}")
    except Exception as e:
        logger.error(f"YouTube info error: {e}")
        raise HTTPException(500, f"Failed to fetch video info: {e}")


@app.post("/youtube/download")
async def youtube_download(request: YouTubeDownloadRequest):
    """
    Download video/audio from YouTube URL.
    Returns: path to downloaded file.
    """
    import subprocess
    import shutil

    url = request.url.strip()
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

    output_path = UPLOAD_DIR / output_filename

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
        raise HTTPException(500, f"Download failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Video to Text (Video Captioning) via SmolVLM / local models
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/caption-video")
async def caption_video(
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
        upload_path = UPLOAD_DIR / upload_filename

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
        raise HTTPException(status_code=500, detail=str(e))


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
    """
    Generate audio from text (TTS, music, or SFX) via ComfyUI.
    Requires authentication and credits.

    Args:
        text: Input text (speech text or music/sfx prompt)
        mode: tts, music, sfx
        voice: TTS voice preset (alloy, echo, fable, onyx, nova, shimmer)
        style: Music style (ambient, cinematic, electronic, etc.)
        duration: Duration in seconds (for music/sfx)
        speed: TTS speed multiplier (not used with ChatterBox)
        pitch: TTS pitch multiplier (not used with ChatterBox)
    """
    logger.info(f"🎵 Audio request: mode={mode}, text={text[:50]}...")

    # Calculate and check credits based on mode
    if mode == "tts":
        # TTS is cheaper, just text length matters
        credits_required = calculate_credits(
            "mmaudio_short", duration_seconds=min(10, duration)
        )
    elif mode == "music":
        credits_required = calculate_credits(
            "mmaudio_long" if duration > 10 else "mmaudio_short",
            duration_seconds=duration,
        )
    elif mode == "sfx":
        credits_required = calculate_credits(
            "mmaudio_short", duration_seconds=min(10, duration)
        )
    else:
        credits_required = 3  # Default

    logger.info(
        f"💰 Audio generation costs {credits_required} credits (mode={mode}, duration={duration}s) [user={user.id}]"
    )
    await check_credits(user, credits_required)
    job_id = str(uuid.uuid4())

    import random

    client = get_comfyui_client()
    output_id = uuid.uuid4().hex[:8]

    try:
        if mode == "tts":
            # Use ChatterBox TTS via ComfyUI
            voice_settings = VOICE_PRESETS.get(voice, VOICE_PRESETS["nova"])

            workflow = {
                "1": {
                    "class_type": "ChatterBoxEngineNode",
                    "inputs": {
                        "language": voice_settings["language"],
                        "device": "auto",
                        "exaggeration": voice_settings["exaggeration"],
                        "temperature": voice_settings["temperature"],
                        "cfg_weight": 0.5,
                        "crash_protection_template": "hmm ,, {seg} hmm ,,",
                    },
                },
                "2": {
                    "class_type": "UnifiedTTSTextNode",
                    "inputs": {
                        "TTS_engine": ["1", 0],
                        "text": text,
                        "narrator_voice": "none",
                        "seed": random.randint(0, 2**32 - 1),
                        "enable_chunking": True,
                        "max_chars_per_chunk": 400,
                        "chunk_combination_method": "auto",
                        "silence_between_chunks_ms": 100,
                        "enable_audio_cache": True,
                        "batch_size": 0,
                    },
                },
                "3": {
                    "class_type": "SaveAudio",
                    "inputs": {
                        "audio": ["2", 0],
                        "filename_prefix": f"tts_{output_id}",
                    },
                },
            }

            logger.info(f"🎤 TTS workflow: voice={voice}, text_len={len(text)}")
            prompt_id = client.queue_prompt(workflow)

            # Deduct credits after successful queue
            await deduct_credits(user, credits_required, prompt_id, "TTS Audio")
            logger.info(f"🎤 TTS queued: {prompt_id} (💰 -{credits_required} credits)")

            return {
                "status": "queued",
                "prompt_id": prompt_id,
                "job_id": job_id,
                "credits_used": credits_required,
                "mode": "tts",
                "voice": voice,
                "text_preview": text[:100] + ("..." if len(text) > 100 else ""),
            }

        elif mode == "music":
            # Use MMAudio for text-to-audio music generation
            # Build prompt with style prefix
            music_prompt = f"{style} music, {text}"

            # NOTE: MMAudio requires specific models to be downloaded
            # Models needed from: https://huggingface.co/Kijai/MMAudio_safetensors
            # These go in: ComfyUI/models/mmaudio/
            workflow = {
                "1": {
                    "class_type": "MMAudioModelLoader",
                    "inputs": {
                        "mmaudio_model": "mmaudio_large_44k_v2_fp16.safetensors",
                        "base_precision": "fp16",
                    },
                },
                "2": {
                    "class_type": "MMAudioFeatureUtilsLoader",
                    "inputs": {
                        "synchformer_model": "mmaudio_synchformer_fp16.safetensors",
                        "vae_model": "mmaudio_vae_44k_fp16.safetensors",
                        "clip_model": "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors",
                        "mode": "44k",
                        "precision": "fp16",
                    },
                },
                "3": {
                    "class_type": "MMAudioSampler",
                    "inputs": {
                        "mmaudio_model": ["1", 0],
                        "feature_utils": ["2", 0],
                        "prompt": music_prompt,
                        "negative_prompt": "noise, distortion, glitch, silence",
                        "duration": float(duration),
                        "steps": 25,
                        "cfg": 4.5,
                        "seed": random.randint(0, 2**32 - 1),
                        "mask_away_clip": False,
                        "force_offload": True,
                    },
                },
                "4": {
                    "class_type": "SaveAudio",
                    "inputs": {
                        "audio": ["3", 0],
                        "filename_prefix": f"music_{output_id}",
                    },
                },
            }

            logger.info(f"🎵 Music workflow: style={style}, duration={duration}s")
            prompt_id = client.queue_prompt(workflow)

            # Deduct credits after successful queue
            await deduct_credits(user, credits_required, prompt_id, "Music Generation")
            logger.info(
                f"🎵 Music queued: {prompt_id} (💰 -{credits_required} credits)"
            )

            return {
                "status": "queued",
                "prompt_id": prompt_id,
                "job_id": job_id,
                "credits_used": credits_required,
                "mode": "music",
                "style": style,
                "duration": duration,
                "prompt": music_prompt,
            }

        elif mode == "sfx":
            # Use MMAudio for sound effects (shorter duration)
            sfx_duration = min(duration, 10)

            workflow = {
                "1": {
                    "class_type": "MMAudioModelLoader",
                    "inputs": {
                        "mmaudio_model": "mmaudio_large_44k_v2_fp16.safetensors",
                        "base_precision": "fp16",
                    },
                },
                "2": {
                    "class_type": "MMAudioFeatureUtilsLoader",
                    "inputs": {
                        "synchformer_model": "mmaudio_synchformer_fp16.safetensors",
                        "vae_model": "mmaudio_vae_44k_fp16.safetensors",
                        "clip_model": "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors",
                        "mode": "44k",
                        "precision": "fp16",
                    },
                },
                "3": {
                    "class_type": "MMAudioSampler",
                    "inputs": {
                        "mmaudio_model": ["1", 0],
                        "feature_utils": ["2", 0],
                        "prompt": text,
                        "negative_prompt": "music, speech, voice, singing",
                        "duration": float(sfx_duration),
                        "steps": 25,
                        "cfg": 4.5,
                        "seed": random.randint(0, 2**32 - 1),
                        "mask_away_clip": False,
                        "force_offload": True,
                    },
                },
                "4": {
                    "class_type": "SaveAudio",
                    "inputs": {
                        "audio": ["3", 0],
                        "filename_prefix": f"sfx_{output_id}",
                    },
                },
            }

            logger.info(
                f"🔊 SFX workflow: prompt={text[:50]}, duration={sfx_duration}s"
            )
            prompt_id = client.queue_prompt(workflow)

            # Deduct credits after successful queue
            await deduct_credits(user, credits_required, prompt_id, "SFX Generation")
            logger.info(f"🔊 SFX queued: {prompt_id} (💰 -{credits_required} credits)")

            return {
                "status": "queued",
                "prompt_id": prompt_id,
                "job_id": job_id,
                "credits_used": credits_required,
                "mode": "sfx",
                "duration": sfx_duration,
                "prompt": text,
            }

    except Exception as e:
        logger.error(f"❌ Audio error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return {"error": "Unknown audio mode"}


# ─────────────────────────────────────────────────────────────────────────────
# Voice Cloning (F5-TTS) via ComfyUI
# ─────────────────────────────────────────────────────────────────────────────


class VoiceCloneRequest(BaseModel):
    voice_sample_path: str
    text: str
    model: str = "F5v1"
    speed: float = 1.0


@app.post("/voice-clone")
async def voice_clone(request: VoiceCloneRequest):
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
    voice_path = request.voice_sample_path
    if not voice_path.startswith("/"):
        voice_path = str(UPLOAD_DIR / voice_path)

    # Check if file exists
    if not Path(voice_path).exists():
        raise HTTPException(400, f"Voice sample not found: {voice_path}")

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
        raise HTTPException(status_code=500, detail=str(e))


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
async def generate_lip_sync(request: LipSyncRequest):
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
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Image-to-Image (I2I) via ComfyUI
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/generate-i2i")
async def generate_i2i(
    file: UploadFile = File(...),
    prompt: str = Form(...),
    negative_prompt: str = Form(
        "ugly, deformed, blurry, low quality, bad anatomy, watermark"
    ),
    denoise: float = Form(0.7),
    checkpoint: str = Form("CyberRealistic_Pony_v14.1_FP16.safetensors"),
    steps: int = Form(25),
    cfg: float = Form(7.5),
    seed: int = Form(-1),
    sampler_name: str = Form("dpmpp_2m"),
    scheduler: str = Form("karras"),
    user: User = Depends(get_current_user),  # Require authenticated user
):
    """
    Image-to-Image generation via ComfyUI.
    Uploads source image, applies style transfer / modification.
    Requires authentication and credits.

    Args:
        file: Source image file
        prompt: What to generate / how to modify
        negative_prompt: What to avoid
        denoise: 0.0 = keep source, 1.0 = ignore source (typical: 0.4-0.7)
        checkpoint: SDXL checkpoint to use
    """
    import random

    logger.info(
        f"🎨 I2I request: {prompt[:50]}... (denoise={denoise}, checkpoint={checkpoint})"
    )

    # Calculate and check credits (image-to-image is similar to T2I)
    credits_required = calculate_credits("sdxl", width=1024, height=1024, steps=steps)
    logger.info(
        f"💰 I2I generation costs {credits_required} credits (denoise={denoise}) [user={user.id}]"
    )
    await check_credits(user, credits_required)
    job_id = str(uuid.uuid4())

    client = get_comfyui_client()
    if not client or not client.is_available():
        raise HTTPException(status_code=503, detail="ComfyUI backend not available")

    # Generate seed
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    # Save uploaded file to temp location
    upload_filename = f"i2i_input_{uuid.uuid4().hex[:8]}.png"
    upload_path = UPLOAD_DIR / upload_filename

    try:
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Upload to ComfyUI
        comfyui_filename = client.upload_image(str(upload_path))
        if not comfyui_filename:
            raise HTTPException(
                status_code=500, detail="Failed to upload image to ComfyUI"
            )

        logger.info(f"📤 Uploaded to ComfyUI: {comfyui_filename}")

        # Build I2I workflow
        workflow = {
            "1": {
                "inputs": {"ckpt_name": checkpoint},
                "class_type": "CheckpointLoaderSimple",
            },
            "2": {
                "inputs": {"image": comfyui_filename, "upload": "image"},
                "class_type": "LoadImage",
            },
            "3": {
                "inputs": {"pixels": ["2", 0], "vae": ["1", 2]},
                "class_type": "VAEEncode",
            },
            "4": {
                "inputs": {"text": prompt, "clip": ["1", 1]},
                "class_type": "CLIPTextEncode",
            },
            "5": {
                "inputs": {"text": negative_prompt, "clip": ["1", 1]},
                "class_type": "CLIPTextEncode",
            },
            "6": {
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": sampler_name,
                    "scheduler": scheduler,
                    "denoise": denoise,
                    "model": ["1", 0],
                    "positive": ["4", 0],
                    "negative": ["5", 0],
                    "latent_image": ["3", 0],
                },
                "class_type": "KSampler",
            },
            "7": {
                "inputs": {"samples": ["6", 0], "vae": ["1", 2]},
                "class_type": "VAEDecode",
            },
            "8": {
                "inputs": {"filename_prefix": "oelala_i2i", "images": ["7", 0]},
                "class_type": "SaveImage",
            },
        }

        prompt_id = client.queue_prompt(workflow)
        if not prompt_id:
            raise HTTPException(status_code=500, detail="Failed to queue I2I workflow")

        # Deduct credits after successful queue
        await deduct_credits(user, credits_required, prompt_id, "I2I Generation")
        logger.info(f"🎨 I2I queued: {prompt_id} (💰 -{credits_required} credits)")

        return {
            "status": "queued",
            "prompt_id": prompt_id,
            "job_id": job_id,
            "credits_used": credits_required,
            "meta": {
                "prompt": prompt,
                "denoise": denoise,
                "checkpoint": checkpoint,
                "seed": seed,
                "source_image": comfyui_filename,
            },
        }

    except Exception as e:
        logger.error(f"❌ I2I error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Image Upscaling via ComfyUI
# ─────────────────────────────────────────────────────────────────────────────

UPSCALE_MODELS = [
    "RealESRGAN_x4plus.pth",
    "RealESRGAN_x4plus_anime_6B.pth",
    "RealESRGAN_x2plus.pth",
    "4x-UltraSharp.pth",
    "4x_NMKD-Siax_200k.pth",
]


@app.get("/upscale/models")
def list_upscale_models():
    """List available upscale models"""
    return {"models": UPSCALE_MODELS}


@app.post("/upscale")
async def upscale_image(
    file: UploadFile = File(...),
    model: str = Form("RealESRGAN_x4plus.pth"),
    scale: int = Form(4),
    face_enhance: bool = Form(False),
):
    """
    Upscale image using Real-ESRGAN via ComfyUI.

    Args:
        file: Source image
        model: Upscale model (RealESRGAN variants)
        scale: Scale factor (2x or 4x)
        face_enhance: Apply GFPGAN face enhancement
    """
    logger.info(
        f"🔍 Upscale request: model={model}, scale={scale}x, face_enhance={face_enhance}"
    )

    client = get_comfyui_client()
    if not client or not client.is_available():
        raise HTTPException(status_code=503, detail="ComfyUI backend not available")

    # Save uploaded file
    upload_filename = f"upscale_input_{uuid.uuid4().hex[:8]}.png"
    upload_path = UPLOAD_DIR / upload_filename

    try:
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Upload to ComfyUI
        comfyui_filename = client.upload_image(str(upload_path))
        if not comfyui_filename:
            raise HTTPException(
                status_code=500, detail="Failed to upload image to ComfyUI"
            )

        logger.info(f"📤 Uploaded to ComfyUI: {comfyui_filename}")

        # Build upscale workflow
        # Uses UpscaleModelLoader + ImageUpscaleWithModel nodes
        workflow = {
            "1": {
                "inputs": {"image": comfyui_filename, "upload": "image"},
                "class_type": "LoadImage",
            },
            "2": {"inputs": {"model_name": model}, "class_type": "UpscaleModelLoader"},
            "3": {
                "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]},
                "class_type": "ImageUpscaleWithModel",
            },
        }

        # Add face enhancement if requested (requires ComfyUI-GFPGAN extension)
        if face_enhance:
            # Try with GFPGAN - falls back gracefully if not installed
            workflow["4"] = {
                "inputs": {
                    "image": ["3", 0],
                    "model_name": "GFPGANv1.4.pth",
                    "strength": 0.8,
                },
                "class_type": "GFPGANFaceRestoration",
            }
            workflow["5"] = {
                "inputs": {"filename_prefix": "oelala_upscale", "images": ["4", 0]},
                "class_type": "SaveImage",
            }
        else:
            workflow["4"] = {
                "inputs": {"filename_prefix": "oelala_upscale", "images": ["3", 0]},
                "class_type": "SaveImage",
            }

        prompt_id = client.queue_prompt(workflow)
        if not prompt_id:
            raise HTTPException(
                status_code=500, detail="Failed to queue upscale workflow"
            )

        return {
            "status": "queued",
            "prompt_id": prompt_id,
            "meta": {
                "model": model,
                "scale": scale,
                "face_enhance": face_enhance,
                "source_image": comfyui_filename,
            },
        }

    except Exception as e:
        logger.error(f"❌ Upscale error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Video Upscaling
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/upscale-video")
async def upscale_video(
    file: UploadFile = File(...),
    model: str = Form("lanczos"),
    scale: float = Form(2.0),
):
    """
    Upscale video using various methods.

    Args:
        file: Source video
        model: Upscale method:
            - lanczos: High-quality lanczos interpolation (fast, no AI)
            - bicubic: Bicubic interpolation (fast, no AI)
            - seedvr2: SeedVR2 AI upscaler (slow, requires GPU, best quality)
        scale: Upscale factor (2.0 = double resolution)

    Note: AI upscale models (realesrgan) are not currently installed.
    Use 'lanczos' for reliable basic upscaling.
    """
    logger.info(f"🎬 Video upscale request: model={model}, scale={scale}x")

    # Validate model
    valid_models = ["lanczos", "bicubic", "bilinear", "nearest-exact", "area"]
    if model not in valid_models and model != "seedvr2":
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model}'. Available: {valid_models + ['seedvr2']}",
        )

    client = get_comfyui_client()
    if not client or not client.is_available():
        raise HTTPException(status_code=503, detail="ComfyUI backend not available")

    # Save uploaded video
    upload_filename = f"upscale_video_input_{uuid.uuid4().hex[:8]}.mp4"
    upload_path = UPLOAD_DIR / upload_filename

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

        if model == "seedvr2":
            # SeedVR2 AI video upscaler workflow
            # Requires: SeedVR2LoadDiTModel, SeedVR2LoadVAEModel, SeedVR2VideoUpscaler
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
                        "model": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
                        "device": "cuda:1",  # RTX 5060 Ti has more VRAM
                        "offload_device": "cuda:0",
                        "attention_mode": "sdpa",
                    },
                    "class_type": "SeedVR2LoadDiTModel",
                },
                "3": {
                    "inputs": {
                        "model": "seedvr2_ema_vae_fp32.safetensors",
                        "device": "cuda:0",
                    },
                    "class_type": "SeedVR2LoadVAEModel",
                },
                "4": {
                    "inputs": {
                        "image": ["1", 0],
                        "dit": ["2", 0],
                        "vae": ["3", 0],
                        "seed": 42,
                        "resolution": int(1080 * scale / 2),  # Target resolution
                        "max_resolution": 0,
                        "batch_size": 5,
                        "uniform_batch_size": False,
                        "color_correction": "lab",
                    },
                    "class_type": "SeedVR2VideoUpscaler",
                },
                "5": {
                    "inputs": {
                        "frame_rate": 30,
                        "loop_count": 0,
                        "filename_prefix": "oelala_upscale_seedvr2",
                        "format": "video/h264-mp4",
                        "pix_fmt": "yuv420p",
                        "crf": 19,
                        "save_metadata": True,
                        "pingpong": False,
                        "save_output": True,
                        "images": ["4", 0],
                    },
                    "class_type": "VHS_VideoCombine",
                },
            }
        else:
            # Basic upscaling with ImageScale (lanczos, bicubic, etc.)
            # Note: ImageScale requires explicit width/height, not scale factor
            # We'll use a reasonable output size based on scale
            # Default input assumed ~480p, so 2x = ~960p
            target_width = int(1920 * scale / 2)  # Scale from 960 base
            target_height = int(1080 * scale / 2)  # Scale from 540 base

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
                        "image": ["1", 0],
                        "upscale_method": model,
                        "width": target_width,
                        "height": target_height,
                        "crop": "disabled",
                    },
                    "class_type": "ImageScale",
                },
                "3": {
                    "inputs": {
                        "frame_rate": 30,
                        "loop_count": 0,
                        "filename_prefix": f"oelala_upscale_{model}",
                        "format": "video/h264-mp4",
                        "pix_fmt": "yuv420p",
                        "crf": 19,
                        "save_metadata": True,
                        "pingpong": False,
                        "save_output": True,
                        "images": ["2", 0],
                    },
                    "class_type": "VHS_VideoCombine",
                },
            }

        prompt_id = client.queue_prompt(workflow)
        if not prompt_id:
            raise HTTPException(
                status_code=500, detail="Failed to queue video upscale workflow"
            )

        return {
            "status": "queued",
            "prompt_id": prompt_id,
            "meta": {
                "model": model,
                "scale": scale,
                "source_video": comfyui_filename,
            },
        }

    except Exception as e:
        logger.error(f"❌ Video upscale error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Frame Interpolation
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/interpolate-video")
async def interpolate_video(
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
    upload_path = UPLOAD_DIR / upload_filename

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
        raise HTTPException(status_code=500, detail=str(e))


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
    """
    Video-to-Video style transfer via ComfyUI.
    Extracts frames, applies img2img to each, reassembles video.
    Requires authentication and credits.

    Args:
        file: Source video file
        prompt: Style/transformation prompt
        denoise: 0.0 = keep original, 1.0 = ignore original (0.3-0.6 recommended)
        fps: Output FPS
        max_frames: Maximum frames to process
    """
    import random

    logger.info(
        f"🎬 V2V request: {prompt[:50]}... (denoise={denoise}, fps={fps}, max_frames={max_frames})"
    )

    # Calculate duration and credits (V2V is expensive - frame by frame processing)
    duration_seconds = max_frames / fps

    # V2V multiplier for frame-by-frame processing overhead
    V2V_COST_MULTIPLIER = 1.5

    credits_required = calculate_credits(
        "wan22_i2v",  # Similar cost to video generation
        width=512,
        height=512,
        duration_seconds=int(duration_seconds),
        steps=steps,
    )
    # V2V is more expensive due to frame processing overhead
    credits_required = int(credits_required * V2V_COST_MULTIPLIER)

    logger.info(
        f"💰 V2V generation costs {credits_required} credits ({max_frames} frames, {duration_seconds:.1f}s) [user={user.id}]"
    )
    await check_credits(user, credits_required)
    job_id = str(uuid.uuid4())

    client = get_comfyui_client()
    if not client or not client.is_available():
        raise HTTPException(status_code=503, detail="ComfyUI backend not available")

    # Generate seed
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    # Save uploaded video
    upload_filename = f"v2v_input_{uuid.uuid4().hex[:8]}.mp4"
    upload_path = UPLOAD_DIR / upload_filename

    try:
        with open(upload_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Upload video to ComfyUI input folder
        comfyui_input = Path("/home/flip/oelala/ComfyUI/input")
        video_dest = comfyui_input / upload_filename
        shutil.copy(str(upload_path), str(video_dest))

        logger.info(f"📤 Video copied to ComfyUI: {upload_filename}")

        # Build V2V workflow using AnimateDiff or frame-by-frame approach
        # Using VideoToFrames + img2img batch + FramesToVideo pattern
        workflow = {
            # Load video and extract frames
            "1": {
                "inputs": {
                    "video": upload_filename,
                    "force_rate": fps,
                    "force_size": "Disabled",
                    "custom_width": 512,
                    "custom_height": 512,
                    "frame_load_cap": max_frames,
                    "skip_first_frames": 0,
                    "select_every_nth": 1,
                },
                "class_type": "VHS_LoadVideo",
            },
            # Load checkpoint for img2img
            "2": {
                "inputs": {"ckpt_name": "CyberRealistic_Pony_v14.1_FP16.safetensors"},
                "class_type": "CheckpointLoaderSimple",
            },
            # Positive prompt
            "3": {
                "inputs": {"text": prompt, "clip": ["2", 1]},
                "class_type": "CLIPTextEncode",
            },
            # Negative prompt
            "4": {
                "inputs": {"text": negative_prompt, "clip": ["2", 1]},
                "class_type": "CLIPTextEncode",
            },
            # VAE encode frames
            "5": {
                "inputs": {"pixels": ["1", 0], "vae": ["2", 2]},
                "class_type": "VAEEncode",
            },
            # KSampler batch - applies style to all frames
            "6": {
                "inputs": {
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "dpmpp_2m",
                    "scheduler": "karras",
                    "denoise": denoise,
                    "model": ["2", 0],
                    "positive": ["3", 0],
                    "negative": ["4", 0],
                    "latent_image": ["5", 0],
                },
                "class_type": "KSampler",
            },
            # VAE decode
            "7": {
                "inputs": {"samples": ["6", 0], "vae": ["2", 2]},
                "class_type": "VAEDecode",
            },
            # Combine frames back to video
            "8": {
                "inputs": {
                    "frame_rate": fps,
                    "loop_count": 0,
                    "filename_prefix": "oelala_v2v",
                    "format": "video/h264-mp4",
                    "pingpong": False,
                    "save_output": True,
                    "images": ["7", 0],
                },
                "class_type": "VHS_VideoCombine",
            },
        }

        prompt_id = client.queue_prompt(workflow)
        if not prompt_id:
            raise HTTPException(status_code=500, detail="Failed to queue V2V workflow")

        # Deduct credits after successful queue
        await deduct_credits(user, credits_required, prompt_id, "V2V Style Transfer")
        logger.info(f"🎬 V2V queued: {prompt_id} (💰 -{credits_required} credits)")

        return {
            "status": "queued",
            "prompt_id": prompt_id,
            "job_id": job_id,
            "credits_used": credits_required,
            "meta": {
                "prompt": prompt,
                "denoise": denoise,
                "fps": fps,
                "max_frames": max_frames,
                "seed": seed,
                "source_video": upload_filename,
            },
        }

    except Exception as e:
        logger.error(f"❌ V2V error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/videos/{filename}")
async def get_video(filename: str):
    """Download generated video file"""
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(path=file_path, media_type="video/mp4", filename=filename)


@app.get("/images/{filename}")
async def get_image(filename: str):
    """Download uploaded image file"""
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(path=file_path, media_type="image/jpeg", filename=filename)


@app.get("/list-videos")
async def list_videos():
    """List all generated videos from both output directories"""
    videos = []

    # Scan OUTPUT_DIR (generated/)
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

    # Also scan COMFYUI_OUTPUT_DIR
    if COMFYUI_OUTPUT_DIR.exists():
        for file_path in COMFYUI_OUTPUT_DIR.glob("*.mp4"):
            stat = file_path.stat()
            videos.append(
                {
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    "mtime": stat.st_mtime,
                    "url": f"/comfyui-outputs/{file_path.name}",
                }
            )

    # Sort by mtime (newest first)
    videos.sort(key=lambda v: v.get("mtime", 0), reverse=True)

    return {"videos": videos, "count": len(videos)}


@app.post("/train-lora")
async def train_lora_model(
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
    training_dir = UPLOAD_DIR / training_id
    training_dir.mkdir(exist_ok=True)

    # Save uploaded files
    image_paths = []
    for i, file in enumerate(files):
        input_filename = f"train_{i:03d}_{file.filename}"
        input_path = training_dir / input_filename

        try:
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            image_paths.append(str(input_path))
        except Exception as e:
            logger.error(f"Error saving file {file.filename}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to save {file.filename}"
            )

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
    training_dir = UPLOAD_DIR / training_id
    training_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for i, file in enumerate(files):
        input_filename = f"train_{i:03d}_{file.filename}"
        input_path = training_dir / input_filename
        try:
            with open(input_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            image_paths.append(str(input_path))
        except Exception as e:
            logger.error(f"Error saving file {file.filename}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to save {file.filename}"
            )

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
# Reframe / Outpainting via ComfyUI
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/reframe")
async def reframe_image(
    image: UploadFile = File(...),
    target_width: int = Form(1280),
    target_height: int = Form(720),
    position: str = Form("center"),
    prompt: str = Form("seamless natural extension, high quality"),
    model: str = Form("CyberRealisticPony_v8.safetensors"),
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
    upload_path = UPLOAD_DIR / upload_filename

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
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Face Swap via ComfyUI (ReActor / InsightFace)
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/detect-faces")
async def detect_faces(image: UploadFile = File(...)):
    """
    Detect faces in an image for face swap.
    Returns list of detected face bounding boxes.
    """
    import cv2
    import numpy as np

    logger.info(f"👤 Detecting faces in {image.filename}...")

    try:
        content = await image.read()
        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        # Use OpenCV's built-in face detector
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        face_list = []
        for i, (x, y, w, h) in enumerate(faces):
            face_list.append(
                {
                    "index": i,
                    "bbox": {
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                    },
                    "confidence": 0.9,  # OpenCV doesn't provide confidence, placeholder
                }
            )

        logger.info(f"👤 Detected {len(face_list)} face(s)")

        return {"faces": face_list, "total": len(face_list)}

    except Exception as e:
        logger.error(f"❌ Face detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/face-swap")
async def face_swap(
    target: UploadFile = File(...),
    source: UploadFile = File(...),
    model: str = Form("inswapper"),
    enhance: str = Form("gfpgan"),
    strength: float = Form(1.0),
    blend: float = Form(0.8),
    face_index: int = Form(0),  # -1 = all faces
):
    """
    Face swap using ComfyUI ReActor node.

    Args:
        target: Image/video with face(s) to replace
        source: Image with source face
        model: Face swap model (inswapper, simswap)
        enhance: Post-processing (none, gfpgan, codeformer, both)
        strength: Swap strength 0-1
        blend: Edge blend amount
        face_index: Which face to swap (-1 for all)
    """
    logger.info(f"👤 Face swap: model={model}, enhance={enhance}, strength={strength}")

    client = get_comfyui_client()
    if not client or not client.is_available():
        raise HTTPException(status_code=503, detail="ComfyUI backend not available")

    # Save files
    target_filename = f"faceswap_target_{uuid.uuid4().hex[:8]}"
    source_filename = f"faceswap_source_{uuid.uuid4().hex[:8]}.png"

    is_video = target.content_type and target.content_type.startswith("video/")
    target_filename += ".mp4" if is_video else ".png"

    target_path = UPLOAD_DIR / target_filename
    source_path = UPLOAD_DIR / source_filename

    try:
        # Save target
        content = await target.read()
        with open(target_path, "wb") as f:
            f.write(content)

        # Save source
        content = await source.read()
        with open(source_path, "wb") as f:
            f.write(content)

        # Upload images to ComfyUI
        target_comfy = client.upload_image(str(target_path))
        source_comfy = client.upload_image(str(source_path))

        if not target_comfy or not source_comfy:
            raise HTTPException(
                status_code=500, detail="Failed to upload images to ComfyUI"
            )

        # Build ReActor workflow
        # Requires ComfyUI-ReActor custom node: https://github.com/Gourieff/comfyui-reactor-node
        workflow = {
            # Load target image
            "1": {
                "inputs": {"image": target_comfy, "upload": "image"},
                "class_type": "LoadImage",
            },
            # Load source face
            "2": {
                "inputs": {"image": source_comfy, "upload": "image"},
                "class_type": "LoadImage",
            },
            # ReActor face swap
            "3": {
                "inputs": {
                    "input_image": ["1", 0],
                    "source_image": ["2", 0],
                    "swap_model": "inswapper_128.onnx",
                    "facedetection": "retinaface_resnet50",
                    "face_restore_model": "GFPGANv1.4.pth"
                    if enhance in ["gfpgan", "both"]
                    else "none",
                    "face_restore_visibility": blend,
                    "codeformer_weight": 0.5
                    if enhance in ["codeformer", "both"]
                    else 0,
                    "console_log_level": 1,
                    "detect_gender_source": "no",
                    "detect_gender_input": "no",
                    "source_faces_index": "0",
                    "input_faces_index": str(face_index)
                    if face_index >= 0
                    else "0,1,2,3,4",
                },
                "class_type": "ReActorFaceSwap",
            },
            # Save result
            "4": {
                "inputs": {"filename_prefix": "oelala_faceswap", "images": ["3", 0]},
                "class_type": "SaveImage",
            },
        }

        prompt_id = client.queue_prompt(workflow)
        if not prompt_id:
            raise HTTPException(
                status_code=500, detail="Failed to queue face swap workflow"
            )

        return {
            "status": "queued",
            "prompt_id": prompt_id,
            "meta": {
                "model": model,
                "enhance": enhance,
                "strength": strength,
                "face_index": face_index,
                "is_video": is_video,
            },
        }

    except Exception as e:
        logger.error(f"❌ Face swap error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="192.168.1.2", port=7998, reload=True, log_level="info")
