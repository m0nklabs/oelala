#!/usr/bin/env python3
"""
Oelala Web Interface Backend
FastAPI application for AI Video Generation Pipeline
"""

import os
import sys
import uvicorn
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List
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
sys.path.append('/home/flip/oelala')  # Add oelala root directory

# GPU generators are disabled by default - ComfyUI handles all GPU work
# Set OELALA_LOAD_GPU_GENERATORS=1 to enable local torch-based generators
LOAD_GPU_GENERATORS = os.environ.get("OELALA_LOAD_GPU_GENERATORS", "0") == "1"

if LOAD_GPU_GENERATORS:
    # Wan2VideoGenerator - local torch-based generation
    try:
        from src.backend.wan2_generator import Wan2VideoGenerator
        print("✅ Wan2VideoGenerator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import Wan2VideoGenerator: {e}")
        Wan2VideoGenerator = None

    # SD3ImageGenerator - local torch-based generation
    try:
        from src.backend.sd3_generator import SD3ImageGenerator
        print("✅ SD3ImageGenerator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import SD3ImageGenerator: {e}")
        SD3ImageGenerator = None

    # RealVisXL Image Generator (SDXL RealVis V5.0)
    try:
        from src.backend.realvis_generator import RealVisXLImageGenerator
        print("✅ RealVisXLImageGenerator imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import RealVisXLImageGenerator: {e}")
        RealVisXLImageGenerator = None
else:
    print("ℹ️ GPU generators disabled - using ComfyUI backend only")
    Wan2VideoGenerator = None
    SD3ImageGenerator = None
    RealVisXLImageGenerator = None

# ComfyUI Client for Wan2.2 Q5 GGUF workflows
try:
    from src.backend.comfyui_client import ComfyUIClient, get_comfyui_client
    print("✅ ComfyUIClient imported successfully")
except ImportError as e:
    print(f"❌ Failed to import ComfyUIClient: {e}")
    ComfyUIClient = None
    get_comfyui_client = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Log Buffer for UI
log_buffer = deque(maxlen=1000) # Increased buffer size for shell output
progress_store = {}  # job_id -> {progress, status, message, updated_at}
ticker_store = {}    # job_id -> threading.Event to stop ticker

# Global debug switch for verbose backend traces
DEBUG_ENABLED = os.getenv("OELALA_DEBUG", "0") == "1"


def debug_log(message: str):
    """Emit debug logs when DEBUG_ENABLED is true."""
    if DEBUG_ENABLED:
        logger.info(f"🐛 {message}")


def start_progress_ticker(job_id: str, step: int = 5, interval: float = 2.0, ceiling: int = 95):
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
                "updated_at": datetime.now().isoformat()
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


def inject_png_workflow_metadata(image_path: str, workflow: dict, prompt_params: dict) -> bool:
    """
    Inject ComfyUI-compatible workflow metadata into a PNG file.
    This allows ComfyUI to read the workflow when opening the image.
    
    Args:
        image_path: Path to the PNG file
        workflow: The ComfyUI API workflow dict
        prompt_params: Additional prompt parameters for reference
    
    Returns:
        True if successful, False otherwise
    """
    try:
        img = Image.open(image_path)
        
        # Create PNG metadata
        metadata = PngInfo()
        
        # ComfyUI expects 'prompt' to contain the API workflow
        metadata.add_text("prompt", json.dumps(workflow))
        
        # Add extra info for reference
        metadata.add_text("workflow", json.dumps(workflow))  # Some versions look for this
        metadata.add_text("oelala_params", json.dumps(prompt_params))
        
        # Save with metadata
        img.save(image_path, pnginfo=metadata)
        logger.info(f"📝 Injected workflow metadata into {image_path}")
        return True
    except Exception as e:
        logger.warning(f"⚠️ Failed to inject PNG metadata: {e}")
        return False


class BufferHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            log_buffer.append({
                "timestamp": datetime.now().isoformat(),
                "level": record.levelname,
                "message": msg
            })
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
                clean_msg = message.replace('\r', '').strip()
                if clean_msg:
                    log_buffer.append({
                        "timestamp": datetime.now().isoformat(),
                        "level": self.level,
                        "message": clean_msg
                    })
        except Exception:
            # If writing to buffer fails, don't crash the app
            pass

    def flush(self):
        try:
            self.original_stream.flush()
        except Exception:
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
sys.stderr = StreamToBuffer(sys.stderr, "SHELL") # Use SHELL level for stderr (tqdm usually goes here)

# Add buffer handler to root logger only (module loggers propagate by default)
buffer_handler = BufferHandler()
buffer_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(buffer_handler)

# Attach to common libraries used by the generators so their INFO logs appear
for noisy_logger in ["diffusers", "transformers", "accelerate"]:
    logging.getLogger(noisy_logger).setLevel(logging.INFO)
    logging.getLogger(noisy_logger).addHandler(buffer_handler)

# Create FastAPI app
app = FastAPI(
    title="Oelala AI Video Generator",
    description="AI-powered video generation from images using Wan2.2",
    version="1.0.0"
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
    app.mount("/comfyui-output", StaticFiles(directory=str(COMFYUI_OUTPUT_DIR)), name="comfyui_output")

# Global generator instance
generator = None
image_generator = None
realvis_generator = None

@app.get("/logs")
async def get_logs():
    """Get recent server logs"""
    return list(log_buffer)


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
    """Initialize the Wan2.2 generator on startup"""
    global generator
    if Wan2VideoGenerator:
        try:
            # Use lightweight model for better memory efficiency
            model_type = os.getenv("MODEL_TYPE", "light")  # Default to light model
            # Use Diffusers-compatible model path
            model_path = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
            logger.info(f"Initializing generator with model type: {model_type}")
            generator = Wan2VideoGenerator(model_path=model_path, model_type=model_type)
            # Don't call load_model() for light model - __init__ already configured it
            logger.info("✅ Wan2.2 generator ready!")
        except Exception as e:
            logger.error(f"❌ Error initializing generator: {e}")
            logger.info("💡 The model will be loaded on first use instead")
    else:
        logger.error("❌ Wan2VideoGenerator not available")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Oelala AI Video Generator API",
        "version": "1.0.0",
        "status": "ready" if generator else "error",
        "endpoints": {
            "POST /generate": "Generate video from image",
            "POST /generate-text": "Generate video from text prompt",
            "POST /generate-image": "Generate image from text prompt",
            "GET /files/{filename}": "Download generated video",
            "GET /health": "Health check"
        }
    }

@app.get("/list-comfyui-media")
async def list_comfyui_media(type: str = "all", grouped: bool = False, include_metadata: bool = False):
    """List media files from ComfyUI output directory"""
    comfyui_output = Path("/home/flip/oelala/ComfyUI/output")
    
    if not comfyui_output.exists():
        return {"media": [], "stats": {"videos": 0, "images": 0}}
    
    media = []
    video_count = 0
    image_count = 0
    
    # Collect all media files
    for file_path in comfyui_output.iterdir():
        if not file_path.is_file():
            continue
            
        ext = file_path.suffix.lower()
        media_type = None
        
        if ext in ['.mp4', '.webm', '.mov', '.avi']:
            media_type = 'video'
            video_count += 1
        elif ext in ['.png', '.jpg', '.jpeg', '.webp']:
            media_type = 'image'
            image_count += 1
        else:
            continue
        
        # Filter by type if requested
        if type != 'all' and media_type != type:
            continue
        
        stat = file_path.stat()
        item = {
            "filename": file_path.name,
            "type": media_type,
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "url": f"/comfyui-output/{file_path.name}"
        }
        
        media.append(item)
    
    # Sort by modified time descending
    media.sort(key=lambda x: x['modified'], reverse=True)
    
    return {
        "media": media,
        "stats": {
            "videos": video_count,
            "images": image_count
        }
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
    
    return {
        "deleted": deleted,
        "errors": errors,
        "count": len(deleted)
    }


@app.get("/loras")
async def list_loras():
    """
    List available LoRA models from ComfyUI/models/loras folder.
    Returns LoRAs grouped by noise type (high/low) for Wan2.2 dual-pass workflow.
    """
    loras_dir = Path("/home/flip/oelala/ComfyUI/models/loras")
    
    if not loras_dir.exists():
        return {"loras": [], "high_noise": [], "low_noise": [], "general": []}
    
    all_loras = []
    high_noise = []
    low_noise = []
    general = []
    
    for lora_path in loras_dir.rglob("*.safetensors"):
        # Get relative path from loras folder
        rel_path = str(lora_path.relative_to(loras_dir))
        name = lora_path.stem
        
        lora_info = {
            "path": rel_path,
            "name": name,
            "size_mb": round(lora_path.stat().st_size / (1024 * 1024), 1)
        }
        all_loras.append(lora_info)
        
        # Categorize by noise type
        lower_name = name.lower()
        lower_path = rel_path.lower()
        
        if "high" in lower_name or "high" in lower_path or "_h_" in lower_name or "-h-" in lower_name:
            high_noise.append(lora_info)
        elif "low" in lower_name or "low" in lower_path or "_l_" in lower_name or "-l-" in lower_name:
            low_noise.append(lora_info)
        else:
            general.append(lora_info)
    
    # Sort by name
    all_loras.sort(key=lambda x: x["name"].lower())
    high_noise.sort(key=lambda x: x["name"].lower())
    low_noise.sort(key=lambda x: x["name"].lower())
    general.sort(key=lambda x: x["name"].lower())
    
    return {
        "loras": all_loras,
        "high_noise": high_noise,
        "low_noise": low_noise,
        "general": general,
        "count": len(all_loras)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if generator or image_generator else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": generator is not None,
        "image_model_loaded": image_generator is not None,
        "upload_dir": str(UPLOAD_DIR),
        "output_dir": str(OUTPUT_DIR)
    }

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
        logs_dir = Path('/home/flip/oelala/logs')
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / 'ui_client.log'

        entry = {
            'received_at': datetime.now().isoformat(),
            'payload': payload
        }

        # Append JSON line to log file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        logger.info(f"Received client log: {payload.get('level', 'info')} {payload.get('message', '')}")
        return {"success": True}
    except Exception as e:
        logger.error(f"Error saving client log: {e}")
        raise HTTPException(status_code=500, detail="Failed to save client log")

@app.post("/generate-image")
def generate_image(
    prompt: str = Form(...),
    aspect_ratio: str = Form("1:1"),
    mode: str = Form("normal"),
    output_filename: str = Form(""),
    job_id: str = Form(None),
    model: str = Form("sd3.5-large-int8")
):
    """
    Generate image from text prompt using selectable models (SD3.5 INT8 or RealVisXL V5.0)
    """
    global image_generator, realvis_generator
    
    logger.info(f"🚀 Received generate-image request: {prompt[:50]}... (model={model})")
    if not job_id:
        job_id = str(uuid.uuid4())

    progress_store[job_id] = {
        "job_id": job_id,
        "progress": 5,
        "status": "running",
        "updated_at": datetime.now().isoformat()
    }
    start_progress_ticker(job_id)

    # Select generator based on requested model
    model_key = (model or "sd3.5-large-int8").lower().strip()
    selected = None
    selected_name = "sd3.5-large-int8"

    if model_key in ["sd3", "sd3.5", "sd3.5-large", "sd3.5-large-int8", "sd35", "sd35-int8"]:
        if image_generator is None and SD3ImageGenerator:
            try:
                logger.info("Loading SD3.5 Image Generator on first use...")
                image_generator = SD3ImageGenerator()
            except Exception as e:
                logger.error(f"Failed to initialize SD3.5 image generator: {e}")
                raise HTTPException(status_code=503, detail="SD3.5 image generator initialization failed")
        if not image_generator:
            logger.error("SD3.5 image generator not available (import failed or crashed)")
            raise HTTPException(status_code=503, detail="SD3.5 image generator not available")
        selected = image_generator
        selected_name = "sd3.5-large-int8"

    elif model_key in ["realvis", "realvisxl", "realvisxl-v5", "realvisxl_v5", "realvisxl-v5.0", "realvisxl v5.0"]:
        if realvis_generator is None and RealVisXLImageGenerator:
            try:
                logger.info("Loading RealVisXL Image Generator on first use...")
                realvis_generator = RealVisXLImageGenerator()
            except Exception as e:
                logger.error(f"Failed to initialize RealVisXL image generator: {e}")
                raise HTTPException(status_code=503, detail="RealVisXL image generator initialization failed")
        if not realvis_generator:
            logger.error("RealVisXL image generator not available (import failed or crashed)")
            raise HTTPException(status_code=503, detail="RealVisXL image generator not available")
        selected = realvis_generator
        selected_name = "realvisxl-v5.0"

    else:
        logger.error(f"Unsupported model requested: {model}")
        raise HTTPException(status_code=400, detail=f"Unsupported model '{model}'")

    try:
        # Map aspect ratios to resolutions (approx 1MP)
        resolutions = {
            "1:1": (1024, 1024),
            "16:9": (1344, 768),
            "9:16": (768, 1344),
            "4:3": (1152, 864),
            "3:4": (864, 1152),
            "2:3": (832, 1216),
            "3:2": (1216, 832),
            "4:5": (896, 1120),
            "5:4": (1120, 896),
            "21:9": (1536, 640),
            "9:21": (640, 1536),
        }
        
        width, height = resolutions.get(aspect_ratio, (1024, 1024))
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not output_filename:
            output_filename = f"t2i_{timestamp}.png"
        
        if not output_filename.endswith('.png'):
            output_filename += '.png'
            
        output_path = OUTPUT_DIR / output_filename
        
        logger.info(f"Generating image: {prompt} ({width}x{height})")
        
        image = selected.generate(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=28
        )
        
        image.save(output_path)
        logger.info(f"Image saved to {output_path}")

        progress_store[job_id] = {
            "job_id": job_id,
            "progress": 100,
            "status": "done",
            "url": f"/files/{output_filename}",
            "updated_at": datetime.now().isoformat()
        }
        stop_progress_ticker(job_id)
        
        return {
            "status": "success",
            "url": f"/files/{output_filename}",
            "filename": output_filename,
            "meta": {
                "prompt": prompt,
                "width": width,
                "height": height,
                "model": selected_name
            },
            "job_id": job_id
        }
        
    except Exception as e:
        logger.error(f"Image generation failed: {e}")
        progress_store[job_id] = {
            "job_id": job_id,
            "progress": progress_store.get(job_id, {}).get("progress", 0),
            "status": "failed",
            "message": str(e),
            "updated_at": datetime.now().isoformat()
        }
        stop_progress_ticker(job_id)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_video(
    file: UploadFile = File(...),
    prompt: str = Form(""),
    num_frames: int = Form(16, description="Number of frames in video"),
    output_filename: str = Form("", description="Custom output filename"),
    resolution: str = Form("720p", description="Video resolution: 480p, 720p, 1080p"),
    fps: int = Form(16, description="Frames per second: 8, 12, 16, 24"),
    aspect_ratio: str = Form("16:9", description="Video aspect ratio")
):
    """
    Generate video from uploaded image

    Args:
        file: Image file to process
        prompt: Text prompt to guide generation
        num_frames: Number of frames in output video
        output_filename: Custom name for output file
    """
    global generator

    # Initialize generator if not already done
    if generator is None and Wan2VideoGenerator:
        try:
            logger.info("Loading Wan2.2 generator on first use...")
            generator = Wan2VideoGenerator()
        except Exception as e:
            logger.error(f"Failed to initialize generator: {e}")
            raise HTTPException(status_code=503, detail="Video generator initialization failed")

    if not generator:
        # If generator isn't available but placeholder forcing is enabled, allow placeholder flow
        if os.environ.get("OELALA_FORCE_LORA_PLACEHOLDER", "0") == "1":
            logger.info("Generator unavailable but placeholder flow is enabled; will create placeholder artifact")
        else:
            raise HTTPException(status_code=503, detail="Video generator not available")

    # Load model if not already loaded (only when generator exists)
    if generator and (not hasattr(generator, 'pipeline') or generator.pipeline is None):
        logger.info("🔄 Attempting to load Wan2.2 model...")
        logger.info(f"📊 Generator object: {type(generator)}")
        logger.info(f"🔧 Generator attributes: {dir(generator)}")

        success = generator.load_model()
        logger.info(f"📈 Model loading result: {success}")

        if not success:
            logger.error("❌ Model loading failed - check Wan2VideoGenerator logs above")
            logger.error("🔍 Troubleshooting:")
            logger.error("   1. Wan2.2 pipeline may not be available in current diffusers version")
            logger.error("   2. Check GPU memory and CUDA availability")
            logger.error("   3. Consider using alternative video generation models")
            raise HTTPException(status_code=503, detail="Failed to load Wan2.2 model")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = f"{timestamp}_{file.filename}"
    input_path = UPLOAD_DIR / input_filename

    # Save uploaded file
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Generate output filename
    if not output_filename:
        output_filename = f"video_{timestamp}.mp4"
    elif not output_filename.endswith(".mp4"):
        output_filename += ".mp4"

    output_path = OUTPUT_DIR / output_filename

    try:
        logger.info(f"Generating video: {input_path} -> {output_path}")
        logger.info(f"Prompt: '{prompt}', Frames: {num_frames}, FPS: {fps}, Resolution: {resolution}")

        # Generate video
        result = generator.generate_video_from_image(
            image_path=str(input_path),
            prompt=prompt,
            output_path=str(output_path),
            num_frames=num_frames
        )

        if result:
            # Return video info
            return {
                "success": True,
                "message": "Video generated successfully",
                "input_image": input_filename,
                "output_video": output_filename,
                "video_url": f"/files/{output_filename}",
                "video_path": str(output_path),
                "prompt": prompt,
                "num_frames": num_frames,
                "timestamp": timestamp
            }
        else:
            raise HTTPException(status_code=500, detail="Video generation failed")

    except Exception as e:
        logger.error(f"Error generating video: {e}")
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")


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
    lora_high_noise: str = Form("", description="LoRA for high noise model (path relative to loras folder)"),
    lora_low_noise: str = Form("", description="LoRA for low noise model (path relative to loras folder)"),
    lora_strength: float = Form(1.0, description="LoRA strength (0.0 - 2.0)")
):
    """
    Generate Wan2.2 I2V video via ComfyUI with DisTorch2 Dual-Pass workflow.
    
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
            detail="ComfyUI not running. Start with: cd ~/oelala/ComfyUI && python main.py --listen"
        )
    
    # Wan2.2 requires num_frames in format 4k+1 (5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, ...)
    # Round to nearest valid value
    k = round((num_frames - 1) / 4)
    k = max(1, k)  # Minimum k=1 gives 5 frames
    num_frames = 4 * k + 1
    logger.info(f"🎞️ Adjusted num_frames to Wan2.2 format: {num_frames} (4*{k}+1)")
    
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
        "lora_high_noise": lora_high_noise or None,
        "lora_low_noise": lora_low_noise or None,
        "lora_strength": lora_strength,
    }
    inject_png_workflow_metadata(str(input_path), workflow, prompt_params)

    try:
        logger.info(f"🎬 Starting Wan2.2 ComfyUI generation")
        logger.info(f"   📐 Resolution: {resolution}, Aspect: {aspect_ratio}")
        logger.info(f"   🎞️ Frames: {num_frames}, FPS: {fps}")
        logger.info(f"   ⚙️ Steps: {steps}, CFG: {cfg}, Seed: {seed}")
        if lora_high_noise or lora_low_noise:
            logger.info(f"   🎨 LoRA: H={lora_high_noise or 'none'}, L={lora_low_noise or 'none'} @ {lora_strength}")
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
                lora_high_noise=lora_high_noise if lora_high_noise else None,
                lora_low_noise=lora_low_noise if lora_low_noise else None,
                lora_strength=lora_strength
            )
        )
        
        if result_path and Path(result_path).exists():
            # Copy to expected output path if different
            final_output = OUTPUT_DIR / output_filename
            if str(result_path) != str(final_output):
                shutil.copy(result_path, final_output)
                result_path = str(final_output)
            
            return {
                "success": True,
                "message": "Wan2.2 video generated via ComfyUI",
                "input_image": input_filename,
                "output_video": output_filename,
                "video_url": f"/files/{output_filename}",
                "video_path": result_path,
                "prompt": prompt,
                "num_frames": num_frames,
                "fps": fps,
                "resolution": resolution,
                "aspect_ratio": aspect_ratio,
                "steps": steps,
                "cfg": cfg,
                "seed": seed,
                "timestamp": timestamp,
                "backend": "comfyui",
                "model": "wan2.2_i2v_low_noise_14B_Q5_K_S"
            }
        else:
            raise HTTPException(status_code=500, detail="ComfyUI video generation returned no output")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ ComfyUI generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Wan2.2 ComfyUI generation failed: {str(e)}")


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
                "model": "wan2.2_i2v_low_noise_14B_Q5_K_S.gguf"
            }
        except Exception as e:
            return {"available": True, "host": comfyui.host, "port": comfyui.port, "stats_error": str(e)}
    else:
        return {
            "available": False,
            "host": comfyui.host,
            "port": comfyui.port,
            "suggestion": "Start ComfyUI: cd ~/oelala/ComfyUI && python main.py --listen"
        }


@app.post("/generate-text")
async def generate_text_video(
    prompt: str = Form(..., description="Text description of the video to generate"),
    num_frames: int = Form(16, description="Number of frames in video"),
    model_type: str = Form("light", description="Model type to use: light, svd, wan2.2"),
    output_filename: str = Form("", description="Custom output filename"),
    resolution: str = Form("720p", description="Video resolution: 480p, 720p, 1080p"),
    fps: int = Form(16, description="Frames per second: 8, 12, 16, 24"),
    aspect_ratio: str = Form("16:9", description="Video aspect ratio")
):
    """
    Generate video from text prompt only

    Args:
        prompt: Text description of the video to generate
        num_frames: Number of frames in output video
        output_filename: Custom name for output file
    """
    global generator

    if not prompt or len(prompt.strip()) == 0:
        raise HTTPException(status_code=400, detail="Prompt is required")

    # Map frontend mode values to backend model types
    mode_mapping = {
        "🚀 Turbo (Light)": "light",
        "⚡ Standard (SVD)": "svd", 
        "🌟 Quality (Wan2.2)": "wan2.2",
        "light": "light",
        "svd": "svd",
        "wan2.2": "wan2.2"
    }
    requested_model_type = mode_mapping.get(model_type, model_type)
    logger.info(f"🎯 Requested model_type: {model_type} → {requested_model_type}")

    # Initialize or re-initialize generator if model type changed
    if generator is None or generator.model_type != requested_model_type:
        if Wan2VideoGenerator:
            try:
                if generator is not None:
                    logger.info(f"🔄 Switching model from '{generator.model_type}' to '{requested_model_type}'")
                else:
                    logger.info(f"🆕 Initializing generator with model_type: {requested_model_type}")
                generator = Wan2VideoGenerator(model_type=requested_model_type)
            except Exception as e:
                logger.error(f"Failed to initialize generator: {e}")
                raise HTTPException(status_code=503, detail="Video generator initialization failed")

    if not generator:
        raise HTTPException(status_code=503, detail="Video generator not available")

    # Load model if not already loaded (skip for light model - already initialized)
    if generator.model_type != "light" and (not hasattr(generator, 'pipeline') or generator.pipeline is None):
        logger.info("🔄 Attempting to load model for text-to-video...")
        success = generator.load_model()
        if not success:
            raise HTTPException(status_code=503, detail="Failed to load model")

    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_filename:
        output_filename = f"{timestamp}_{output_filename}.mp4"
    else:
        output_filename = f"{timestamp}_text_video.mp4"

    output_path = OUTPUT_DIR / output_filename

    try:
        logger.info(f"🎬 Generating text-to-video with prompt: {prompt}")
        logger.info(f"📊 Settings: {num_frames} frames, {fps} fps, {resolution}, aspect {aspect_ratio}")

        # Try text-to-video generation
        result = generator.generate_text_video(
            prompt=prompt,
            output_path=str(output_path),
            num_frames=num_frames
        )

        if result:
            return {
                "success": True,
                "message": "Text-to-video generated successfully",
                "output_video": output_filename,
                "video_url": f"/files/{output_filename}",
                "video_path": str(output_path),
                "prompt": prompt,
                "num_frames": num_frames,
                "timestamp": timestamp,
                "type": "text-to-video"
            }
        else:
            raise HTTPException(status_code=500, detail="Text-to-video generation failed")

    except Exception as e:
        logger.error(f"Error generating text-to-video: {e}")
        raise HTTPException(status_code=500, detail=f"Text-to-video generation failed: {str(e)}")

@app.post("/generate-pose")
async def generate_pose_video(
    file: UploadFile = File(...),
    num_frames: int = Form(16, description="Number of frames in video"),
    output_filename: str = Form("", description="Custom output filename")
):
    """
    Generate pose-guided video from uploaded image using OpenPose

    Args:
        file: Image file to process
        num_frames: Number of frames in output video
        output_filename: Custom name for output file
    """
    global generator

    # Initialize generator if not already done
    if generator is None and Wan2VideoGenerator:
        try:
            logger.info("Loading Wan2.2 generator on first use...")
            generator = Wan2VideoGenerator()
        except Exception as e:
            logger.error(f"Failed to initialize generator: {e}")
            raise HTTPException(status_code=503, detail="Video generator initialization failed")

    if not generator:
        # Generator is unavailable; continue and create a placeholder artifact instead of failing.
        logger.info("Generator unavailable; continuing with placeholder LoRA artifact creation")

    # Load model if not already loaded
    if not hasattr(generator, 'pipeline') or generator.pipeline is None:
        logger.info("Loading Wan2.2 model...")
        if not generator.load_model():
            raise HTTPException(status_code=503, detail="Failed to load Wan2.2 model")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = f"{timestamp}_{file.filename}"
    input_path = UPLOAD_DIR / input_filename

    # Save uploaded file
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded file")

    # Generate output filename
    if not output_filename:
        output_filename = f"pose_video_{timestamp}.mp4"
    elif not output_filename.endswith(".mp4"):
        output_filename += ".mp4"

    output_path = OUTPUT_DIR / output_filename

    try:
        logger.info(f"Generating pose-guided video: {input_path} -> {output_path}")
        logger.info(f"Frames: {num_frames}")

        # Generate pose-guided video
        result = generator.generate_pose_guided_video(
            image_path=str(input_path),
            output_path=str(output_path),
            num_frames=num_frames
        )

        if result:
            # Return video info
            return {
                "success": True,
                "message": "Pose-guided video generated successfully",
                "input_image": input_filename,
                "output_video": output_filename,
                "video_url": f"/videos/{output_filename}",
                "num_frames": num_frames,
                "timestamp": timestamp,
                "type": "pose-guided"
            }
        else:
            raise HTTPException(status_code=500, detail="Pose-guided video generation failed")

    except Exception as e:
        logger.error(f"Error generating pose-guided video: {e}")
        raise HTTPException(status_code=500, detail=f"Pose-guided video generation failed: {str(e)}")

@app.get("/videos/{filename}")
async def get_video(filename: str):
    """Download generated video file"""
    file_path = OUTPUT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Video file not found")

    return FileResponse(
        path=file_path,
        media_type="video/mp4",
        filename=filename
    )

@app.get("/images/{filename}")
async def get_image(filename: str):
    """Download uploaded image file"""
    file_path = UPLOAD_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(
        path=file_path,
        media_type="image/jpeg",
        filename=filename
    )

@app.get("/list-videos")
async def list_videos():
    """List all generated videos"""
    videos = []
    for file_path in OUTPUT_DIR.glob("*.mp4"):
        stat = file_path.stat()
        videos.append({
            "filename": file_path.name,
            "size": stat.st_size,
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "url": f"/videos/{file_path.name}"
        })

    return {"videos": videos, "count": len(videos)}

@app.post("/train-lora")
async def train_lora_model(
    files: List[UploadFile] = File(...),
    model_name: str = Form("", description="Name for the trained model"),
    num_epochs: int = Form(10, description="Number of training epochs"),
    learning_rate: float = Form(1e-4, description="Learning rate")
):
    """
    Train LoRA adapter on multiple uploaded images for consistent avatar generation

    Args:
        files: List of image files for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for training
        output_name: Custom name for the trained model
    """
    global generator
    global Wan2VideoGenerator

    # Initialize generator if not already done
    if generator is None:
        # Try to import Wan2VideoGenerator lazily
        if Wan2VideoGenerator is None:
            try:
                from wan2_generator import Wan2VideoGenerator as _W2
                Wan2VideoGenerator = _W2
            except Exception as e:
                logger.warning(f"Wan2VideoGenerator import failed at request time: {e}")

        if Wan2VideoGenerator:
            try:
                logger.info("Instantiating Wan2.2 generator for LoRA training...")
                generator = Wan2VideoGenerator()
            except Exception as e:
                logger.error(f"Failed to initialize generator: {e}")
                # Continue: allow placeholder flow if forced
        else:
            logger.info("Wan2VideoGenerator not available; will attempt placeholder flow if allowed")

    if not generator:
        raise HTTPException(status_code=503, detail="Video generator not available")

    # Load model if not already loaded
    if not hasattr(generator, 'pipeline') or generator.pipeline is None:
        logger.info("Loading Wan2.2 model...")
        if not generator.load_model():
            raise HTTPException(status_code=503, detail="Failed to load Wan2.2 model")

    # Validate files
    if len(files) < 2:
        raise HTTPException(status_code=400, detail="At least 2 images required for LoRA training")

    for file in files:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image")

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
            raise HTTPException(status_code=500, detail=f"Failed to save {file.filename}")

    # Generate output name
    if not model_name:
        model_name = f"lora_model_{timestamp}"

    output_dir = OUTPUT_DIR / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        logger.info(f"Starting LoRA training: {len(image_paths)} images, {num_epochs} epochs")
        logger.info(f"Training ID: {training_id}, Output: {output_dir}")

        # If generator is available, run real training; otherwise, create a placeholder artifact
        if generator:
            lora_path = generator.train_lora(
                image_paths=image_paths,
                output_dir=str(output_dir),
                num_epochs=num_epochs,
                learning_rate=learning_rate
            )

            if not lora_path:
                raise HTTPException(status_code=500, detail="LoRA training failed")

        else:
            # Create placeholder artifact
            os.makedirs(output_dir, exist_ok=True)
            placeholder = {
                "note": "LoRA training skipped because generator unavailable; placeholder created",
                "image_count": len(image_paths),
                "training_id": training_id,
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
            placeholder_path = output_dir / "lora_placeholder.json"
            with open(placeholder_path, "w") as fh:
                json.dump(placeholder, fh, indent=2)
            lora_path = str(placeholder_path)

        # Return training results
        return {
            "success": True,
            "message": "LoRA training completed successfully",
            "training_id": training_id,
            "lora_path": lora_path,
            "num_images": len(image_paths),
            "num_epochs": num_epochs,
            "learning_rate": learning_rate,
            "model_name": model_name,
            "status": "completed",
            "timestamp": timestamp
        }

    except Exception as e:
        logger.error(f"Error during LoRA training: {e}")
        raise HTTPException(status_code=500, detail=f"LoRA training failed: {str(e)}")


@app.post("/train-lora-placeholder")
async def train_lora_placeholder(
    files: List[UploadFile] = File(...),
    model_name: str = Form("", description="Name for the trained model")
):
    """
    Create a LoRA placeholder artifact from uploaded images. This endpoint does not require the model stack.
    """
    # Validate files
    if len(files) < 1:
        raise HTTPException(status_code=400, detail="At least 1 image required to create placeholder")

    import mimetypes
    for file in files:
        ct = getattr(file, 'content_type', None)
        logger.info(f"Placeholder upload file: {file.filename}, content_type={ct}")
        if ct and ct.startswith('image/'):
            continue
        # If content_type missing, do a lightweight filename-based check
        ext = os.path.splitext(file.filename)[1].lower()
        if ext in ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.gif']:
            continue
        raise HTTPException(status_code=400, detail=f"File {file.filename} must be an image")

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
            raise HTTPException(status_code=500, detail=f"Failed to save {file.filename}")

    output_dir = OUTPUT_DIR / model_name
    os.makedirs(output_dir, exist_ok=True)

    placeholder = {
        "note": "This is a placeholder LoRA artifact",
        "training_id": training_id,
        "image_count": len(image_paths),
        "images": [os.path.basename(p) for p in image_paths],
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    placeholder_path = output_dir / "lora_placeholder.json"
    with open(placeholder_path, "w") as fh:
        json.dump(placeholder, fh, indent=2)

    return {"success": True, "lora_path": str(placeholder_path), "training_id": training_id}

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="192.168.1.2",
            port=7998,
        reload=True,
        log_level="info"
    )
