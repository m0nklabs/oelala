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
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
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

# WebSocket connections for live log streaming
log_subscribers: set[WebSocket] = set()

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
        existing_info = img.info if hasattr(img, 'info') else {}
        
        # Try to extract original T2I prompt from existing workflow
        original_t2i_prompt = None
        if 'prompt' in existing_info:
            try:
                existing_wf = json.loads(existing_info['prompt'])
                for node_id, node in existing_wf.items():
                    if isinstance(node, dict):
                        inputs = node.get('inputs', {})
                        # CLIPTextEncode has long T2I prompts
                        if 'text' in inputs and isinstance(inputs['text'], str):
                            text = inputs['text']
                            if len(text) > 50:  # Long prompts are likely T2I
                                original_t2i_prompt = text
                                break
            except json.JSONDecodeError:
                pass
        
        # Create PNG metadata
        metadata = PngInfo()
        
        # ComfyUI expects 'prompt' to contain the API workflow
        metadata.add_text("prompt", json.dumps(workflow))
        
        # Add extra info for reference
        metadata.add_text("workflow", json.dumps(workflow))  # Some versions look for this
        
        # Preserve original T2I prompt if we found one
        if original_t2i_prompt:
            prompt_params = dict(prompt_params)  # Don't modify original
            prompt_params['original_t2i_prompt'] = original_t2i_prompt
            logger.info(f"📝 Preserved original T2I prompt ({len(original_t2i_prompt)} chars)")
        
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
                "message": msg
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
                clean_msg = message.replace('\r', '').strip()
                if clean_msg:
                    log_entry = {
                        "timestamp": datetime.now().isoformat(),
                        "level": self.level,
                        "message": clean_msg
                    }
                    log_buffer.append(log_entry)
                    # Broadcast to WebSocket subscribers
                    if log_subscribers:
                        try:
                            asyncio.get_event_loop().create_task(broadcast_log(log_entry))
                        except RuntimeError:
                            pass
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
        logger.info(f"📡 Log WebSocket disconnected (remaining: {len(log_subscribers)})")


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
        logger.info("ℹ️ ComfyUI backend active - GPU generators disabled (set OELALA_LOAD_GPU_GENERATORS=1 to enable)")

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
async def list_comfyui_media(type: str = "all", grouped: bool = False, include_metadata: bool = False, hide_start_images: bool = True):
    """List media files from ComfyUI output directory
    
    Args:
        type: Filter by media type ('all', 'video', 'image')
        grouped: Group videos with source images (not implemented yet)
        include_metadata: Include PNG metadata in response
        hide_start_images: Hide images that are start frames for videos (default True)
    """
    comfyui_output = Path("/home/flip/oelala/ComfyUI/output")
    
    if not comfyui_output.exists():
        return {"media": [], "stats": {"videos": 0, "images": 0}}
    
    media = []
    video_count = 0
    image_count = 0
    
    # First pass: collect all files and extract timestamps from videos
    video_timestamps = set()
    all_files = []
    
    for file_path in comfyui_output.iterdir():
        if not file_path.is_file():
            continue
        ext = file_path.suffix.lower()
        if ext in ['.mp4', '.webm', '.mov', '.avi']:
            # Extract timestamp from video filename (e.g., oelala_20260102_075057)
            import re
            match = re.search(r'(\d{8}_\d{6})', file_path.name)
            if match:
                video_timestamps.add(match.group(1))
            all_files.append((file_path, 'video'))
        elif ext in ['.png', '.jpg', '.jpeg', '.webp']:
            all_files.append((file_path, 'image'))
    
    # Second pass: process files and mark start images
    for file_path, media_type in all_files:
        ext = file_path.suffix.lower()
        
        if media_type == 'video':
            video_count += 1
        else:
            image_count += 1
        
        # Filter by type if requested
        if type != 'all' and media_type != type:
            continue
        
        # Check if this image is a start image for a video
        is_start_image = False
        if media_type == 'image' and hide_start_images:
            import re
            match = re.search(r'(\d{8}_\d{6})', file_path.name)
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
        if include_metadata and media_type == 'image':
            try:
                from PIL import Image
                img = Image.open(file_path)
                metadata = {"has_metadata": False}
                
                if hasattr(img, 'info') and img.info:
                    # Try to extract prompt from ComfyUI workflow JSON
                    if 'prompt' in img.info:
                        metadata["has_metadata"] = True
                        try:
                            workflow = json.loads(img.info['prompt'])
                            # Collect all text prompts for later analysis
                            all_texts = []
                            loras_found = []
                            
                            # Extract prompts from various node types
                            for node_id, node in workflow.items():
                                if isinstance(node, dict):
                                    inputs = node.get('inputs', {})
                                    class_type = node.get('class_type', '')
                                    
                                    # Wan2.2 / standard positive_prompt
                                    if 'positive_prompt' in inputs and isinstance(inputs['positive_prompt'], str):
                                        text = inputs['positive_prompt'].strip()
                                        if len(text) > 20:
                                            metadata['positive_prompt'] = text
                                    # Negative prompt
                                    if 'negative_prompt' in inputs and isinstance(inputs['negative_prompt'], str):
                                        text = inputs['negative_prompt'].strip()
                                        if text:
                                            metadata['negative_prompt'] = text
                                    
                                    # CLIPTextEncode text - collect all for analysis
                                    if 'text' in inputs and isinstance(inputs['text'], str):
                                        text = inputs['text'].strip()
                                        if len(text) > 10:
                                            all_texts.append({'text': text, 'class_type': class_type, 'node_id': node_id})
                                    
                                    # Extract generation params
                                    if 'steps' in inputs and isinstance(inputs['steps'], (int, float)):
                                        metadata['steps'] = int(inputs['steps'])
                                    if 'cfg' in inputs and isinstance(inputs['cfg'], (int, float)):
                                        metadata['cfg'] = float(inputs['cfg'])
                                    if 'seed' in inputs and isinstance(inputs['seed'], (int, float)):
                                        metadata['seed'] = int(inputs['seed'])
                                    
                                    # Extract sampler info
                                    if 'sampler_name' in inputs and isinstance(inputs['sampler_name'], str):
                                        metadata['sampler'] = inputs['sampler_name']
                                    if 'scheduler' in inputs and isinstance(inputs['scheduler'], str):
                                        metadata['scheduler'] = inputs['scheduler']
                                    
                                    # Extract resolution from EmptyLatentImage or similar
                                    if 'width' in inputs and 'height' in inputs:
                                        w = inputs.get('width')
                                        h = inputs.get('height')
                                        if isinstance(w, (int, float)) and isinstance(h, (int, float)):
                                            metadata['width'] = int(w)
                                            metadata['height'] = int(h)
                                    
                                    # Extract LoRA info
                                    if 'LoraLoader' in class_type or 'lora' in class_type.lower():
                                        lora_name = inputs.get('lora_name', '')
                                        lora_strength = inputs.get('strength_model', inputs.get('strength', 1.0))
                                        if lora_name:
                                            loras_found.append({
                                                'name': lora_name,
                                                'strength': float(lora_strength) if isinstance(lora_strength, (int, float)) else 1.0
                                            })
                                    
                                    # Wan2.2 specific LoRA loader
                                    if 'WanVideoLoraSelect' in class_type or 'lora_high' in inputs or 'lora_low' in inputs:
                                        for key in ['lora_high', 'lora_low', 'lora_name']:
                                            if key in inputs and inputs[key]:
                                                lora_name = inputs[key]
                                                if isinstance(lora_name, str) and lora_name not in ['None', 'none', '']:
                                                    strength = inputs.get('strength', inputs.get('lora_strength', 1.0))
                                                    loras_found.append({
                                                        'name': lora_name,
                                                        'strength': float(strength) if isinstance(strength, (int, float)) else 1.0
                                                    })
                                    
                                    # Extract model/checkpoint info
                                    if 'ckpt_name' in inputs and isinstance(inputs['ckpt_name'], str):
                                        metadata['model'] = inputs['ckpt_name']
                                    if 'unet_name' in inputs and isinstance(inputs['unet_name'], str):
                                        if not metadata.get('model'):
                                            metadata['model'] = inputs['unet_name']
                            
                            # Store unique LoRAs
                            if loras_found:
                                # Deduplicate by name
                                seen = set()
                                unique_loras = []
                                for lora in loras_found:
                                    if lora['name'] not in seen:
                                        seen.add(lora['name'])
                                        unique_loras.append(lora)
                                metadata['loras'] = unique_loras
                            
                            # If no positive_prompt found, analyze CLIPTextEncode texts
                            if not metadata.get('positive_prompt') and all_texts:
                                # Heuristics: negative prompts often contain these keywords
                                negative_indicators = ['worst', 'bad', 'ugly', 'blurry', 'low quality', '低质量', '最差', 'deformed']
                                
                                for item_text in all_texts:
                                    text = item_text['text']
                                    text_lower = text.lower()
                                    
                                    # Check if it looks like a negative prompt
                                    is_negative = any(ind in text_lower for ind in negative_indicators)
                                    
                                    if is_negative and not metadata.get('negative_prompt'):
                                        metadata['negative_prompt'] = text
                                    elif not is_negative and not metadata.get('positive_prompt'):
                                        metadata['positive_prompt'] = text
                                
                                # Fallback: if still no positive, use first text
                                if not metadata.get('positive_prompt') and all_texts:
                                    metadata['positive_prompt'] = all_texts[0]['text']
                                    
                        except json.JSONDecodeError:
                            pass
                    
                    # Oelala params format
                    if 'oelala_params' in img.info:
                        metadata["has_metadata"] = True
                        try:
                            params = json.loads(img.info['oelala_params'])
                            if params.get('prompt'):
                                metadata['positive_prompt'] = params['prompt']
                            if params.get('negative_prompt'):
                                metadata['negative_prompt'] = params['negative_prompt']
                            if params.get('steps'):
                                metadata['steps'] = params['steps']
                            if params.get('cfg'):
                                metadata['cfg'] = params['cfg']
                            if params.get('seed'):
                                metadata['seed'] = params['seed']
                        except json.JSONDecodeError:
                            pass
                
                img.close()
                item["metadata"] = metadata
            except Exception as e:
                item["metadata"] = {"has_metadata": False, "error": str(e)}
        
        # For videos, try to find associated PNG with same timestamp or base name
        if include_metadata and media_type == 'video':
            import re
            metadata_found = False
            
            def extract_metadata_from_png(png_path):
                """Extract full metadata from PNG file including LoRAs, sampler, model etc."""
                from PIL import Image
                img = Image.open(png_path)
                metadata = {"has_metadata": False}
                
                if hasattr(img, 'info') and 'prompt' in img.info:
                    metadata["has_metadata"] = True
                    try:
                        workflow = json.loads(img.info['prompt'])
                        all_texts = []
                        loras_found = []
                        
                        for node_id, node in workflow.items():
                            if isinstance(node, dict):
                                inputs = node.get('inputs', {})
                                class_type = node.get('class_type', '')
                                
                                # Wan2.2 / standard positive_prompt
                                if 'positive_prompt' in inputs and isinstance(inputs['positive_prompt'], str):
                                    text = inputs['positive_prompt'].strip()
                                    if len(text) > 20:
                                        metadata['positive_prompt'] = text
                                if 'negative_prompt' in inputs and isinstance(inputs['negative_prompt'], str):
                                    text = inputs['negative_prompt'].strip()
                                    if text:
                                        metadata['negative_prompt'] = text
                                
                                # CLIPTextEncode text
                                if 'text' in inputs and isinstance(inputs['text'], str):
                                    text = inputs['text'].strip()
                                    if len(text) > 10:
                                        all_texts.append({'text': text, 'class_type': class_type})
                                
                                # Generation params
                                if 'steps' in inputs and isinstance(inputs['steps'], (int, float)):
                                    metadata['steps'] = int(inputs['steps'])
                                if 'cfg' in inputs and isinstance(inputs['cfg'], (int, float)):
                                    metadata['cfg'] = float(inputs['cfg'])
                                if 'seed' in inputs and isinstance(inputs['seed'], (int, float)):
                                    metadata['seed'] = int(inputs['seed'])
                                
                                # Sampler info
                                if 'sampler_name' in inputs and isinstance(inputs['sampler_name'], str):
                                    metadata['sampler'] = inputs['sampler_name']
                                if 'scheduler' in inputs and isinstance(inputs['scheduler'], str):
                                    metadata['scheduler'] = inputs['scheduler']
                                
                                # Resolution
                                if 'width' in inputs and 'height' in inputs:
                                    w, h = inputs.get('width'), inputs.get('height')
                                    if isinstance(w, (int, float)) and isinstance(h, (int, float)):
                                        metadata['width'] = int(w)
                                        metadata['height'] = int(h)
                                
                                # LoRA info
                                if 'LoraLoader' in class_type or 'lora' in class_type.lower():
                                    lora_name = inputs.get('lora_name', '')
                                    lora_strength = inputs.get('strength_model', inputs.get('strength', 1.0))
                                    if lora_name:
                                        loras_found.append({
                                            'name': lora_name,
                                            'strength': float(lora_strength) if isinstance(lora_strength, (int, float)) else 1.0
                                        })
                                
                                # Wan2.2 LoRA loader
                                if 'WanVideoLoraSelect' in class_type or 'lora_high' in inputs or 'lora_low' in inputs:
                                    for key in ['lora_high', 'lora_low', 'lora_name']:
                                        if key in inputs and inputs[key]:
                                            lora_name = inputs[key]
                                            if isinstance(lora_name, str) and lora_name not in ['None', 'none', '']:
                                                strength = inputs.get('strength', inputs.get('lora_strength', 1.0))
                                                loras_found.append({
                                                    'name': lora_name,
                                                    'strength': float(strength) if isinstance(strength, (int, float)) else 1.0
                                                })
                                
                                # Model/checkpoint info
                                if 'ckpt_name' in inputs and isinstance(inputs['ckpt_name'], str):
                                    metadata['model'] = inputs['ckpt_name']
                                if 'unet_name' in inputs and isinstance(inputs['unet_name'], str):
                                    if not metadata.get('model'):
                                        metadata['model'] = inputs['unet_name']
                        
                        # Store unique LoRAs
                        if loras_found:
                            seen = set()
                            unique_loras = []
                            for lora in loras_found:
                                if lora['name'] not in seen:
                                    seen.add(lora['name'])
                                    unique_loras.append(lora)
                            metadata['loras'] = unique_loras
                        
                        # Analyze CLIPTextEncode texts if no positive_prompt found
                        if not metadata.get('positive_prompt') and all_texts:
                            negative_indicators = ['worst', 'bad', 'ugly', 'blurry', 'low quality', '低质量', '最差', 'deformed']
                            for item in all_texts:
                                text = item['text']
                                text_lower = text.lower()
                                is_negative = any(ind in text_lower for ind in negative_indicators)
                                if is_negative and not metadata.get('negative_prompt'):
                                    metadata['negative_prompt'] = text
                                elif not is_negative and not metadata.get('positive_prompt'):
                                    metadata['positive_prompt'] = text
                            if not metadata.get('positive_prompt') and all_texts:
                                metadata['positive_prompt'] = all_texts[0]['text']
                                
                    except json.JSONDecodeError:
                        pass
                
                img.close()
                return metadata
            
            # Method 1: Look for PNG with same timestamp
            match = re.search(r'(\d{8}_\d{6})', file_path.name)
            if match:
                timestamp = match.group(1)
                for png_file in comfyui_output.glob(f"*{timestamp}*.png"):
                    try:
                        metadata = extract_metadata_from_png(png_file)
                        item["metadata"] = metadata
                        metadata_found = True
                        break  # Use first matching PNG
                    except Exception:
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
                            pass
            
            if not metadata_found:
                item["metadata"] = {"has_metadata": False}
        
        media.append(item)
    
    # Sort by modified time descending
    media.sort(key=lambda x: x['modified'], reverse=True)
    
    return {
        "media": media,
        "videos": video_count,
        "images": image_count,
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
        return {"loras": [], "high_noise": [], "low_noise": [], "general": [], "by_category": {}}
    
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
        
        lora_info = {
            "path": rel_path,
            "name": name,
            "category": category,
            "size_mb": round(lora_path.stat().st_size / (1024 * 1024), 1)
        }
        all_loras.append(lora_info)
        
        # Group by category
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(lora_info)
        
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
    
    # Sort each category
    for cat in by_category:
        by_category[cat].sort(key=lambda x: x["name"].lower())
    
    return {
        "loras": all_loras,
        "high_noise": high_noise,
        "low_noise": low_noise,
        "general": general,
        "by_category": by_category,
        "count": len(all_loras)
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
            "size_gb": round(model_path.stat().st_size / (1024 * 1024 * 1024), 2)
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
        h_base = h["name"].lower().replace("high", "").replace("_h_", "_").replace("-h-", "-")
        for l in low_noise:
            l_base = l["name"].lower().replace("low", "").replace("_l_", "_").replace("-l-", "-")
            # Check similarity
            if h_base == l_base or h_base.replace("noise", "") == l_base.replace("noise", ""):
                pairs.append({
                    "name": h["name"].replace("high", "").replace("High", "").replace("_H_", "_").replace("HIGH", "").strip("_- ") or h["name"],
                    "high": h,
                    "low": l
                })
                break
    
    return {
        "models": all_models,
        "high_noise": high_noise,
        "low_noise": low_noise,
        "pairs": pairs,
        "count": len(all_models)
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
        raise HTTPException(status_code=502, detail=f"ComfyUI connection failed: {str(e)}")


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
        history_resp = requests.get(f"http://localhost:8188/history/{prompt_id}", timeout=5)
        if history_resp.status_code == 200:
            history = history_resp.json()
            if prompt_id in history:
                job_data = history[prompt_id]
                outputs = job_data.get("outputs", {})
                
                # Find video output
                output_video = None
                for node_id, node_output in outputs.items():
                    if "gifs" in node_output:
                        for gif in node_output["gifs"]:
                            if gif.get("type") == "output":
                                output_video = f"/comfyui/output/{gif['filename']}"
                                break
                
                return {
                    "prompt_id": prompt_id,
                    "status": "completed",
                    "output_video": output_video,
                    **job_info
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
                    return {
                        "prompt_id": prompt_id,
                        "status": "running",
                        **job_info
                    }
            
            # Check pending
            for idx, item in enumerate(queue_data.get("queue_pending", [])):
                if len(item) >= 2 and item[1] == prompt_id:
                    return {
                        "prompt_id": prompt_id,
                        "status": "pending",
                        "queue_position": idx + 1,
                        **job_info
                    }
    except Exception as e:
        logger.warning(f"Error checking queue for {prompt_id}: {e}")
    
    # Not found anywhere - might have failed or been cancelled
    return {
        "prompt_id": prompt_id,
        "status": "unknown",
        **job_info
    }


@app.get("/comfyui/output/{filename}")
async def get_comfyui_output(filename: str):
    """Serve ComfyUI output files (videos/images)"""
    output_path = Path("/home/flip/oelala/ComfyUI/output") / filename
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    return FileResponse(output_path)


@app.delete("/comfyui/queue/{prompt_id}")
async def cancel_job(prompt_id: str):
    """Cancel a queued or running job"""
    import requests
    
    try:
        # ComfyUI interrupt endpoint
        resp = requests.post(
            "http://localhost:8188/interrupt",
            json={"prompt_id": prompt_id},
            timeout=5
        )
        
        # Also try to delete from queue
        delete_resp = requests.post(
            "http://localhost:8188/queue",
            json={"delete": [prompt_id]},
            timeout=5
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
    suffix = Path(file.filename).suffix if file.filename else '.png'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name
    
    metadata = {}
    try:
        # Try to read PNG metadata
        img = Image.open(tmp_path)
        
        # Check for various metadata formats
        if hasattr(img, 'info'):
            info = img.info
            
            # Oelala params (our format)
            if 'oelala_params' in info:
                try:
                    params = json.loads(info['oelala_params'])
                    metadata['prompt'] = params.get('prompt', '')
                    metadata['negative_prompt'] = params.get('negative_prompt', '')
                    metadata['workflow'] = params.get('workflow', '')
                    metadata['resolution'] = params.get('resolution', '')
                    metadata['steps'] = params.get('steps', '')
                    metadata['cfg'] = params.get('cfg', '')
                    metadata['seed'] = params.get('seed', '')
                    metadata['source'] = 'oelala'
                    # Check for preserved original T2I prompt (longer, more descriptive)
                    if params.get('original_t2i_prompt'):
                        metadata['prompt'] = params['original_t2i_prompt']
                        metadata['source'] = 'oelala_t2i'
                    # Store oelala prompt separately so we can compare later
                    metadata['oelala_prompt'] = params.get('prompt', '')
                except json.JSONDecodeError:
                    pass
            
            # ComfyUI workflow format - extract longer prompts from workflow nodes
            if 'prompt' in info:
                try:
                    workflow = json.loads(info['prompt'])
                    # Extract prompt from various node types
                    for node_id, node in workflow.items():
                        if isinstance(node, dict):
                            inputs = node.get('inputs', {})
                            class_type = node.get('class_type', '')
                            
                            # WanVideo text encoder (our I2V workflow) - skip short motion prompts
                            if 'positive_prompt' in inputs and isinstance(inputs['positive_prompt'], str):
                                if len(inputs['positive_prompt']) > 50 and not metadata.get('prompt'):
                                    metadata['prompt'] = inputs['positive_prompt']
                                    metadata['source'] = 'comfyui_wan'
                            if 'negative_prompt' in inputs and isinstance(inputs['negative_prompt'], str):
                                if len(inputs['negative_prompt']) > 10 and not metadata.get('negative_prompt'):
                                    metadata['negative_prompt'] = inputs['negative_prompt']
                            
                            # CLIPTextEncode (standard ComfyUI T2I) - prefer longer prompts
                            if 'text' in inputs and isinstance(inputs['text'], str):
                                text = inputs['text']
                                if len(text) > 20:
                                    # Check if it's a positive or negative prompt
                                    if 'negative' in class_type.lower():
                                        if not metadata.get('negative_prompt') or len(text) > len(metadata.get('negative_prompt', '')):
                                            metadata['negative_prompt'] = text
                                    else:
                                        # Prefer longer prompts (T2I prompts are usually longer than I2V motion prompts)
                                        current = metadata.get('prompt', '')
                                        if len(text) > len(current):
                                            metadata['prompt'] = text
                                            metadata['source'] = 'comfyui'
                except json.JSONDecodeError:
                    pass
            
            # A1111/Invoke AI format (parameters in 'parameters' key)
            if 'parameters' in info and not metadata.get('prompt'):
                params_text = info['parameters']
                # Format: "prompt text\nNegative prompt: negative text\nSteps: X, ..."
                lines = params_text.split('\n')
                if lines:
                    # First line(s) until "Negative prompt:" is the positive prompt
                    positive_lines = []
                    negative_started = False
                    for line in lines:
                        if line.startswith('Negative prompt:'):
                            negative_started = True
                            neg = line.replace('Negative prompt:', '').strip()
                            if neg:
                                metadata['negative_prompt'] = neg
                        elif line.startswith('Steps:'):
                            # Parse generation params
                            parts = line.split(',')
                            for part in parts:
                                if ':' in part:
                                    k, v = part.split(':', 1)
                                    k = k.strip().lower().replace(' ', '_')
                                    v = v.strip()
                                    if k in ['steps', 'cfg', 'seed', 'sampler']:
                                        metadata[k] = v
                        elif not negative_started:
                            positive_lines.append(line)
                    
                    if positive_lines:
                        metadata['prompt'] = '\n'.join(positive_lines).strip()
                    metadata['source'] = 'a1111'
        
        logger.info(f"📋 Extracted metadata from {file.filename}: {list(metadata.keys())}")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to extract metadata from {file.filename}: {e}")
    finally:
        # Cleanup temp file
        try:
            Path(tmp_path).unlink()
        except:
            pass
    
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
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        
        # Extract metadata (same logic as file upload)
        img = Image.open(tmp_path)
        
        if hasattr(img, 'info'):
            info = img.info
            
            # Oelala params (our format)
            if 'oelala_params' in info:
                try:
                    params = json.loads(info['oelala_params'])
                    metadata['positive_prompt'] = params.get('prompt', '')
                    metadata['negative_prompt'] = params.get('negative_prompt', '')
                    metadata['workflow'] = params.get('workflow', '')
                    metadata['source'] = 'oelala'
                    # Check for preserved original T2I prompt
                    if params.get('original_t2i_prompt'):
                        metadata['positive_prompt'] = params['original_t2i_prompt']
                        metadata['source'] = 'oelala_t2i'
                except json.JSONDecodeError:
                    pass
            
            # ComfyUI workflow format
            if 'prompt' in info and not metadata.get('positive_prompt'):
                try:
                    workflow = json.loads(info['prompt'])
                    for node_id, node in workflow.items():
                        if isinstance(node, dict):
                            inputs = node.get('inputs', {})
                            class_type = node.get('class_type', '')
                            
                            # Look for prompt inputs in various node types
                            # Wan nodes use positive_prompt/negative_prompt
                            if 'positive_prompt' in inputs and isinstance(inputs['positive_prompt'], str):
                                text = inputs['positive_prompt'].strip()
                                if text and len(text) > 5:
                                    current = metadata.get('positive_prompt', '')
                                    if len(text) > len(current):
                                        metadata['positive_prompt'] = text
                                        metadata['source'] = 'comfyui_wan'
                            
                            if 'negative_prompt' in inputs and isinstance(inputs['negative_prompt'], str):
                                text = inputs['negative_prompt'].strip()
                                if text and len(text) > 3:
                                    metadata['negative_prompt'] = text
                            
                            # CLIP/text nodes use 'text' key
                            if 'text' in inputs and isinstance(inputs['text'], str):
                                text = inputs['text'].strip()
                                if text and len(text) > 10:
                                    if 'negative' in class_type.lower():
                                        if not metadata.get('negative_prompt') or len(text) > len(metadata.get('negative_prompt', '')):
                                            metadata['negative_prompt'] = text
                                    else:
                                        current = metadata.get('positive_prompt', '')
                                        if len(text) > len(current):
                                            metadata['positive_prompt'] = text
                                            metadata['source'] = 'comfyui'
                except json.JSONDecodeError:
                    pass
            
            # A1111 format
            if 'parameters' in info and not metadata.get('positive_prompt'):
                params_text = info['parameters']
                lines = params_text.split('\n')
                positive_lines = []
                for line in lines:
                    if line.startswith('Negative prompt:'):
                        metadata['negative_prompt'] = line.replace('Negative prompt:', '').strip()
                        break
                    elif not line.startswith('Steps:'):
                        positive_lines.append(line)
                if positive_lines:
                    metadata['positive_prompt'] = '\n'.join(positive_lines).strip()
                    metadata['source'] = 'a1111'
        
        logger.info(f"📋 Extracted metadata from URL: {list(metadata.keys())}")
        
    except Exception as e:
        logger.warning(f"⚠️ Failed to extract metadata from URL: {e}")
        metadata['error'] = str(e)
    finally:
        if tmp_path:
            try:
                Path(tmp_path).unlink()
            except:
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
        except:
            pass
    
    # We're healthy if ComfyUI is available OR legacy generators are loaded
    is_healthy = comfyui_available or generator is not None or image_generator is not None
    
    return {
        "status": "healthy" if is_healthy else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "comfyui_available": comfyui_available,
        "model_loaded": generator is not None,
        "image_model_loaded": image_generator is not None,
        "upload_dir": str(UPLOAD_DIR),
        "output_dir": str(OUTPUT_DIR)
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
                "parameters": workflow.get("parameters", {})
            }
            presets.append(preset)
        
        # Sort by category, then name
        presets.sort(key=lambda p: (p["category"], p["name"]))
        
        return {
            "presets": presets,
            "total": len(presets),
            "categories": list(set(p["category"] for p in presets))
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
            raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found")
        
        return {
            "id": preset_id,
            "name": workflow.get("name", preset_id),
            "file": workflow.get("file"),
            "category": workflow.get("category", "Unknown"),
            "description": workflow.get("description", ""),
            "parameters": workflow.get("parameters", {})
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
    negative_prompt: str = Form("ugly, deformed, blurry, low quality, bad anatomy, watermark, signature, text"),
    checkpoint: str = Form("CyberRealistic_Pony_v14.1_FP16.safetensors"),
    aspect_ratio: str = Form("1:1"),
    steps: int = Form(30),
    cfg: float = Form(7.5),
    seed: int = Form(-1),
    sampler_name: str = Form("dpmpp_2m"),
    scheduler: str = Form("karras"),
    lora_configs: str = Form("[]"),  # JSON string of [{name, strength}]
):
    """
    Generate image using SDXL checkpoint via ComfyUI.
    Supports all SDXL checkpoints including Pony, Illustrious, JuggernautXL, etc.
    """
    from .comfyui_client import get_comfyui_client
    import json as json_lib
    
    logger.info(f"🎨 SDXL T2I request: {prompt[:50]}... (checkpoint={checkpoint})")
    
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
    
    try:
        client = get_comfyui_client()
        
        output_path = client.generate_sdxl_image(
            prompt=prompt,
            output_dir=str(OUTPUT_DIR),
            negative_prompt=negative_prompt,
            checkpoint=checkpoint,
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
            raise HTTPException(status_code=500, detail="SDXL generation failed")
        
        # Get just the filename
        filename = Path(output_path).name
        
        return {
            "status": "success",
            "url": f"/files/{filename}",
            "filename": filename,
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
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"SDXL generation failed: {e}")
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
):
    """
    Generate image using Flux Dev via ComfyUI.
    Flux doesn't use negative prompts - uses guidance instead.
    """
    from .comfyui_client import get_comfyui_client
    import json as json_lib
    
    logger.info(f"⚡ Flux T2I request: {prompt[:50]}...")
    
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
        
        filename = Path(output_path).name
        
        return {
            "status": "success",
            "url": f"/files/{filename}",
            "filename": filename,
            "meta": {
                "prompt": prompt,
                "model": "flux1-dev-fp8",
                "width": width,
                "height": height,
                "steps": steps,
                "guidance": guidance,
                "seed": seed,
            }
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
    negative_prompt: str = Form("(deformed, blurry, bad anatomy, extra fingers, mutated hands, poorly drawn face, low quality:1.4)"),
    aspect_ratio: str = Form("2:3"),
    steps: int = Form(25),
    cfg: float = Form(7.0),
    seed: int = Form(-1),
    sampler_name: str = Form("dpmpp_sde"),
    scheduler: str = Form("karras"),
    lora_configs: str = Form("[]"),  # JSON string of [{name, strength}]
):
    """
    Generate image using SD 1.5 (Realistic Vision V5.1) via ComfyUI.
    """
    from .comfyui_client import get_comfyui_client
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
        
        return {
            "status": "success",
            "url": f"/files/{filename}",
            "filename": filename,
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
            }
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
):
    """
    Generate image using Wan2.2 T2V model in T2I mode via ComfyUI.
    Uses DisTorch2 multi-GPU setup with high/low noise models.
    Very high quality but slower than other T2I models.
    """
    from .comfyui_client import get_comfyui_client
    
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
        
        return {
            "status": "success",
            "url": f"/files/{filename}",
            "filename": filename,
            "meta": {
                "prompt": prompt,
                "model": "wan2.2-t2i-distorch2",
                "width": width,
                "height": height,
                "steps": steps,
                "seed": seed,
            }
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
    unet_high_noise: str = Form("wan2.2_i2v_high_noise_14B_Q6_K.gguf", description="GGUF model for high noise pass"),
    unet_low_noise: str = Form("wan2.2_i2v_low_noise_14B_Q6_K.gguf", description="GGUF model for low noise pass"),
    lora_configs: str = Form("", description="JSON array of LoRA configs [{high, low, strength}, ...]"),
    extend_mode: str = Form("false", description="Enable sequential clip extension"),
    clip_count: int = Form(1, description="Number of sequential clips (1-5)")
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
            logger.info(f"🎬 Starting Sequential Wan2.2 generation ({actual_clip_count} clips)")
            logger.info(f"   📐 Resolution: {resolution}, Aspect: {aspect_ratio}")
            logger.info(f"   🎞️ Frames per clip: {num_frames}, Total: {total_frames}, FPS: {fps}")
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
                    lora_configs=parsed_lora_configs
                )
            )
        else:
            # Standard single-clip generation
            logger.info(f"🎬 Starting Wan2.2 ComfyUI generation")
            logger.info(f"   📐 Resolution: {resolution}, Aspect: {aspect_ratio}")
            logger.info(f"   🎞️ Frames: {num_frames}, FPS: {fps}")
            logger.info(f"   ⚙️ Steps: {steps}, CFG: {cfg}, Seed: {seed}")
            logger.info(f"   🔧 Unet: H={unet_high_noise}, L={unet_low_noise}")
            if parsed_lora_configs:
                logger.info(f"   🎨 LoRAs: {len(parsed_lora_configs)} configured")
                for i, lc in enumerate(parsed_lora_configs):
                    logger.info(f"      [{i+1}] H={lc.get('high') or 'none'}, L={lc.get('low') or 'none'} @ {lc.get('strength', 1.0)}")
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
                    lora_configs=parsed_lora_configs
                )
            )
        
        if result_path and Path(result_path).exists():
            # Copy to expected output path if different
            final_output = OUTPUT_DIR / output_filename
            if str(result_path) != str(final_output):
                shutil.copy(result_path, final_output)
                result_path = str(final_output)
            
            total_frames = num_frames * actual_clip_count if is_extend_mode else num_frames
            
            return {
                "success": True,
                "message": f"Wan2.2 video generated via ComfyUI{' (sequential)' if actual_clip_count > 1 else ''}",
                "input_image": input_filename,
                "output_video": output_filename,
                "video_url": f"/files/{output_filename}",
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
                "model": "wan2.2_i2v_14B_Q6"
            }
        else:
            raise HTTPException(status_code=500, detail="ComfyUI video generation returned no output")
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ ComfyUI generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Wan2.2 ComfyUI generation failed: {str(e)}")


@app.post("/generate-wan22-async")
async def generate_wan22_async(
    file: UploadFile = File(...),
    prompt: str = Form("Motion, subject moving naturally"),
    num_frames: int = Form(41, description="Number of frames in video"),
    output_filename: str = Form("", description="Custom output filename"),
    resolution: str = Form("480p", description="Video resolution: 480p, 576p, 720p, 1080p"),
    fps: int = Form(16, description="Frames per second: 8, 12, 16, 24"),
    aspect_ratio: str = Form("1:1", description="Video aspect ratio"),
    steps: int = Form(6, description="Sampling steps"),
    cfg: float = Form(1.0, description="CFG guidance scale (1.0 for DisTorch2)"),
    seed: int = Form(-1, description="Random seed (-1 for random)"),
    unet_high_noise: str = Form("wan2.2_i2v_high_noise_14B_Q6_K.gguf", description="GGUF model for high noise pass"),
    unet_low_noise: str = Form("wan2.2_i2v_low_noise_14B_Q6_K.gguf", description="GGUF model for low noise pass"),
    lora_configs: str = Form("", description="JSON array of LoRA configs [{high, low, strength}, ...]"),
    extend_mode: str = Form("false", description="Enable sequential clip extension"),
    clip_count: int = Form(1, description="Number of sequential clips (1-5)")
):
    """
    Queue Wan2.2 I2V video generation and return immediately.
    
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
            detail="ComfyUI not running. Start with: cd ~/oelala/ComfyUI && python main.py --listen"
        )
    
    # Wan2.2 requires num_frames in format 4k+1
    k = round((num_frames - 1) / 4)
    k = max(1, k)
    num_frames = 4 * k + 1
    
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
    actual_seed = seed if seed >= 0 else int(datetime.now().timestamp() * 1000) % 2147483647
    
    # Map resolution to long_edge
    resolution_map = {"480p": 480, "576p": 576, "720p": 720, "1080p": 1080}
    long_edge = resolution_map.get(resolution, 480)
    
    # Check if sequential/extend mode is enabled
    is_extend_mode = extend_mode.lower() in ("true", "1", "yes")
    actual_clip_count = max(1, min(5, clip_count)) if is_extend_mode else 1
    
    # Build workflow
    if is_extend_mode and actual_clip_count > 1:
        # Build sequential workflow for multiple clips
        logger.info(f"🎬 Building sequential workflow: {actual_clip_count} clips × {num_frames} frames")
        width, height = comfyui.get_resolution_dimensions(resolution, aspect_ratio)
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
        raise HTTPException(status_code=500, detail="Failed to queue workflow to ComfyUI")
    
    # Store job info for tracking
    total_frames = num_frames * actual_clip_count if is_extend_mode else num_frames
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
        logger.info(f"🚀 Queued sequential job: {prompt_id} ({actual_clip_count} clips)")
    else:
        logger.info(f"🚀 Queued async job: {prompt_id}")
    logger.info(f"   📐 {resolution} {aspect_ratio}, {num_frames}f @ {fps}fps")
    logger.info(f"   📝 {prompt[:50]}...")
    
    return {
        "success": True,
        "prompt_id": prompt_id,
        "status": "queued",
        "message": "Job queued successfully. Poll /comfyui/job/{prompt_id} for status.",
        **job_info
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
        raise HTTPException(
            status_code=503, 
            detail="Text-to-Video is not yet available. Please use Image-to-Video instead (ComfyUI-based)."
        )

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
