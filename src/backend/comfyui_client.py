#!/usr/bin/env python3
"""
ComfyUI API Client for Oelala Backend
Enables integration with ComfyUI for video generation workflows (MiniMax-H3, LTX-2.3)
"""

import json
import os
import uuid
import time
import requests
import websocket
import random
import threading
import math
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime
import logging
import io
import copy

# Import for auto-upload functionality (legacy sync client)
from storage_client import get_client as get_storage_client

# Guardian LLM proxy — VRAM management
from guardian_client import get_guardian

# Import MediaService for async uploads with Supabase sync
try:
    from media_service import MediaService

    _media_service: Optional[MediaService] = None

    def get_media_service() -> MediaService:
        """Get or create the global MediaService instance."""
        global _media_service
        if _media_service is None:
            _media_service = MediaService()
        return _media_service
except ImportError:
    get_media_service = None  # type: ignore

logger = logging.getLogger(__name__)

# MiniMax-H3 turbo presets (ModelTC/Minimax-H3-Turbo, Comfy-Org mirrored files).
# lora + steps + sigma shift must be set together to match the model table:
#  - 4-step v1.0 768p is trained at 1344x768 with sigma shift 6/3 (the base
#    checkpoint bakes 12/3), so it needs the MiniMaxH3SigmaShift node override.
#  - 8-step v1.0 (544p-trained) uses the base shift 12/3 — no override node.
# Turbo graphs use the euler sampler (ModelTC reference workflows).
MINIMAX_H3_TURBO_PRESETS = {
    "draft": {
        "lora": "minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
        "steps": 4,
        "shift_video": 6.0,
        "shift_audio": 3.0,
    },
    "standard": {
        "lora": "minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors",
        "steps": 8,
        "shift_video": None,
        "shift_audio": None,
    },
}

# MiniMax-H3 checkpoint variants: the official Comfy-Org release plus community
# finetunes mirrored from Civitai into the private HF model repo.
#
# Filenames are the upstream Civitai names — the trailing number is the Civitai
# file id — so a manual download and a worker download produce the same file.
#
#   h3ErosMax_beta5_3185144  turbo-hybrid fp8/w4a8 (14.0 GB, 180 layers 4-bit)
#   h3ErosMax_beta5_3178732  turbo int8        (21.0 GB, every layer 8-bit)
#   h3ErosMax_beta5_3185154  non-turbo int8    (21.0 GB, every layer 8-bit)
#   DasiwaMinimaxH3_..._3203135  DaSiWa Hybrid Turbo v2 int8
#   DasiwaMinimaxH3_..._3203130  DaSiWa Hybrid v2 int8 (non-turbo)
#
# "turbo" variants carry a baked distillation and are sampled with few steps;
# non-turbo variants need the caller's full step count.
MINIMAX_H3_VARIANTS = {
    "official": {
        "checkpoint": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        "turbo": False,
    },
    "eros": {
        "checkpoint": "h3ErosMax_beta5_3185144.safetensors",
        "turbo": True,
    },
    "eros_int8_turbo": {
        "checkpoint": "h3ErosMax_beta5_3178732.safetensors",
        "turbo": True,
    },
    "eros_int8": {
        "checkpoint": "h3ErosMax_beta5_3185154.safetensors",
        "turbo": False,
    },
    "dasiwa_turbo": {
        "checkpoint": "DasiwaMinimaxH3_dasiwaHybridTurboV2_3203135.safetensors",
        "turbo": True,
    },
    "dasiwa": {
        "checkpoint": "DasiwaMinimaxH3_dasiwaHybridV2_3203130.safetensors",
        "turbo": False,
    },
}

MINIMAX_H3_CHECKPOINTS = {
    name: spec["checkpoint"] for name, spec in MINIMAX_H3_VARIANTS.items()
}

# ─────────────────────────────────────────────────────────────────────────────
# Workflow Directory and Dynamic Loading
# ─────────────────────────────────────────────────────────────────────────────
WORKFLOWS_DIR = Path("/home/flip/oelala/workflows")

# Available I2V generation modes with their workflow files
# Wan2.2 I2V generation modes removed — model retired 2026-10-01 and
# their workflow JSONs deleted. Modes now come from the v2 adapter registry.
I2V_GENERATION_MODES: Dict[str, Dict] = {}

# Available T2V (Text-to-Video) generation modes - different base models
T2V_GENERATION_MODES = {
    "ltx2": {
        "name": "LTX-2.3 22B",
        "description": "Lightricks LTX-2.3 22B distilled, fast 8-step generation",
        "workflow_file": None,  # Uses cloud workflow builder
        "model_type": "ltx2",
        "default_steps": 8,
        "default_cfg": 1.0,
        "max_frames": 481,
        "default_frames": 97,
    },
}

def load_workflow_from_file(workflow_path: str) -> Optional[Dict]:
    """Load a workflow JSON file and return as dict."""
    full_path = WORKFLOWS_DIR / workflow_path
    if not full_path.exists():
        logger.error(f"❌ Workflow file not found: {full_path}")
        return None
    try:
        with open(full_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"❌ Failed to load workflow {workflow_path}: {e}")
        return None

def get_available_i2v_modes() -> Dict:
    """Return available I2V generation modes."""
    return I2V_GENERATION_MODES

def get_available_t2v_modes() -> Dict:
    """Return available T2V generation modes (base models)."""
    return T2V_GENERATION_MODES

def _match_lora_name(requested: str, available: List[str]) -> Optional[str]:
    """Map a requested LoRA name onto a filename the target server actually has.

    Workflow JSON carries forward-slash paths (`minimax-h3/x.safetensors`) while
    Windows ComfyUI servers enumerate files with backslashes
    (`minimax-h3\\x.safetensors`), and prompt validation does an exact string
    match against that list. Returns the matching server-side name, a unique
    basename match, or None when the server does not have the LoRA.
    """
    if requested in available:
        return requested
    for candidate in (requested.replace("/", "\\"), requested.replace("\\", "/")):
        if candidate in available:
            return candidate
    basename = requested.replace("\\", "/").rsplit("/", 1)[-1]
    matches = [
        a for a in available if a.replace("\\", "/").rsplit("/", 1)[-1] == basename
    ]
    return matches[0] if len(matches) == 1 else None

class ComfyUIClient:
    """Client for ComfyUI API integration"""

    def __init__(self, host: str = "localhost", port: int = 8188):
        self.host = host
        self.port = port
        self.base_url = f"http://{host}:{port}"
        self.client_id = str(uuid.uuid4())
        # Job tracking: prompt_id -> {user_id, prompt, settings, started_at}
        self.job_metadata = {}
        # Thread-safe lock for job_metadata access
        self._metadata_lock = threading.Lock()

    def is_available(self, timeout: float = 5) -> bool:
        """Check if ComfyUI is running and accessible.

        *timeout* must cover lazy-start backends: a wake proxy (caretaker) may
        need 1-2 minutes to cold-start ComfyUI before answering /system_stats.
        Connection-refused fails fast regardless of the timeout.
        """
        try:
            resp = requests.get(f"{self.base_url}/system_stats", timeout=timeout)
            return resp.status_code == 200
        except Exception:
            return False

    def get_model_names(self, loader_type: str = "UnetLoaderGGUF") -> list[str]:
        """Fetch available models dynamically from ComfyUI"""
        try:
            import requests

            resp = requests.get(f"{self.base_url}/object_info/{loader_type}", timeout=5)
            data = resp.json()
            if loader_type in data:
                return data[loader_type]["input"]["required"].get("unet_name", [[]])[0]
        except Exception as e:
            logger.error(f"Error fetching {loader_type} models: {e}")
        return []

    def get_lora_names(self) -> List[str]:
        """Fetch the LoRA filenames exactly as this server enumerates them.

        The list is what prompt validation matches `lora_name` against — on
        Windows installs subfolder entries use backslash separators.
        """
        try:
            resp = requests.get(
                f"{self.base_url}/object_info/LoraLoaderModelOnly", timeout=5
            )
            data = resp.json()
            if "LoraLoaderModelOnly" in data:
                return data["LoraLoaderModelOnly"]["input"]["required"].get(
                    "lora_name", [[]]
                )[0]
        except Exception as e:
            logger.error(f"Error fetching LoRA list: {e}")
        return []

    def upload_image(self, image_path: str, subfolder: str = "") -> Optional[str]:
        """Upload image to ComfyUI input folder"""
        try:
            path = Path(image_path)
            if not path.exists():
                logger.error(f"Image not found: {image_path}")
                return None

            with open(path, "rb") as f:
                files = {"image": (path.name, f, "image/png")}
                data = {"subfolder": subfolder, "overwrite": "true"}
                resp = requests.post(
                    f"{self.base_url}/upload/image", files=files, data=data
                )

            if resp.status_code == 200:
                result = resp.json()
                logger.info(f"📤 Image uploaded: {result.get('name')}")
                return result.get("name")
            else:
                logger.error(f"Upload failed: {resp.status_code} - {resp.text}")
                return None
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return None

    def upload_image_from_bytes(
        self, image_bytes: bytes, filename: str = "input_image.png"
    ) -> Optional[str]:
        """Upload image from bytes to ComfyUI"""
        try:
            files = {"image": (filename, io.BytesIO(image_bytes), "image/png")}
            data = {"subfolder": "", "overwrite": "true"}
            resp = requests.post(
                f"{self.base_url}/upload/image", files=files, data=data
            )

            if resp.status_code == 200:
                result = resp.json()
                logger.info(f"📤 Image uploaded from bytes: {result.get('name')}")
                return result.get("name")
            else:
                logger.error(f"Upload failed: {resp.status_code}")
                return None
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return None

    def upload_video(self, video_path: str, subfolder: str = "") -> Optional[str]:
        """Upload video to ComfyUI input folder.

        ComfyUI's /upload/image endpoint accepts any file type,
        despite the 'image' field name.
        """
        try:
            path = Path(video_path)
            if not path.exists():
                logger.error(f"🎬 Video not found: {video_path}")
                return None

            # Determine content type based on extension
            ext = path.suffix.lower()
            content_types = {
                ".mp4": "video/mp4",
                ".webm": "video/webm",
                ".mov": "video/quicktime",
                ".avi": "video/x-msvideo",
                ".mkv": "video/x-matroska",
                ".gif": "image/gif",
            }
            content_type = content_types.get(ext, "application/octet-stream")

            with open(path, "rb") as f:
                # ComfyUI uses 'image' field name but accepts any file
                files = {"image": (path.name, f, content_type)}
                data = {"subfolder": subfolder, "overwrite": "true"}
                resp = requests.post(
                    f"{self.base_url}/upload/image", files=files, data=data
                )

            if resp.status_code == 200:
                result = resp.json()
                logger.info(f"🎬 Video uploaded: {result.get('name')}")
                return result.get("name")
            else:
                logger.error(
                    f"🎬 Video upload failed: {resp.status_code} - {resp.text}"
                )
                return None
        except Exception as e:
            logger.error(f"🎬 Video upload error: {e}")
            return None

    def upload_video_from_bytes(
        self, video_bytes: bytes, filename: str = "input_video.mp4"
    ) -> Optional[str]:
        """Upload video from bytes to ComfyUI"""
        try:
            # Determine content type based on extension
            ext = Path(filename).suffix.lower()
            content_types = {
                ".mp4": "video/mp4",
                ".webm": "video/webm",
                ".mov": "video/quicktime",
                ".avi": "video/x-msvideo",
                ".mkv": "video/x-matroska",
                ".gif": "image/gif",
            }
            content_type = content_types.get(ext, "video/mp4")

            files = {"image": (filename, io.BytesIO(video_bytes), content_type)}
            data = {"subfolder": "", "overwrite": "true"}
            resp = requests.post(
                f"{self.base_url}/upload/image", files=files, data=data
            )

            if resp.status_code == 200:
                result = resp.json()
                logger.info(f"🎬 Video uploaded from bytes: {result.get('name')}")
                return result.get("name")
            else:
                logger.error(f"🎬 Video upload failed: {resp.status_code}")
                return None
        except Exception as e:
            logger.error(f"🎬 Video upload error: {e}")
            return None

    def get_resolution_dimensions(
        self, resolution: str, aspect_ratio: str
    ) -> Tuple[int, int]:
        """Calculate width/height from resolution and aspect ratio"""
        # Base heights for each resolution
        base_heights = {"480p": 480, "576p": 576, "720p": 720, "1080p": 1080}

        # Aspect ratio multipliers
        aspect_ratios = {
            "16:9": (16, 9),
            "9:16": (9, 16),
            "1:1": (1, 1),
            "4:3": (4, 3),
            "3:4": (3, 4),
            "21:9": (21, 9),
            "auto": (1, 1),  # Default to square
        }

        height = base_heights.get(resolution, 480)
        ar_w, ar_h = aspect_ratios.get(aspect_ratio, (1, 1))

        # Calculate width based on aspect ratio
        if ar_w >= ar_h:
            # Landscape or square
            width = int(height * ar_w / ar_h)
        else:
            # Portrait - use width as base
            width = height
            height = int(width * ar_h / ar_w)

        # Ensure dimensions are multiples of 8 for VAE
        width = (width // 8) * 8
        height = (height // 8) * 8

        return width, height

    def build_cloud_ltx23_t2v_workflow(
        self,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted, artifacts, watermark",
        width: int = 768,
        height: int = 512,
        num_frames: int = 97,
        fps: int = 25,
        seed: int = -1,
        output_prefix: str = "oelala_ltx23_t2v",
        checkpoint: str = "ltx-2.3-22b-distilled.safetensors",
        text_encoder: str = "gemma_3_12B_it_fp8_scaled.safetensors",
        aspect_ratio: str = "9:16",
        long_edge: int = 768,
        lora_configs: Optional[list] = None,
        audio_prompt: Optional[str] = None,
        audio_vae_checkpoint: str = "ltx2_audio_vae.safetensors",
    ) -> Optional[Dict[str, Any]]:
        """
        Build LTX-2.3 22B Cloud T2V workflow — single-stage distilled pipeline.

        Uses the 8-step distilled sigma schedule. When audio_prompt is provided,
        generates audio-video using MultimodalGuider + audio VAE.
        """
        # LTX frame count must be 8k+1
        k = round((num_frames - 1) / 8)
        k = max(1, k)
        num_frames = 8 * k + 1

        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        # Calculate dimensions from aspect ratio
        aspect_ratios = {
            "1:1": (1, 1),
            "9:16": (9, 16),
            "16:9": (16, 9),
            "4:3": (4, 3),
            "3:4": (3, 4),
            "3:2": (3, 2),
            "2:3": (2, 3),
            "21:9": (21, 9),
            "9:21": (9, 21),
        }
        ar_w, ar_h = aspect_ratios.get(aspect_ratio, (9, 16))

        if ar_w >= ar_h:
            height = long_edge
            width = int(long_edge * ar_w / ar_h)
        else:
            width = long_edge
            height = int(long_edge * ar_h / ar_w)

        # LTX requires dimensions divisible by 32
        width = (width // 32) * 32
        height = (height // 32) * 32

        logger.info(
            f"☁️ Building LTX-2.3 Cloud T2V: {width}x{height}, "
            f"{num_frames}f@{fps}fps, seed={seed}"
        )

        workflow = {}

        # Node 1: CheckpointLoaderSimple — loads MODEL + CLIP + VAE
        workflow["1"] = {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        }

        # Node 2: LTXAVTextEncoderLoader — Gemma 3 12B fp8 text encoder
        workflow["2"] = {
            "class_type": "LTXAVTextEncoderLoader",
            "inputs": {
                "text_encoder": text_encoder,
                "ckpt_name": checkpoint,
                "device": "default",
            },
        }

        # Build effective prompt (append audio description if provided)
        effective_prompt = prompt
        if audio_prompt:
            effective_prompt = f"{prompt}\nAudio: {audio_prompt}"

        # Node 3: Positive prompt
        workflow["3"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": effective_prompt, "clip": ["2", 0]},
        }

        # Node 4: Negative prompt
        workflow["4"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["2", 0]},
        }

        # Node 5: LTXVConditioning — set frame rate on conditioning
        workflow["5"] = {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["3", 0],
                "negative": ["4", 0],
                "frame_rate": float(fps),
            },
        }

        # Node 6: EmptyLTXVLatentVideo — create latent
        workflow["6"] = {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1,
            },
        }

        # Node 7: CFGGuider — cfg=1 for distilled model
        workflow["7"] = {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["1", 0],
                "positive": ["5", 0],
                "negative": ["5", 1],
                "cfg": 1.0,
            },
        }

        # Node 8: KSamplerSelect
        workflow["8"] = {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler_ancestral"},
        }

        # Node 9: ManualSigmas — 8-step distilled schedule
        workflow["9"] = {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
            },
        }

        # Node 10: RandomNoise
        workflow["10"] = {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        }

        # ── Audio-Video Pipeline (conditional) ──────────────────────
        sampler_latent_input = ["6", 0]  # default: empty video latent
        sampler_guider_input = ["7", 0]  # default: CFGGuider

        if audio_prompt:
            # Node 20: LTXVAudioVAELoader — load audio VAE
            workflow["20"] = {
                "class_type": "LTXVAudioVAELoader",
                "inputs": {"ckpt_name": audio_vae_checkpoint},
            }

            # Node 21: LTXVEmptyLatentAudio — empty audio latent matching video frames
            workflow["21"] = {
                "class_type": "LTXVEmptyLatentAudio",
                "inputs": {
                    "frames_number": num_frames,
                    "frame_rate": fps,
                    "batch_size": 1,
                    "audio_vae": ["20", 0],
                },
            }

            # Node 22: LTXVConcatAVLatent — combine video + audio latents
            workflow["22"] = {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {
                    "video_latent": ["6", 0],  # empty video latent
                    "audio_latent": ["21", 0],  # empty audio latent
                },
            }

            # Node 23: GuiderParameters — VIDEO modality (cfg=1.0 for distilled)
            workflow["23"] = {
                "class_type": "GuiderParameters",
                "inputs": {
                    "modality": "VIDEO",
                    "cfg": 1.0,
                    "stg": 0.0,
                    "perturb_attn": True,
                    "rescale": 0.0,
                    "modality_scale": 1.0,
                    "skip_step": 0,
                    "cross_attn": True,
                },
            }

            # Node 24: GuiderParameters — AUDIO modality (chained from VIDEO)
            workflow["24"] = {
                "class_type": "GuiderParameters",
                "inputs": {
                    "modality": "AUDIO",
                    "cfg": 1.0,
                    "stg": 0.0,
                    "perturb_attn": True,
                    "rescale": 0.0,
                    "modality_scale": 1.0,
                    "skip_step": 0,
                    "cross_attn": True,
                    "parameters": ["23", 0],  # chain from VIDEO params
                },
            }

            # Node 25: MultimodalGuider — replaces CFGGuider for AV generation
            workflow["25"] = {
                "class_type": "MultimodalGuider",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["5", 0],
                    "negative": ["5", 1],
                    "parameters": ["24", 0],
                    "skip_blocks": "",
                },
            }

            sampler_latent_input = ["22", 0]  # combined AV latent
            sampler_guider_input = ["25", 0]  # MultimodalGuider

            logger.info("🔊 T2V Audio pipeline enabled — MultimodalGuider + AV latents")

        # Node 11: SamplerCustomAdvanced — main sampling
        workflow["11"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["10", 0],
                "guider": sampler_guider_input,
                "sampler": ["8", 0],
                "sigmas": ["9", 0],
                "latent_image": sampler_latent_input,
            },
        }

        # ── Post-sampling: video decode (always) + audio decode (if AV) ──
        if audio_prompt:
            # Node 26: LTXVSeparateAVLatent — split denoised AV latent
            workflow["26"] = {
                "class_type": "LTXVSeparateAVLatent",
                "inputs": {
                    "av_latent": ["11", 1],  # denoised_output from sampler
                },
            }

            # Node 12: VAEDecodeTiled — decode VIDEO latent
            workflow["12"] = {
                "class_type": "VAEDecodeTiled",
                "inputs": {
                    "samples": ["26", 0],  # video_latent from separator
                    "vae": ["1", 2],
                    "tile_size": 512,
                    "overlap": 64,
                    "temporal_size": 512,
                    "temporal_overlap": 64,
                },
            }

            # Node 27: LTXVAudioVAEDecode — decode AUDIO latent
            workflow["27"] = {
                "class_type": "LTXVAudioVAEDecode",
                "inputs": {
                    "samples": ["26", 1],  # audio_latent from separator
                    "audio_vae": ["20", 0],  # audio VAE from loader
                },
            }

            # Node 13: VHS_VideoCombine — save video WITH audio
            workflow["13"] = {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "frame_rate": fps,
                    "loop_count": 0,
                    "filename_prefix": output_prefix,
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 17,
                    "save_metadata": True,
                    "trim_to_audio": True,
                    "pingpong": False,
                    "save_output": True,
                    "images": ["12", 0],
                    "audio": ["27", 0],  # decoded audio waveform
                },
            }
        else:
            # Node 12: VAEDecodeTiled — decode video-only latent
            workflow["12"] = {
                "class_type": "VAEDecodeTiled",
                "inputs": {
                    "samples": ["11", 1],  # denoised_output
                    "vae": ["1", 2],
                    "tile_size": 512,
                    "overlap": 64,
                    "temporal_size": 512,
                    "temporal_overlap": 64,
                },
            }

            # Node 13: VHS_VideoCombine — save video (no audio)
            workflow["13"] = {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "frame_rate": fps,
                    "loop_count": 0,
                    "filename_prefix": output_prefix,
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 17,
                    "save_metadata": True,
                    "trim_to_audio": False,
                    "pingpong": False,
                    "save_output": True,
                    "images": ["12", 0],
                },
            }

        # ── LoRA chain (single-stage: insert between checkpoint and CFGGuider) ──
        if lora_configs:
            current_model = ["1", 0]  # CheckpointLoaderSimple MODEL output
            lora_node_id = 100
            for i, config in enumerate(lora_configs):
                lora_name = config.get("high") or config.get("name", "")
                if not lora_name:
                    continue
                strength = config.get("strength", 1.0)
                node_id = str(lora_node_id + i)
                workflow[node_id] = {
                    "class_type": "LoraLoaderModelOnly",
                    "inputs": {
                        "lora_name": lora_name,
                        "strength_model": strength,
                        "model": current_model,
                    },
                }
                current_model = [node_id, 0]
                logger.info(f"🎨 LTX-2.3 T2V LoRA #{i + 1}: {lora_name} @ {strength}")
            # Wire final LoRA output to guider (CFGGuider or MultimodalGuider)
            guider_node = "25" if audio_prompt else "7"
            workflow[guider_node]["inputs"]["model"] = current_model

        lora_info = ""
        if lora_configs and len(lora_configs) > 0:
            lora_info = (
                f", {len(lora_configs)} LoRA{'s' if len(lora_configs) > 1 else ''}"
            )
        audio_info = " + audio" if audio_prompt else ""
        logger.info(
            f"☁️ Built LTX-2.3 Cloud T2V: {width}x{height}, "
            f"{num_frames}f@{fps}fps, 8-step distilled{lora_info}{audio_info}"
        )
        return workflow

    def build_cloud_ltx23_i2v_workflow(
        self,
        image_name: str,
        prompt: str,
        negative_prompt: str = "low quality, blurry, distorted, artifacts, watermark",
        width: int = 768,
        height: int = 512,
        num_frames: int = 97,
        fps: int = 25,
        seed: int = -1,
        strength: float = 1.0,
        output_prefix: str = "oelala_ltx23_i2v",
        checkpoint: str = "ltx-2.3-22b-distilled.safetensors",
        text_encoder: str = "gemma_3_12B_it_fp8_scaled.safetensors",
        lora_configs: Optional[list] = None,
        audio_prompt: Optional[str] = None,
        audio_vae_checkpoint: str = "ltx2_audio_vae.safetensors",
    ) -> Optional[Dict[str, Any]]:
        """
        Build LTX-2.3 22B Cloud I2V workflow — image-to-video with
        LTXVImgToVideoConditionOnly from ComfyUI-LTXVideo.

        Uses 8-step distilled sigma schedule. Image is preprocessed
        with LTXVPreprocess and used to condition the latent.

        When audio_prompt is provided, enables audio-video generation:
        appends audio description to prompt, adds audio VAE, empty audio
        latent, MultimodalGuider, and audio decode/mux nodes.
        """
        # LTX frame count must be 8k+1
        k = round((num_frames - 1) / 8)
        k = max(1, k)
        num_frames = 8 * k + 1

        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        # LTX requires dimensions divisible by 32
        width = (width // 32) * 32
        height = (height // 32) * 32

        logger.info(
            f"☁️ Building LTX-2.3 Cloud I2V: {width}x{height}, "
            f"{num_frames}f@{fps}fps, strength={strength}, seed={seed}"
        )

        workflow = {}

        # Node 1: CheckpointLoaderSimple — MODEL + CLIP + VAE
        workflow["1"] = {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint},
        }

        # Node 2: LTXAVTextEncoderLoader — Gemma 3 12B fp8
        workflow["2"] = {
            "class_type": "LTXAVTextEncoderLoader",
            "inputs": {
                "text_encoder": text_encoder,
                "ckpt_name": checkpoint,
                "device": "default",
            },
        }

        # Node 3: LoadImage — input image
        workflow["3"] = {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        }

        # Node 4: LTXVPreprocess — compress/noise condition the image
        workflow["4"] = {
            "class_type": "LTXVPreprocess",
            "inputs": {
                "image": ["3", 0],
                "img_compression": 35,
            },
        }

        # Combine audio description into the main prompt when audio is enabled
        # LTX-2.3 AV model uses the same text encoder for both modalities
        effective_prompt = prompt
        if audio_prompt:
            effective_prompt = f"{prompt}\nAudio: {audio_prompt}"
            logger.info(f"🔊 Audio prompt appended: {audio_prompt[:80]}...")

        # Node 5: Positive prompt
        workflow["5"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": effective_prompt, "clip": ["2", 0]},
        }

        # Node 6: Negative prompt
        workflow["6"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["2", 0]},
        }

        # Node 7: LTXVConditioning — set frame rate
        workflow["7"] = {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["5", 0],
                "negative": ["6", 0],
                "frame_rate": float(fps),
            },
        }

        # Node 8: EmptyLTXVLatentVideo — create empty latent
        workflow["8"] = {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1,
            },
        }

        # Node 9: LTXVImgToVideoConditionOnly — condition latent on image
        workflow["9"] = {
            "class_type": "LTXVImgToVideoConditionOnly",
            "inputs": {
                "vae": ["1", 2],  # VAE from checkpoint
                "image": ["4", 0],  # preprocessed image
                "latent": ["8", 0],  # empty latent
                "strength": strength,
            },
        }

        # Node 10: CFGGuider — cfg=1 for distilled
        workflow["10"] = {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["1", 0],
                "positive": ["7", 0],
                "negative": ["7", 1],
                "cfg": 1.0,
            },
        }

        # Node 11: KSamplerSelect
        workflow["11"] = {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler_ancestral"},
        }

        # Node 12: ManualSigmas — 8-step distilled schedule
        workflow["12"] = {
            "class_type": "ManualSigmas",
            "inputs": {
                "sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0",
            },
        }

        # Node 13: RandomNoise
        workflow["13"] = {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        }

        # ── Audio-Video Pipeline (conditional) ──────────────────────
        # When audio_prompt is provided, we enable the full AV pipeline:
        # - Load audio VAE, create empty audio latents
        # - Concat video + audio latents
        # - Use MultimodalGuider instead of CFGGuider
        # - After sampling: separate, decode audio, mux into video
        sampler_latent_input = ["9", 0]  # default: image-conditioned video latent
        sampler_guider_input = ["10", 0]  # default: CFGGuider

        if audio_prompt:
            # Node 20: LTXVAudioVAELoader — load audio VAE
            workflow["20"] = {
                "class_type": "LTXVAudioVAELoader",
                "inputs": {"ckpt_name": audio_vae_checkpoint},
            }

            # Node 21: LTXVEmptyLatentAudio — empty audio latent matching video frames
            workflow["21"] = {
                "class_type": "LTXVEmptyLatentAudio",
                "inputs": {
                    "frames_number": num_frames,
                    "frame_rate": fps,
                    "batch_size": 1,
                    "audio_vae": ["20", 0],
                },
            }

            # Node 22: LTXVConcatAVLatent — combine video + audio latents
            workflow["22"] = {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {
                    "video_latent": ["9", 0],  # image-conditioned video latent
                    "audio_latent": ["21", 0],  # empty audio latent
                },
            }

            # Node 23: GuiderParameters — VIDEO modality (cfg=1.0 for distilled)
            workflow["23"] = {
                "class_type": "GuiderParameters",
                "inputs": {
                    "modality": "VIDEO",
                    "cfg": 1.0,
                    "stg": 0.0,
                    "perturb_attn": True,
                    "rescale": 0.0,
                    "modality_scale": 1.0,
                    "skip_step": 0,
                    "cross_attn": True,
                },
            }

            # Node 24: GuiderParameters — AUDIO modality (chained from VIDEO)
            workflow["24"] = {
                "class_type": "GuiderParameters",
                "inputs": {
                    "modality": "AUDIO",
                    "cfg": 1.0,
                    "stg": 0.0,
                    "perturb_attn": True,
                    "rescale": 0.0,
                    "modality_scale": 1.0,
                    "skip_step": 0,
                    "cross_attn": True,
                    "parameters": ["23", 0],  # chain from VIDEO params
                },
            }

            # Node 25: MultimodalGuider — replaces CFGGuider for AV generation
            workflow["25"] = {
                "class_type": "MultimodalGuider",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["7", 0],
                    "negative": ["7", 1],
                    "parameters": ["24", 0],
                    "skip_blocks": "",
                },
            }

            sampler_latent_input = ["22", 0]  # combined AV latent
            sampler_guider_input = ["25", 0]  # MultimodalGuider

            logger.info("🔊 Audio pipeline enabled — MultimodalGuider + AV latents")

        # Node 14: SamplerCustomAdvanced — main sampling
        workflow["14"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["13", 0],
                "guider": sampler_guider_input,
                "sampler": ["11", 0],
                "sigmas": ["12", 0],
                "latent_image": sampler_latent_input,
            },
        }

        # ── Post-sampling: video decode (always) + audio decode (if AV) ──
        if audio_prompt:
            # Node 26: LTXVSeparateAVLatent — split denoised AV latent
            workflow["26"] = {
                "class_type": "LTXVSeparateAVLatent",
                "inputs": {
                    "av_latent": ["14", 1],  # denoised_output from sampler
                },
            }

            # Node 15: VAEDecodeTiled — decode VIDEO latent (from separated output)
            workflow["15"] = {
                "class_type": "VAEDecodeTiled",
                "inputs": {
                    "samples": ["26", 0],  # video_latent from separator
                    "vae": ["1", 2],
                    "tile_size": 512,
                    "overlap": 64,
                    "temporal_size": 512,
                    "temporal_overlap": 64,
                },
            }

            # Node 27: LTXVAudioVAEDecode — decode AUDIO latent
            workflow["27"] = {
                "class_type": "LTXVAudioVAEDecode",
                "inputs": {
                    "samples": ["26", 1],  # audio_latent from separator
                    "audio_vae": ["20", 0],  # audio VAE from loader
                },
            }

            # Node 16: VHS_VideoCombine — save video WITH audio
            workflow["16"] = {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "frame_rate": fps,
                    "loop_count": 0,
                    "filename_prefix": output_prefix,
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 17,
                    "save_metadata": True,
                    "trim_to_audio": True,
                    "pingpong": False,
                    "save_output": True,
                    "images": ["15", 0],
                    "audio": ["27", 0],  # decoded audio waveform
                },
            }
        else:
            # Node 15: VAEDecodeTiled — decode video-only latent
            workflow["15"] = {
                "class_type": "VAEDecodeTiled",
                "inputs": {
                    "samples": ["14", 1],  # denoised_output
                    "vae": ["1", 2],
                    "tile_size": 512,
                    "overlap": 64,
                    "temporal_size": 512,
                    "temporal_overlap": 64,
                },
            }

            # Node 16: VHS_VideoCombine — save video (no audio)
            workflow["16"] = {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "frame_rate": fps,
                    "loop_count": 0,
                    "filename_prefix": output_prefix,
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 17,
                    "save_metadata": True,
                    "trim_to_audio": False,
                    "pingpong": False,
                    "save_output": True,
                    "images": ["15", 0],
                },
            }

        # ── LoRA chain (single-stage: insert between checkpoint and CFGGuider) ──
        if lora_configs:
            current_model = ["1", 0]  # CheckpointLoaderSimple MODEL output
            lora_node_id = 100
            for i, config in enumerate(lora_configs):
                lora_name = config.get("high") or config.get("name", "")
                if not lora_name:
                    continue
                strength_lora = config.get("strength", 1.0)
                node_id = str(lora_node_id + i)
                workflow[node_id] = {
                    "class_type": "LoraLoaderModelOnly",
                    "inputs": {
                        "lora_name": lora_name,
                        "strength_model": strength_lora,
                        "model": current_model,
                    },
                }
                current_model = [node_id, 0]
                logger.info(
                    f"🎨 LTX-2.3 I2V LoRA #{i + 1}: {lora_name} @ {strength_lora}"
                )
            # Wire final LoRA output to guider (CFGGuider or MultimodalGuider)
            guider_node = "25" if audio_prompt else "10"
            workflow[guider_node]["inputs"]["model"] = current_model

        lora_info = ""
        if lora_configs and len(lora_configs) > 0:
            lora_info = (
                f", {len(lora_configs)} LoRA{'s' if len(lora_configs) > 1 else ''}"
            )
        audio_info = " + audio" if audio_prompt else ""
        logger.info(
            f"☁️ Built LTX-2.3 Cloud I2V: {width}x{height}, "
            f"{num_frames}f@{fps}fps, strength={strength}, 8-step distilled{lora_info}{audio_info}"
        )
        return workflow

    # ── MiniMax-H3 cloud workflows ────────────────────────────────────
    #
    # MiniMax-H3 is a joint video+audio DiT (FL2VA). It always generates a
    # synchronized soundtrack: the AV latent carries both a video stream
    # (24 channels, 24 fps) and an audio stream (40 fps). The workflows below
    # mirror Comfy-Org's official "Image to Video (MiniMax H3)" template:
    #   UNETLoader(fl2va int8) + CLIPLoader(qwen3vl minimax) + 2× VAELoader
    #   → MiniMaxH3ImageToVideo (conditioning + AV latent)
    #   → BasicGuider + KSamplerSelect(res_multistep) + BasicScheduler(simple)
    #   → SamplerCustomAdvanced → VAEDecode (video) + VAEDecodeAudio (audio)
    #   → VHS_VideoCombine (mp4 with muxed audio).
    # The same fl2va checkpoint serves t2v AND i2v (i2v anchors the input
    # image as the first keyframe via MiniMaxH3ImageToVideo.first_frame).

    @staticmethod
    def _align_minimax_h3_frames(num_frames: int) -> int:
        """Snap a frame count to MiniMax-H3's 17k+5 grid at 24 fps."""
        n = max(5, int(num_frames))
        while n % 17 != 5:
            n += 1
        return n

    @staticmethod
    def _aspect_ratio_pair(aspect_ratio: str) -> Tuple[int, int]:
        """Resolve an aspect ratio string (e.g. '16:9') to a (w, h) pair."""
        pairs = {
            "1:1": (1, 1),
            "9:16": (9, 16),
            "16:9": (16, 9),
            "4:3": (4, 3),
            "3:4": (3, 4),
            "3:2": (3, 2),
            "2:3": (2, 3),
            "21:9": (21, 9),
            "9:21": (9, 21),
        }
        return pairs.get(aspect_ratio, (16, 9))

    @staticmethod
    def _minimax_h3_canvas(
        aspect_ratio: str, megapixels: Optional[float] = None
    ) -> Tuple[int, int]:
        """Compute the MiniMax-H3 output canvas (w, h), rounded to multiples of 32.

        Without *megapixels*: H3's native canvas — a 768px short edge capped at
        768×1344 pixels, using the exact same math as ComfyUI's official
        MiniMaxH3 nodes (`adapt_canvas` in nodes_minimax_h3.py).

        With *megapixels*: the official ResolutionSelector formula —
        target area = megapixels × 1024² px at the given aspect ratio, rounded
        to the nearest multiple of 32 (0.4 MP @16:9 → 864×480, 0.98 MP @16:9 →
        1344×768, 2.0 MP @16:9 → 1920×1088). Clamped to 0.2–2.0 MP.
        """
        ar_w, ar_h = ComfyUIClient._aspect_ratio_pair(aspect_ratio)

        if megapixels is None:
            ratio = ar_w / ar_h
            if ratio >= 1.0:
                nom_w, nom_h = 768.0 * ratio, 768.0
            else:
                nom_w, nom_h = 768.0, 768.0 / ratio
            max_pixels = 768 * 1344
            if nom_w * nom_h > max_pixels:
                s = math.sqrt(max_pixels / (nom_w * nom_h))
                nom_w, nom_h = nom_w * s, nom_h * s
            width = max(32, round(nom_w / 32) * 32)
            height = max(32, round(nom_h / 32) * 32)
            return width, height

        mp = max(0.2, min(2.0, float(megapixels)))
        total_pixels = mp * 1024 * 1024
        scale = math.sqrt(total_pixels / (ar_w * ar_h))
        width = max(32, round(ar_w * scale / 32) * 32)
        height = max(32, round(ar_h * scale / 32) * 32)
        return width, height

    def _build_minimax_h3_workflow(
        self,
        *,
        prompt: str,
        width: int,
        height: int,
        length: int,
        fps: int,
        seed: int,
        steps: int,
        output_prefix: str,
        checkpoint: str,
        text_encoder: str,
        video_vae: str,
        audio_vae: str,
        first_frame_link: Optional[list] = None,
        last_frame_link: Optional[list] = None,
        lora_configs: Optional[List[Dict[str, Any]]] = None,
        output_kind: str = "vhs",
        quality_mode: str = "full",
        model_variant: str = "official",
    ) -> Dict[str, Any]:
        """Shared MiniMax-H3 sampling graph (model load → AV decode → mp4).

        *output_kind* selects how the decoded frames + audio are written:
          - "vhs" (default, cloud / ai-kvm2): the `VHS_VideoCombine` custom-node
            (VideoHelperSuite), which is available on the RunPod workers.
          - "savevideo" (local Windows PC): core `CreateVideo` + `SaveVideo`
            nodes, which exist on the Windows portable install where
            VideoHelperSuite (VHS_VideoCombine) is NOT installed.
        *quality_mode* selects the turbo preset ("draft" = 4-step LoRA + shift
        6/3, "standard" = 8-step LoRA, "full" = base 20-step behaviour).
        *model_variant* selects the checkpoint ("official" = Comfy-Org,
        "eros" = H3 Eros Max NSFW finetune with turbo baked in — draft/
        standard only; "full" falls back to official).
        """
        workflow = {}

        preset = MINIMAX_H3_TURBO_PRESETS.get(quality_mode or "full")
        variant = MINIMAX_H3_VARIANTS.get(model_variant)
        if variant and model_variant != "official":
            # Community checkpoints: turbo variants carry a baked distillation,
            # so draft/standard run them without the turbo LoRA or sigma-shift
            # nodes. Non-turbo variants (and turbo variants asked for "full")
            # keep the caller's step count and the base res_multistep sampler.
            checkpoint = variant["checkpoint"]
            if variant["turbo"] and quality_mode in ("draft", "standard"):
                preset = {
                    "lora": None,
                    "steps": 4 if quality_mode == "draft" else 8,
                    "shift_video": None,
                    "shift_audio": None,
                }
                logger.info(
                    f"🔥 MiniMax-H3 variant '{model_variant}' (turbo): "
                    f"{preset['steps']} steps, distillation baked in"
                )
            else:
                preset = None
                logger.info(
                    f"🔥 MiniMax-H3 variant '{model_variant}': "
                    f"{steps} steps, base sampler"
                )
        if preset:
            steps = preset["steps"]
            logger.info(
                f"⚡ MiniMax-H3 turbo mode '{quality_mode}': "
                f"{preset['steps']} steps, lora={preset['lora']}"
            )

        # Node 1: UNETLoader — FL2VA diffusion model (t2v + i2v keyframes)
        workflow["1"] = {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": checkpoint,
                "weight_dtype": "default",
            },
        }
        # The diffusion model the guider + scheduler sample from. With LoRAs we
        # chain LoraLoaderModelOnly nodes in front of the base model, so this
        # link points at the last loader in the chain.
        model_link = ["1", 0]

        # Node 2: CLIPLoader — Qwen3-VL-32B MiniMax text encoder
        workflow["2"] = {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": text_encoder,
                "type": "minimax",
                "device": "default",
            },
        }

        # Node 3: VAELoader — video VAE
        workflow["3"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": video_vae},
        }

        # Node 4: VAELoader — audio VAE (H3 generates audio unconditionally)
        workflow["4"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": audio_vae},
        }

        # Node 5: MiniMaxH3ImageToVideo — prompt → conditioning + AV latent
        h3_inputs = {
            "clip": ["2", 0],
            "vae": ["3", 0],
            "prompt": prompt,
            "width": width,
            "height": height,
            "length": length,
        }
        if first_frame_link:
            h3_inputs["first_frame"] = first_frame_link
        if last_frame_link:
            h3_inputs["last_frame"] = last_frame_link
        workflow["5"] = {
            "class_type": "MiniMaxH3ImageToVideo",
            "inputs": h3_inputs,
        }

        # Apply LoRAs (single-stage) by chaining LoraLoaderModelOnly in front of
        # the base model. With a turbo preset the turbo LoRA chains first, then
        # any user LoRAs; the guider + scheduler reference model_link, which
        # points at the last loader in the chain.
        chain: List[Dict[str, Any]] = list(lora_configs or [])
        if preset and preset.get("lora"):
            chain = [{"name": preset["lora"], "strength": 1.0}] + chain

        lora_node_id = 16
        for i, cfg in enumerate(chain):
            if not cfg:
                continue
            lora_name = cfg.get("name") or ""
            strength = cfg.get("strength", 1.0)
            if not lora_name:
                continue
            workflow[str(lora_node_id)] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": model_link,
                    "lora_name": lora_name,
                    "strength_model": strength,
                },
            }
            model_link = [str(lora_node_id), 0]
            logger.info(f"🎨 MiniMax-H3 LoRA #{i + 1}: {lora_name} @ {strength}")
            lora_node_id += 1

        # Turbo sigma-shift override: only the 768p-trained presets need it
        # (shift 6/3 instead of the checkpoint-baked 12/3).
        if preset and preset.get("shift_video") is not None:
            workflow[str(lora_node_id)] = {
                "class_type": "MiniMaxH3SigmaShift",
                "inputs": {
                    "model": model_link,
                    "shift_video": preset["shift_video"],
                    "shift_audio": preset["shift_audio"],
                },
            }
            model_link = [str(lora_node_id), 0]
            logger.info(
                f"⚡ MiniMax-H3 turbo sigma shift: "
                f"{preset['shift_video']}/{preset['shift_audio']}"
            )
            lora_node_id += 1

        # Node 6: BasicGuider — no CFG / no negative prompt for H3
        workflow["6"] = {
            "class_type": "BasicGuider",
            "inputs": {
                "model": model_link,
                "conditioning": ["5", 0],
            },
        }

        # Node 7: KSamplerSelect — res_multistep (official template); turbo
        # presets use euler (ModelTC reference workflows).
        workflow["7"] = {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler" if preset else "res_multistep"},
        }

        # Node 8: BasicScheduler — simple schedule (official template)
        workflow["8"] = {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": model_link,
                "scheduler": "simple",
                "steps": steps,
                "denoise": 1.0,
            },
        }

        # Node 9: RandomNoise
        workflow["9"] = {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        }

        # Node 10: SamplerCustomAdvanced — samples the flat AV pack
        workflow["10"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["9", 0],
                "guider": ["6", 0],
                "sampler": ["7", 0],
                "sigmas": ["8", 0],
                "latent_image": ["5", 1],
            },
        }

        # Node 11: VAEDecode — video frames from the AV latent
        workflow["11"] = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["10", 0],
                "vae": ["3", 0],
            },
        }

        # Node 12: VAEDecodeAudio — soundtrack from the AV latent
        workflow["12"] = {
            "class_type": "VAEDecodeAudio",
            "inputs": {
                "samples": ["10", 0],
                "vae": ["4", 0],
            },
        }

        # Node 13: output node — mp4 with muxed audio
        if output_kind == "savevideo":
            # Core CreateVideo (frames + audio -> VIDEO) + SaveVideo (-> mp4).
            # Used on the Windows PC portable ComfyUI where VideoHelperSuite
            # (VHS_VideoCombine) is not installed.
            #
            # Node 14 is reserved for the I2V LoadImage first-frame keyframe, so
            # SaveVideo lives on node 15 to avoid a collision.
            workflow["13"] = {
                "class_type": "CreateVideo",
                "inputs": {
                    "images": ["11", 0],
                    "fps": float(fps),
                    "audio": ["12", 0],
                },
            }
            workflow["15"] = {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["13", 0],
                    "filename_prefix": output_prefix,
                    "format": "auto",
                    "codec": "auto",
                },
            }
        else:
            # VHS_VideoCombine — VideoHelperSuite custom node (available on
            # the RunPod / ai-kvm2 workers).
            workflow["13"] = {
                "class_type": "VHS_VideoCombine",
                "inputs": {
                    "frame_rate": fps,
                    "loop_count": 0,
                    "filename_prefix": output_prefix,
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 17,
                    "save_metadata": True,
                    "trim_to_audio": True,
                    "pingpong": False,
                    "save_output": True,
                    "images": ["11", 0],
                    "audio": ["12", 0],
                },
            }

        return workflow

    def build_cloud_minimax_h3_t2v_workflow(
        self,
        prompt: str,
        width: int = 1344,
        height: int = 768,
        num_frames: int = 124,
        fps: int = 24,
        seed: int = -1,
        steps: int = 20,
        output_prefix: str = "oelala_minimax_h3_t2v",
        checkpoint: str = "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        text_encoder: str = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        video_vae: str = "minimax_h3_video_vae_fp16.safetensors",
        audio_vae: str = "minimax_h3_audio_vae_fp32.safetensors",
        aspect_ratio: str = "16:9",
        megapixels: Optional[float] = None,
        long_edge: int = 768,
        lora_configs: Optional[List[Dict[str, Any]]] = None,
        output_kind: str = "vhs",
        quality_mode: str = "full",
        model_variant: str = "official",
    ) -> Optional[Dict[str, Any]]:
        """
        Build MiniMax-H3 (full DiT ~33B; we run the ~21B pruned int8) Cloud T2V workflow — text-to-video+audio.

        Matches Comfy-Org's official MiniMax-H3 template (simple/20-step
        schedule, res_multistep sampler, BasicGuider — no negative prompt).
        The model always generates a synchronized soundtrack.

        Canvas: native 768px short edge (capped 768×1344) unless *megapixels*
        is set, in which case the official ResolutionSelector formula is used
        (0.4 MP @16:9 → 864×480, 0.98 → 1344×768, 2.0 → 1920×1088).
        """
        length = self._align_minimax_h3_frames(num_frames)
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        width, height = self._minimax_h3_canvas(aspect_ratio, megapixels)

        logger.info(
            f"☁️ Building MiniMax-H3 Cloud T2V: {width}x{height}, "
            f"{length}f@{fps}fps, {steps} steps, seed={seed}"
        )

        return self._build_minimax_h3_workflow(
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            fps=fps,
            seed=seed,
            steps=steps,
            output_prefix=output_prefix,
            checkpoint=checkpoint,
            text_encoder=text_encoder,
            video_vae=video_vae,
            audio_vae=audio_vae,
            lora_configs=lora_configs,
            output_kind=output_kind,
            quality_mode=quality_mode,
            model_variant=model_variant,
        )

    def build_cloud_minimax_h3_i2v_workflow(
        self,
        image_name: str,
        prompt: str,
        width: int = 1344,
        height: int = 768,
        num_frames: int = 124,
        fps: int = 24,
        seed: int = -1,
        steps: int = 20,
        output_prefix: str = "oelala_minimax_h3_i2v",
        checkpoint: str = "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        text_encoder: str = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        video_vae: str = "minimax_h3_video_vae_fp16.safetensors",
        audio_vae: str = "minimax_h3_audio_vae_fp32.safetensors",
        aspect_ratio: str = "16:9",
        megapixels: Optional[float] = None,
        long_edge: int = 768,
        lora_configs: Optional[List[Dict[str, Any]]] = None,
        output_kind: str = "vhs",
        quality_mode: str = "full",
        model_variant: str = "official",
    ) -> Optional[Dict[str, Any]]:
        """
        Build MiniMax-H3 (full DiT ~33B; we run the ~21B pruned int8) Cloud I2V workflow — image-to-video+audio.

        Uses the same FL2VA checkpoint as T2V: the input image is anchored
        as the first keyframe of the video (MiniMaxH3ImageToVideo.first_frame).
        Canvas follows the same rule as T2V (native 768p short edge, or the
        official ResolutionSelector formula when *megapixels* is set).
        """
        length = self._align_minimax_h3_frames(num_frames)
        if seed < 0:
            seed = random.randint(0, 2**31 - 1)

        width, height = self._minimax_h3_canvas(aspect_ratio, megapixels)

        logger.info(
            f"☁️ Building MiniMax-H3 Cloud I2V: {width}x{height}, "
            f"{length}f@{fps}fps, {steps} steps, seed={seed}, image={image_name}"
        )

        workflow = self._build_minimax_h3_workflow(
            prompt=prompt,
            width=width,
            height=height,
            length=length,
            fps=fps,
            seed=seed,
            steps=steps,
            output_prefix=output_prefix,
            checkpoint=checkpoint,
            text_encoder=text_encoder,
            video_vae=video_vae,
            audio_vae=audio_vae,
            first_frame_link=["14", 0],
            lora_configs=lora_configs,
            output_kind=output_kind,
            quality_mode=quality_mode,
            model_variant=model_variant,
        )

        # Node 14: LoadImage — first-frame keyframe
        workflow["14"] = {
            "class_type": "LoadImage",
            "inputs": {"image": image_name},
        }
        return workflow

    def _resolve_local_lora_configs(
        self, lora_configs: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        """Re-map requested LoRA names onto what the target server actually has.

        The Windows-PC ComfyUI enumerates loras with backslash separators, so a
        forward-slash registry name fails prompt validation ("Value not in
        list"). Names the server does not have at all are dropped with a
        warning instead of queueing a prompt that can never run.
        """
        if not lora_configs:
            return lora_configs
        available = self.get_lora_names()
        if not available:
            logger.warning(
                "🎨 Could not read the server LoRA list — passing requested"
                " LoRA names through unchanged"
            )
            return lora_configs
        resolved = []
        for cfg in lora_configs:
            if not cfg:
                continue
            name = _match_lora_name(cfg.get("name") or "", available)
            if name is None:
                logger.warning(
                    f"🎨 MiniMax-H3 LoRA not present on server, skipping:"
                    f" {cfg.get('name')}"
                )
                continue
            if name != cfg.get("name"):
                logger.info(f"🎨 MiniMax-H3 LoRA resolved to server name: {name}")
            resolved.append({**cfg, "name": name})
        return resolved or None

    def build_local_minimax_h3_t2v_workflow(
        self,
        prompt: str,
        width: int = 1344,
        height: int = 768,
        num_frames: int = 124,
        fps: int = 24,
        seed: int = -1,
        steps: int = 20,
        output_prefix: str = "oelala_minimax_h3_t2v",
        checkpoint: str = "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        text_encoder: str = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        video_vae: str = "minimax_h3_video_vae_fp16.safetensors",
        audio_vae: str = "minimax_h3_audio_vae_fp32.safetensors",
        aspect_ratio: str = "16:9",
        megapixels: Optional[float] = None,
        long_edge: int = 768,
        lora_configs: Optional[List[Dict[str, Any]]] = None,
        quality_mode: str = "full",
    ) -> Optional[Dict[str, Any]]:
        """
        Build MiniMax-H3 local (Windows PC ComfyUI) T2V workflow.

        Same MiniMax-H3 FL2VA graph as the cloud builder, but uses the model
        files on the user's Windows PC (ComfyUI portable). The local text
        encoder is the **int8_convrot** variant (`qwen3vl_32b_minimax_h3_int8_convrot.safetensors`),
        matching the Comfy-Org int8 pruned set that fits the 16 GB GPU —
        unlike the cloud worker's nvfp4_awq encoder.

        Always generates a synchronized soundtrack (H3 FL2VA).
        """
        return self.build_cloud_minimax_h3_t2v_workflow(
            prompt=prompt,
            num_frames=num_frames,
            fps=fps,
            seed=seed,
            steps=steps,
            output_prefix=output_prefix,
            checkpoint=checkpoint,
            text_encoder=text_encoder,
            video_vae=video_vae,
            audio_vae=audio_vae,
            aspect_ratio=aspect_ratio,
            megapixels=megapixels,
            lora_configs=self._resolve_local_lora_configs(lora_configs),
            output_kind="savevideo",
            quality_mode=quality_mode,
        )

    def build_local_minimax_h3_i2v_workflow(
        self,
        image_name: str,
        prompt: str,
        width: int = 1344,
        height: int = 768,
        num_frames: int = 124,
        fps: int = 24,
        seed: int = -1,
        steps: int = 20,
        output_prefix: str = "oelala_minimax_h3_i2v",
        checkpoint: str = "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        text_encoder: str = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors",
        video_vae: str = "minimax_h3_video_vae_fp16.safetensors",
        audio_vae: str = "minimax_h3_audio_vae_fp32.safetensors",
        aspect_ratio: str = "16:9",
        megapixels: Optional[float] = None,
        long_edge: int = 768,
        lora_configs: Optional[List[Dict[str, Any]]] = None,
        quality_mode: str = "full",
    ) -> Optional[Dict[str, Any]]:
        """
        Build MiniMax-H3 local (Windows PC ComfyUI) I2V workflow.

        Same FL2VA graph as the cloud I2V builder, but with the local
        int8_convrot text encoder that lives on the Windows PC. The input
        image is anchored as the first keyframe. Always generates audio.
        """
        return self.build_cloud_minimax_h3_i2v_workflow(
            image_name=image_name,
            prompt=prompt,
            num_frames=num_frames,
            fps=fps,
            seed=seed,
            steps=steps,
            output_prefix=output_prefix,
            checkpoint=checkpoint,
            text_encoder=text_encoder,
            video_vae=video_vae,
            audio_vae=audio_vae,
            aspect_ratio=aspect_ratio,
            megapixels=megapixels,
            lora_configs=self._resolve_local_lora_configs(lora_configs),
            output_kind="savevideo",
            quality_mode=quality_mode,
        )

    def queue_prompt(self, workflow: Dict[str, Any]) -> Optional[str]:
        """Queue workflow for execution, return prompt_id.

        Unloads the Guardian LLM from VRAM before queuing so ComfyUI has
        the full 28 GB available. Guardian auto-reloads on the next
        inference request.
        """
        # Free LLM VRAM before generation (non-fatal if Guardian is down)
        get_guardian().unload_sync()

        try:
            # Check if already in API format (has string keys like "1", "2")
            if (
                isinstance(list(workflow.keys())[0], str)
                and list(workflow.keys())[0].isdigit()
            ):
                api_workflow = workflow  # Already API format
            else:
                # Legacy: convert from node format
                api_workflow = self._convert_to_api_format(workflow)

            payload = {"prompt": api_workflow, "client_id": self.client_id}

            resp = requests.post(f"{self.base_url}/prompt", json=payload)

            if resp.status_code == 200:
                result = resp.json()
                prompt_id = result.get("prompt_id")
                logger.info(f"📋 Workflow queued: {prompt_id}")
                return prompt_id
            else:
                logger.error(f"Queue failed: {resp.status_code} - {resp.text}")
                return None
        except Exception as e:
            logger.error(f"Queue error: {e}")
            return None

    def _convert_to_api_format(self, workflow: Dict[str, Any]) -> Dict[str, Any]:
        """Convert node-based workflow to API format"""
        api_format = {}

        for node in workflow.get("nodes", []):
            node_id = str(node["id"])

            api_node = {"class_type": node["type"], "inputs": {}}

            # Add widget values as inputs
            widgets = node.get("widgets_values", [])
            node_type = node["type"]

            # Map widget values to input names based on node type
            if node_type == "LoadImage":
                if widgets:
                    api_node["inputs"]["image"] = (
                        widgets[0] if isinstance(widgets[0], str) else "input_image.png"
                    )

            elif node_type == "CLIPVisionLoader":
                if widgets:
                    api_node["inputs"]["clip_name"] = widgets[0]

            elif node_type == "VHS_VideoCombine":
                if isinstance(widgets, dict):
                    api_node["inputs"]["frame_rate"] = widgets.get("frame_rate", 16)
                    api_node["inputs"]["loop_count"] = widgets.get("loop_count", 0)
                    api_node["inputs"]["filename_prefix"] = widgets.get(
                        "filename_prefix", "oelala"
                    )
                    api_node["inputs"]["format"] = widgets.get(
                        "format", "video/h264-mp4"
                    )
                    api_node["inputs"]["pingpong"] = widgets.get("pingpong", False)
                    api_node["inputs"]["save_output"] = widgets.get("save_output", True)

            # Add linked inputs from other nodes
            for inp in node.get("inputs", []):
                if inp.get("link") is not None:
                    # Find source node for this link
                    for link in workflow.get("links", []):
                        if link[0] == inp["link"]:
                            source_node_id = str(link[1])
                            source_slot = link[2]
                            api_node["inputs"][inp["name"]] = [
                                source_node_id,
                                source_slot,
                            ]
                            break

            api_format[node_id] = api_node

        return api_format

    def register_job(
        self,
        prompt_id: str,
        user_id: str,
        prompt: str = "",
        settings: Optional[Dict[str, Any]] = None,
    ):
        """Register job metadata for tracking and auto-upload on completion.

        Args:
            prompt_id: ComfyUI prompt ID
            user_id: User ID from JWT token
            prompt: Generation prompt
            settings: Additional settings (resolution, frames, etc.)
        """
        with self._metadata_lock:
            self.job_metadata[prompt_id] = {
                "user_id": user_id,
                "prompt": prompt,
                "settings": (settings or {}).copy(),
                "started_at": datetime.now().isoformat(),
            }
        logger.info(f"📝 Registered job {prompt_id} for user {user_id}")

    def get_job_metadata(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get job metadata for a prompt ID.

        Returns a deep copy to prevent external mutation of internal state,
        including nested structures such as the `settings` dictionary.
        """
        with self._metadata_lock:
            metadata = self.job_metadata.get(prompt_id)
            return copy.deepcopy(metadata) if metadata is not None else None

    def clear_job_metadata(self, prompt_id: str):
        """Clear job metadata after completion."""
        with self._metadata_lock:
            self.job_metadata.pop(prompt_id, None)

    def on_job_complete(
        self,
        prompt_id: str,
        output_path: str,
        output_type: str = "video",
    ) -> Optional[str]:
        """Auto-upload generated content to user storage on job completion.

        Args:
            prompt_id: ComfyUI prompt ID
            output_path: Local path to generated file
            output_type: Type of output ('video', 'image', 'audio')

        Returns:
            Storage path if upload succeeded in the format "{media_type}/{filename}",
            for example "images/1234567890_output.png", or None if the upload failed.
        """
        # Get job metadata
        metadata = self.get_job_metadata(prompt_id)
        if not metadata:
            logger.warning(
                f"⚠️ No job metadata found for {prompt_id}, skipping auto-upload"
            )
            return None

        user_id = metadata.get("user_id")
        if not user_id:
            logger.warning(f"⚠️ No user_id in job metadata for {prompt_id}")
            return None

        file_data = None
        try:
            # Read file content — try local first, fall back to storage
            file_path = Path(output_path)
            if file_path.exists():
                try:
                    with open(output_path, "rb") as f:
                        file_data = f.read()
                except (IOError, OSError) as e:
                    logger.error(f"❌ Failed to read file {output_path}: {e}")
                    self.clear_job_metadata(prompt_id)
                    return None
            else:
                # Try fetching from storage (path may be "generated/cloud-<family>/file.mp4")
                try:
                    sc = get_storage_client()
                    parts = output_path.replace("\\", "/").split("/", 1)
                    if len(parts) == 2:
                        file_data = sc.get(parts[0], parts[1])
                    if not file_data:
                        raise FileNotFoundError(output_path)
                    logger.info(
                        f"📦 Read {len(file_data)} bytes from storage: {output_path}"
                    )
                except Exception as e:
                    logger.error(
                        f"❌ File not found locally or in storage: {output_path} ({e})"
                    )
                    self.clear_job_metadata(prompt_id)
                    return None

            # Generate storage filename with high-precision timestamp (ms) to avoid collisions
            timestamp = int(datetime.now().timestamp() * 1000)
            original_filename = Path(output_path).name
            storage_filename = f"{timestamp}_{original_filename}"

            # Determine media type folder
            media_type_map = {
                "video": "videos",
                "image": "images",
                "audio": "audio",
            }
            media_type = media_type_map.get(output_type, "generated")

            # Determine content type
            ext = Path(output_path).suffix.lower()
            content_type_map = {
                ".mp4": "video/mp4",
                ".webm": "video/webm",
                ".png": "image/png",
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".webp": "image/webp",
                ".mp3": "audio/mpeg",
                ".wav": "audio/wav",
            }
            content_type = content_type_map.get(ext, "application/octet-stream")

            # Upload to user storage
            storage_client = get_storage_client()
            logger.info(
                f"📤 Uploading {output_type} to user storage: {user_id}/{media_type}/{storage_filename}"
            )

            try:
                storage_client.put_user_media(
                    user_id=user_id,
                    media_type=media_type,
                    filename=storage_filename,
                    data=file_data,
                    content_type=content_type,
                )
            except Exception as e:
                logger.error(f"❌ Storage upload failed for {prompt_id}: {e}")
                # Don't clear metadata on upload failure - allows retry
                return None

            storage_path = f"{media_type}/{storage_filename}"
            logger.info(
                f"✅ Auto-uploaded to storage: {storage_path} ({len(file_data)} bytes)"
            )

            # Cleanup local file to keep media dirs empty
            try:
                local_path = Path(output_path)
                minio_data_dir = Path(
                    os.environ.get("MINIO_DATA_DIR", "/home/flip/minio-data")
                )
                if local_path.exists() and str(local_path.parent) != str(
                    minio_data_dir
                ):
                    local_path.unlink()
                    logger.info(f"🗑️ Cleaned up auto-uploaded local file: {local_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup local file {output_path}: {e}")

            # Clear job metadata after successful upload
            self.clear_job_metadata(prompt_id)

            return storage_path

        except Exception as e:
            logger.error(f"❌ Unexpected error during auto-upload for {prompt_id}: {e}")
            # Don't raise - we don't want to break the user flow if upload fails
            return None

    async def on_job_complete_async(
        self,
        prompt_id: str,
        output_path: str,
        output_type: str = "video",
    ) -> Optional[str]:
        """
        Async version of on_job_complete that uses MediaService for upload + Supabase sync.

        This should be called from async contexts (like FastAPI endpoints) for better
        integration with Supabase metadata tracking and signed URL generation.

        Args:
            prompt_id: ComfyUI prompt ID
            output_path: Local path to generated file
            output_type: Type of output ('video', 'image', 'audio')

        Returns:
            Full storage path (e.g., users/{user_id}/videos/file.mp4) or None if failed
        """
        if get_media_service is None:
            logger.warning("⚠️ MediaService not available, falling back to sync upload")
            return self.on_job_complete(prompt_id, output_path, output_type)

        # Get job metadata
        metadata = self.get_job_metadata(prompt_id)
        if not metadata:
            logger.warning(
                f"⚠️ No job metadata found for {prompt_id}, skipping auto-upload"
            )
            return None

        user_id = metadata.get("user_id")
        if not user_id:
            logger.warning(f"⚠️ No user_id in job metadata for {prompt_id}")
            return None

        try:
            # Read file content — try local first, fall back to storage
            file_path = Path(output_path)
            if file_path.exists():
                file_data = file_path.read_bytes()
            else:
                # Try fetching from storage (path may be "generated/cloud-<family>/file.mp4")
                try:
                    sc = get_storage_client()
                    parts = output_path.replace("\\", "/").split("/", 1)
                    if len(parts) == 2:
                        file_data = sc.get(parts[0], parts[1])
                    else:
                        file_data = None
                    if not file_data:
                        raise FileNotFoundError(output_path)
                    logger.info(
                        f"📦 Read {len(file_data)} bytes from storage: {output_path}"
                    )
                except Exception as e:
                    logger.error(
                        f"❌ File not found locally or in storage: {output_path} ({e})"
                    )
                    self.clear_job_metadata(prompt_id)
                    return None

            # Map output_type to generation_type
            # Settings are nested under metadata["settings"]
            settings = metadata.get("settings", {})
            job_type = settings.get("job_type", "") or settings.get("type", "")
            gen_type_map = {
                "video": "t2v" if "t2v" in job_type else "i2v",
                "image": "t2i",
                "audio": "audio",
            }
            generation_type = gen_type_map.get(output_type, output_type)

            # Extract metadata for Supabase from nested settings
            extra_metadata = {
                "model_name": settings.get("model_name")
                or settings.get("model_type")
                or settings.get("model_mode"),
                "resolution": settings.get("resolution"),
                "aspect_ratio": settings.get("aspect_ratio"),
                "num_frames": settings.get("num_frames"),
                "fps": settings.get("fps"),
                "seed": settings.get("seed"),
                "steps": settings.get("steps"),
                "cfg": settings.get("cfg"),
                "size_bytes": len(file_data),
            }
            # Remove None values
            extra_metadata = {k: v for k, v in extra_metadata.items() if v is not None}

            # Upload via MediaService (storage + Supabase sync)
            media_service = get_media_service()
            record = await media_service.upload(
                user_id=user_id,
                file_data=file_data,
                filename=file_path.name,
                generation_type=generation_type,
                prompt=metadata.get("prompt", ""),
                model_name=extra_metadata.get("model_name"),
                workflow_id=prompt_id,
            )

            logger.info(
                f"✅ Async uploaded to storage: {record.storage_path} ({len(file_data)} bytes)"
            )

            # Cleanup local file to keep media dirs empty
            try:
                if file_path.exists():
                    file_path.unlink()
                    logger.info(f"🗑️ Cleaned up async local file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to cleanup async local file {file_path}: {e}")

            # Clear job metadata after successful upload
            self.clear_job_metadata(prompt_id)

            return record.storage_path

        except Exception as e:
            logger.error(f"❌ Async upload failed for {prompt_id}: {e}")
            # Fallback to sync upload
            logger.info("🔄 Falling back to sync upload...")
            return self.on_job_complete(prompt_id, output_path, output_type)

    def wait_for_completion(
        self,
        prompt_id: str,
        timeout: int = 1800,  # 30 minutes for longer generations
        progress_callback=None,
    ) -> Optional[Dict[str, Any]]:
        """Wait for workflow completion using websocket"""
        # Node ID to friendly name mapping for progress display
        NODE_NAMES = {
            "1": "📷 Load Image",
            "2": "🔧 Load GGUF Model",
            "3": "📝 T5 Text Encoder",
            "4": "🎨 VAE Loader",
            "5": "💬 Text Encode",
            "6": "🖼️ Image Encode",
            "7": "🎬 Video Sampler",
            "8": "🔄 VAE Decode",
            "9": "💾 Video Combine",
            "10": "📊 CLIP Vision",
            "11": "🎯 Sampler Stage 2",
            "12": "🎥 Video Output",
        }

        current_node = None
        current_node_name = "Starting..."

        try:
            ws_url = f"ws://{self.host}:{self.port}/ws?clientId={self.client_id}"
            ws = websocket.create_connection(ws_url, timeout=30)

            start_time = time.time()

            while time.time() - start_time < timeout:
                try:
                    message = ws.recv()
                    data = json.loads(message)

                    msg_type = data.get("type")
                    msg_data = data.get("data", {})

                    if msg_type == "progress":
                        value = msg_data.get("value", 0)
                        max_val = msg_data.get("max", 100)
                        pct = int(100 * value / max_val) if max_val > 0 else 0
                        node_id = str(msg_data.get("node", ""))
                        node_name = NODE_NAMES.get(node_id, f"Node {node_id}")
                        logger.info(
                            f"📊 [{node_name}] Progress: {pct}% ({value}/{max_val})"
                        )
                        if progress_callback:
                            # Pass both percentage and process name
                            progress_callback(pct, node_name)

                    elif msg_type == "executing":
                        node_id = msg_data.get("node")
                        if node_id is None and msg_data.get("prompt_id") == prompt_id:
                            logger.info("✅ Workflow execution complete")
                            ws.close()
                            return self._get_history(prompt_id)
                        elif node_id:
                            current_node = str(node_id)
                            current_node_name = NODE_NAMES.get(
                                current_node, f"Node {current_node}"
                            )
                            logger.info(f"🔄 Executing: {current_node_name}")

                    elif msg_type == "execution_error":
                        logger.error(f"❌ Execution error: {msg_data}")
                        ws.close()
                        return None

                except websocket.WebSocketTimeoutException:
                    continue

            ws.close()
            logger.error("⏰ Timeout waiting for completion")
            return None

        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            return None

    def _get_history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Get execution history for prompt"""
        try:
            resp = requests.get(f"{self.base_url}/history/{prompt_id}")
            if resp.status_code == 200:
                return resp.json().get(prompt_id, {})
            return None
        except Exception as e:
            logger.error(f"History error: {e}")
            return None

    def get_output_video(
        self, history: Dict[str, Any], output_dir: str, prompt_id: Optional[str] = None
    ) -> Optional[str]:
        """Extract output video path from history and auto-upload to user storage.

        Args:
            history: ComfyUI execution history
            output_dir: Local directory to save video
            prompt_id: Optional prompt ID for auto-upload tracking

        Returns:
            Local path to downloaded video
        """
        try:
            outputs = history.get("outputs", {})

            # Find VHS_VideoCombine node output
            for node_id, node_output in outputs.items():
                if "gifs" in node_output:
                    for gif in node_output["gifs"]:
                        filename = gif.get("filename")
                        subfolder = gif.get("subfolder", "")

                        # Download the video
                        params = {
                            "filename": filename,
                            "subfolder": subfolder,
                            "type": "output",
                        }
                        resp = requests.get(f"{self.base_url}/view", params=params)

                        if resp.status_code == 200:
                            output_path = Path(output_dir) / filename
                            with open(output_path, "wb") as f:
                                f.write(resp.content)
                            logger.info(f"📥 Video downloaded: {output_path}")

                            # Upload to generated bucket in MinIO
                            try:
                                storage_client = get_storage_client()
                                storage_client.put("generated", filename, resp.content)
                                logger.info(
                                    f"📤 Video uploaded to storage: generated/{filename}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"⚠️ Storage upload failed (non-fatal): {e}"
                                )

                            # Auto-upload to user storage if prompt_id is provided
                            if prompt_id:
                                storage_path = self.on_job_complete(
                                    prompt_id=prompt_id,
                                    output_path=str(output_path),
                                    output_type="video",
                                )
                                if storage_path:
                                    logger.info(
                                        f"📤 Auto-uploaded video to: {storage_path}"
                                    )

                            return str(output_path)

            logger.warning("No video output found in history")
            return None

        except Exception as e:
            logger.error(f"Output extraction error: {e}")
            return None

    def get_output_image(
        self, history: Dict[str, Any], output_dir: str, prompt_id: Optional[str] = None
    ) -> Optional[str]:
        """Extract output image path from history and auto-upload to user storage.

        Args:
            history: ComfyUI execution history
            output_dir: Local directory to save image
            prompt_id: Optional prompt ID for auto-upload tracking

        Returns:
            Local path to downloaded image
        """
        try:
            outputs = history.get("outputs", {})

            # Find SaveImage node output
            for node_id, node_output in outputs.items():
                if "images" in node_output:
                    for img in node_output["images"]:
                        filename = img.get("filename")
                        subfolder = img.get("subfolder", "")

                        # Download the image from ComfyUI
                        params = {
                            "filename": filename,
                            "subfolder": subfolder,
                            "type": "output",
                        }
                        resp = requests.get(f"{self.base_url}/view", params=params)

                        if resp.status_code == 200:
                            output_path = Path(output_dir) / filename
                            output_path.parent.mkdir(parents=True, exist_ok=True)
                            with open(output_path, "wb") as f:
                                f.write(resp.content)
                            logger.info(f"📥 Image downloaded: {output_path}")

                            # Upload to generated bucket in MinIO
                            try:
                                storage_client = get_storage_client()
                                storage_client.put("generated", filename, resp.content)
                                logger.info(
                                    f"📤 Image uploaded to storage: generated/{filename}"
                                )
                            except Exception as e:
                                logger.warning(
                                    f"⚠️ Storage upload failed (non-fatal): {e}"
                                )

                            # Auto-upload to user storage if prompt_id is provided
                            if prompt_id:
                                storage_path = self.on_job_complete(
                                    prompt_id=prompt_id,
                                    output_path=str(output_path),
                                    output_type="image",
                                )
                                if storage_path:
                                    logger.info(
                                        f"📤 Auto-uploaded image to: {storage_path}"
                                    )

                            return str(output_path)

            logger.warning("No image output found in history")
            return None

        except Exception as e:
            logger.error(f"Image extraction error: {e}")
            return None

    def wait_and_download_image(
        self,
        prompt_id: str,
        output_dir: str,
        timeout: int = 300,
        progress_callback=None,
    ) -> Optional[str]:
        """Wait for workflow completion and download resulting image.

        Args:
            prompt_id: ComfyUI prompt ID
            output_dir: Directory to save downloaded image
            timeout: Timeout in seconds
            progress_callback: Optional callback(percent, node_name)

        Returns:
            Path to downloaded image, or None on failure
        """
        history = self.wait_for_completion(prompt_id, timeout, progress_callback)
        if not history:
            return None
        return self.get_output_image(history, output_dir, prompt_id)

    def build_video_upscale_workflow(
        self,
        video_path: str,
        scale: int = 2,
        output_prefix: str = "upscaled",
        model: str = "realesrgan-x4plus",
    ) -> Optional[Dict]:
        """
        Build a video upscaling workflow using Real-ESRGAN.

        Args:
            video_path: Path to input video
            scale: Upscale factor (2 or 4)
            output_prefix: Prefix for output filename
            model: Upscale model to use

        Returns:
            ComfyUI workflow dict or None if video doesn't exist
        """
        if not Path(video_path).exists():
            logger.error(f"Video not found: {video_path}")
            return None

        # Upload video to ComfyUI
        video_name = self.upload_video(video_path)
        if not video_name:
            logger.error("Failed to upload video for upscaling")
            return None

        workflow = {
            "1": {
                "inputs": {
                    "video": video_name,
                    "force_rate": 0,
                    "force_size": "Disabled",
                    "custom_width": 0,
                    "custom_height": 0,
                    "frame_load_cap": 0,
                    "skip_first_frames": 0,
                    "select_every_nth": 1,
                },
                "class_type": "VHS_LoadVideo",
            },
            "2": {
                "inputs": {
                    "model_name": f"{model}.pth",
                },
                "class_type": "UpscaleModelLoader",
            },
            "3": {
                "inputs": {
                    "upscale_model": ["2", 0],
                    "image": ["1", 0],
                },
                "class_type": "ImageUpscaleWithModel",
            },
            "4": {
                "inputs": {
                    "frame_rate": ["1", 2],
                    "loop_count": 0,
                    "filename_prefix": output_prefix,
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 19,
                    "save_metadata": True,
                    "images": ["3", 0],
                    "audio": ["1", 1],
                },
                "class_type": "VHS_VideoCombine",
            },
        }

        logger.info(f"🔧 Built video upscale workflow: {scale}x with {model}")
        return workflow

    def build_rife_workflow(
        self,
        video_path: str,
        target_fps: int = 60,
        output_prefix: str = "interpolated",
        multiplier: int = 2,
    ) -> Optional[Dict]:
        """
        Build a RIFE frame interpolation workflow.

        Args:
            video_path: Path to input video
            target_fps: Target framerate
            output_prefix: Prefix for output filename
            multiplier: Frame multiplier (2 = double frames, 4 = quadruple)

        Returns:
            ComfyUI workflow dict or None if video doesn't exist
        """
        if not Path(video_path).exists():
            logger.error(f"Video not found: {video_path}")
            return None

        # Upload video to ComfyUI
        video_name = self.upload_video(video_path)
        if not video_name:
            logger.error("Failed to upload video for interpolation")
            return None

        workflow = {
            "1": {
                "inputs": {
                    "video": video_name,
                    "force_rate": 0,
                    "force_size": "Disabled",
                    "custom_width": 0,
                    "custom_height": 0,
                    "frame_load_cap": 0,
                    "skip_first_frames": 0,
                    "select_every_nth": 1,
                },
                "class_type": "VHS_LoadVideo",
            },
            "2": {
                "inputs": {
                    "ckpt_name": "rife49.pth",
                    "clear_cache_after_n_frames": 10,
                    "multiplier": multiplier,
                    "fast_mode": True,
                    "ensemble": True,
                    "scale_factor": 1.0,
                    "frames": ["1", 0],
                },
                "class_type": "RIFE VFI",
            },
            "3": {
                "inputs": {
                    "frame_rate": target_fps,
                    "loop_count": 0,
                    "filename_prefix": output_prefix,
                    "format": "video/h264-mp4",
                    "pix_fmt": "yuv420p",
                    "crf": 19,
                    "save_metadata": True,
                    "images": ["2", 0],
                    "audio": ["1", 1],
                },
                "class_type": "VHS_VideoCombine",
            },
        }

        logger.info(f"🔧 Built RIFE workflow: {multiplier}x → {target_fps}fps")
        return workflow

    def build_video_concat_workflow(
        self,
        video_paths: list,
        output_prefix: str = "concatenated",
        transition: str = "none",
    ) -> Optional[Dict]:
        """
        Build a video concatenation workflow to join multiple videos.

        Args:
            video_paths: List of video paths to concatenate
            output_prefix: Prefix for output filename
            transition: Transition type between clips ("none", "crossfade")

        Returns:
            ComfyUI workflow dict or None
        """
        if not video_paths or len(video_paths) < 2:
            logger.error("Need at least 2 videos to concatenate")
            return None

        # Upload all videos
        video_names = []
        for vp in video_paths:
            if not Path(vp).exists():
                logger.error(f"Video not found: {vp}")
                return None
            name = self.upload_video(vp)
            if not name:
                return None
            video_names.append(name)

        # Build workflow with video loaders and batch
        workflow = {}
        node_id = 1

        # Load each video
        load_nodes = []
        for i, vname in enumerate(video_names):
            workflow[str(node_id)] = {
                "inputs": {
                    "video": vname,
                    "force_rate": 0,
                    "force_size": "Disabled",
                    "custom_width": 0,
                    "custom_height": 0,
                    "frame_load_cap": 0,
                    "skip_first_frames": 0,
                    "select_every_nth": 1,
                },
                "class_type": "VHS_LoadVideo",
            }
            load_nodes.append(str(node_id))
            node_id += 1

        # Batch images together
        workflow[str(node_id)] = {
            "inputs": {
                "images": [load_nodes[0], 0],
            },
            "class_type": "ImageBatch",
        }
        batch_node = str(node_id)

        # Chain additional videos
        for ln in load_nodes[1:]:
            node_id += 1
            workflow[str(node_id)] = {
                "inputs": {
                    "images1": [batch_node, 0],
                    "images2": [ln, 0],
                },
                "class_type": "ImageBatch",
            }
            batch_node = str(node_id)

        node_id += 1

        # Output combined video
        workflow[str(node_id)] = {
            "inputs": {
                "frame_rate": 16,  # Will be overridden by first video's rate
                "loop_count": 0,
                "filename_prefix": output_prefix,
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "crf": 19,
                "save_metadata": True,
                "images": [batch_node, 0],
            },
            "class_type": "VHS_VideoCombine",
        }

        logger.info(f"🔧 Built video concat workflow: {len(video_paths)} videos")
        return workflow


# Singleton instance
_comfyui_client: Optional[ComfyUIClient] = None

def get_comfyui_client() -> ComfyUIClient:
    """Get or create ComfyUI client singleton"""
    global _comfyui_client
    if _comfyui_client is None:
        _comfyui_client = ComfyUIClient()
    return _comfyui_client


# ── Generic per-backend ComfyUI clients ─────────────────────────────
# The Compute Backend Inventory (generation/compute_backends.py) drives which
# server an adapter targets. get_comfyui_client_for_backend() returns a
# (cached) ComfyUIClient for any configured 'comfyui' backend so adding a new
# server is purely a config change (Admin panel → compute_backends.json, or the
# COMPUTE_NODE_* env fallback) — there is no longer a bespoke per-server client
# factory.
_backend_clients: dict = {}

def _parse_base_url(base_url: str):
    """Split 'http://host:port' -> (host, port). Defaults to localhost:8188."""
    host, port = "localhost", 8188
    clean = (base_url or "").strip()
    if clean:
        if "://" in clean:
            clean = clean.split("://", 1)[1]
        host = clean.split("/")[0].split(":")[0]
        if ":" in clean.split("/")[0]:
            try:
                port = int(clean.split("/")[0].split(":")[1])
            except (ValueError, IndexError):
                port = 8188
    return host, port

def get_comfyui_client_for_backend(backend) -> Optional[ComfyUIClient]:
    """Get or create a ComfyUIClient for a ComputeBackend.

    Accepts a ComputeBackend object (generation.compute_backends) or a plain
    dict with keys id/base_url/type. Returns None for runpod backends or when
    no base_url is configured.
    """
    if backend is None:
        return None
    if isinstance(backend, dict):
        backend_id = backend.get("id", "")
        base_url = backend.get("base_url", "")
        btype = backend.get("type", "comfyui")
    else:
        backend_id = backend.id
        base_url = backend.base_url
        btype = backend.type

    if btype == "runpod" or not base_url:
        return None

    # Cache by (backend_id, base_url) so an admin edit to a backend's base_url
    # takes effect on the next dispatch instead of returning the stale client.
    cache_key = (backend_id, base_url)
    if cache_key in _backend_clients:
        return _backend_clients[cache_key]

    # A base_url change created a new cache key; drop any prior entry for the
    # same backend id so repeated admin edits don't grow the cache unboundedly
    # during a long-lived backend process.
    for stale_key in [k for k in _backend_clients if k[0] == backend_id]:
        del _backend_clients[stale_key]
    host, port = _parse_base_url(base_url)
    client = ComfyUIClient(host=host, port=port)
    _backend_clients[cache_key] = client
    logger.info(f"⚙️ ComfyUI client ready for backend '{backend_id}': {host}:{port}")
    return client
