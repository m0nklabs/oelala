#!/usr/bin/env python3
"""
LTX-2 Audio+Video Generation Test Script

This script tests the LTX-2 DEV model with native audio generation.
The DEV model (unlike the distilled version) has full audio support.

Requirements:
- ltx-2-19b-dev-Q4_K_M.gguf (12GB, in diffusion_models/)
- ltx2_audio_vae.safetensors (208MB, in checkpoints/)
- Gemma2 for text encoding
"""

import json
import requests
import time
import uuid
import os

COMFYUI_URL = "http://localhost:8188"

def create_workflow(prompt: str, num_frames: int = 97, width: int = 768, height: int = 512):
    """Create LTX-2 audio+video workflow.
    
    The key insight is that we need to:
    1. Create empty video latent
    2. Create empty audio latent 
    3. Concatenate them with LTXVConcatAVLatent
    4. Sample the combined AV latent with SamplerCustomAdvanced
    5. Separate the output into video and audio latents
    6. Decode both
    """
    
    client_id = str(uuid.uuid4())
    
    workflow = {
        # Model loaders
        "1": {
            "class_type": "UnetLoaderGGUFAdvanced",
            "inputs": {
                "unet_name": "ltx-2-19b-dev-Q4_K_M.gguf",
                "dequant_dtype": "bfloat16",
                "patch_dtype": "bfloat16",
                "patch_on_device": False
            }
        },
        "2": {
            "class_type": "VAELoader",
            "inputs": {
                "vae_name": "LTX2_video_vae_bf16.safetensors"
            }
        },
        "3": {
            "class_type": "LTXVAudioVAELoader",
            "inputs": {
                "ckpt_name": "ltx2_audio_vae.safetensors"
            }
        },
        
        # Text encoding
        "4": {
            "class_type": "LTXVCPUGemmaEncode",
            "inputs": {
                "gemma_path": "gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors",
                "ltxv_path": "ltx-2-19b-distilled-fp8.safetensors",
                "text": prompt,
                "max_length": 256,
                "output_device": "cuda:0"
            }
        },
        "5": {
            "class_type": "LTXVCPUGemmaNegativeEncode",
            "inputs": {
                "gemma_path": "gemma-3-12b-it-qat-q4_0-unquantized/model-00001-of-00005.safetensors",
                "ltxv_path": "ltx-2-19b-distilled-fp8.safetensors",
                "text": "worst quality, low quality, blurry, distorted, no audio",
                "max_length": 256,
                "output_device": "cuda:0"
            }
        },
        
        # Create empty latents
        "10": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": num_frames,
                "batch_size": 1
            }
        },
        "11": {
            "class_type": "LTXVEmptyLatentAudio",
            "inputs": {
                "frames_number": num_frames,
                "frame_rate": 25,
                "batch_size": 1,
                "audio_vae": ["3", 0]
            }
        },
        
        # Combine video + audio latents
        "12": {
            "class_type": "LTXVConcatAVLatent",
            "inputs": {
                "video_latent": ["10", 0],
                "audio_latent": ["11", 0]
            }
        },
        
        # Sampling components
        "20": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "cfg": 3.0
            }
        },
        "21": {
            "class_type": "KSamplerSelect",
            "inputs": {
                "sampler_name": "euler"
            }
        },
        "22": {
            "class_type": "LTXVScheduler",
            "inputs": {
                "steps": 30,
                "max_shift": 2.05,
                "base_shift": 0.95,
                "stg_end_percent": 0.0,
                "stretch": True,
                "terminal": 0.1,
                "stretch_strength": 0.25
            }
        },
        "23": {
            "class_type": "RandomNoise",
            "inputs": {
                "noise_seed": int(time.time())
            }
        },
        
        # Sample the combined AV latent
        "30": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["23", 0],
                "guider": ["20", 0],
                "sampler": ["21", 0],
                "sigmas": ["22", 0],
                "latent_image": ["12", 0]  # Combined AV latent
            }
        },
        
        # Separate video and audio latents from output
        "40": {
            "class_type": "LTXVSeparateAVLatent",
            "inputs": {
                "av_latent": ["30", 1]  # output is (output, denoised_output, ...)
            }
        },
        
        # Decode video
        "50": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["40", 0],  # video_latent
                "vae": ["2", 0]
            }
        },
        
        # Decode audio
        "51": {
            "class_type": "LTXVAudioVAEDecode",
            "inputs": {
                "samples": ["40", 1],  # audio_latent
                "audio_vae": ["3", 0]
            }
        },
        
        # Combine to video file with audio
        "60": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "images": ["50", 0],
                "frame_rate": 25,
                "loop_count": 0,
                "pingpong": False,
                "filename_prefix": "ltx2_audio",
                "format": "video/h264-mp4",
                "save_output": True,
                "audio": ["51", 0]
            }
        }
    }
    
    return workflow, client_id


def queue_workflow(workflow: dict, client_id: str) -> str:
    """Queue workflow and return prompt_id."""
    payload = {
        "prompt": workflow,
        "client_id": client_id
    }
    
    resp = requests.post(f"{COMFYUI_URL}/prompt", json=payload)
    result = resp.json()
    
    if "prompt_id" in result:
        return result["prompt_id"]
    else:
        raise Exception(f"Failed to queue: {result}")


def wait_for_completion(prompt_id: str, timeout: int = 600) -> dict:
    """Wait for workflow completion."""
    start = time.time()
    
    while time.time() - start < timeout:
        resp = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
        if resp.status_code == 200:
            data = resp.json()
            if prompt_id in data:
                status = data[prompt_id].get("status", {})
                if status.get("completed", False):
                    return data[prompt_id]
                if status.get("status_str") == "error":
                    raise Exception(f"Workflow error: {data[prompt_id]}")
        time.sleep(2)
    
    raise TimeoutError(f"Workflow timed out after {timeout}s")


def main():
    prompt = "A cat sitting on a windowsill, looking out at a rainy day. The sound of rain on the window."
    
    print("🎬 LTX-2 Audio+Video Test")
    print(f"📝 Prompt: {prompt}")
    print(f"📦 Model: ltx-2-19b-dev-Q4_K_M.gguf (12GB with audio support)")
    print()
    
    # Create workflow
    workflow, client_id = create_workflow(
        prompt=prompt,
        num_frames=97,  # ~3.9 seconds at 25fps
        width=768,
        height=512
    )
    
    # Queue it
    print("⏳ Queueing workflow...")
    prompt_id = queue_workflow(workflow, client_id)
    print(f"📋 Prompt ID: {prompt_id}")
    
    # Wait for completion
    print("⏳ Waiting for completion...")
    result = wait_for_completion(prompt_id)
    
    print("✅ Done!")
    print(f"📁 Output: {json.dumps(result.get('outputs', {}), indent=2)}")


if __name__ == "__main__":
    main()
