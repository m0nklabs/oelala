#!/usr/bin/env python3
"""
Test script for SFW content generation.
Step 1: Generate a random SFW image
Step 2: Convert to video with I2V (if image works)

Usage:
    python scripts/test_sfw_generation.py
"""

import json
import random
import requests
import time
from pathlib import Path
from datetime import datetime

COMFYUI_URL = "http://localhost:8188"

# SFW prompt categories for diversity
SFW_CATEGORIES = {
    "nature": [
        "majestic mountain range at golden hour, snow-capped peaks, dramatic clouds",
        "serene ocean sunset, waves gently lapping shore, vibrant orange and purple sky",
        "ancient redwood forest, misty atmosphere, sunbeams through trees",
        "aurora borealis over frozen lake, green and purple lights dancing",
        "cherry blossom garden in spring, petals floating in wind",
    ],
    "animals": [
        "majestic eagle soaring over canyon, wings spread wide, clear blue sky",
        "pod of dolphins jumping through waves, sparkling water, sunset",
        "colorful tropical fish in coral reef, underwater sunbeams",
        "fox in snowy forest, breath visible in cold air, winter morning",
        "butterflies around wildflowers, meadow in summer, soft sunlight",
    ],
    "urban": [
        "neon-lit Tokyo street at night, rain reflections on pavement",
        "historic European city square, cobblestone streets, golden hour",
        "futuristic city skyline, glass towers, sunset gradient sky",
        "cozy coffee shop interior, warm lighting, rain on windows",
        "grand library with towering bookshelves, reading nooks, soft light",
    ],
    "abstract": [
        "flowing liquid metal in slow motion, chrome and gold reflections",
        "geometric crystal formations, rainbow light refractions",
        "ink drops spreading in water, vibrant colors mixing",
        "particle explosion in space, trails of light and color",
        "mandala pattern made of light, intricate sacred geometry",
    ],
    "space": [
        "spiral galaxy with billions of stars, cosmic dust clouds",
        "ringed planet with multiple moons, nebula background",
        "asteroid field with distant sun, dramatic lighting",
        "nebula nursery with newborn stars, colorful gas clouds",
        "earth from orbit, city lights visible, aurora at poles",
    ],
    "weather": [
        "dramatic thunderstorm over prairie, lightning strikes, dark clouds",
        "peaceful snowfall in forest, fresh powder on pine trees",
        "rainbow after storm over valley, misty atmosphere",
        "sandstorm approaching ancient ruins, dramatic sky",
        "fog rolling through mountain valley at dawn, layers of hills",
    ],
    "water": [
        "massive waterfall in tropical jungle, mist and rainbows",
        "crystal clear lake reflecting mountain peaks, perfect mirror",
        "powerful ocean wave curling, spray and foam, surfer perspective",
        "rain drops creating ripples on pond, lily pads floating",
        "underwater cave with light beams, crystal clear water",
    ],
    "fire_light": [
        "campfire under starry sky, sparks floating up, cozy atmosphere",
        "fireworks over city skyline, celebration, colorful explosions",
        "sun rays through storm clouds, dramatic crepuscular rays",
        "lantern festival, thousands of paper lanterns rising into night sky",
        "northern lights reflected in still lake, dancing greens and purples",
    ],
    "plants": [
        "field of sunflowers at golden hour, endless rows, blue sky",
        "autumn forest canopy, red and orange leaves, sunlight filtering",
        "lotus flower in zen garden, morning dew, peaceful",
        "bonsai tree with delicate branches, moss and stones, misty",
        "wild mushrooms in fairy forest, bioluminescent glow, magical",
    ],
    "technology": [
        "vintage clockwork mechanism, brass gears, intricate detail",
        "sports car on winding mountain road, motion blur, scenic view",
        "retro-futuristic control room, analog gauges, warm lighting",
        "steam locomotive crossing bridge, dramatic steam, landscape",
        "observatory dome under milky way, telescope pointing at stars",
    ],
}

# Animation prompts for I2V
ANIMATION_PROMPTS = [
    "gentle camera pan, smooth motion, cinematic",
    "slow zoom in, atmospheric, dreamlike",
    "subtle movement, natural sway, peaceful",
    "floating particles, ambient motion, serene",
    "light rays moving, dynamic shadows, ethereal",
    "rippling water effect, gentle waves, calming",
    "slow rotation, revealing details, majestic",
    "drifting clouds, time-lapse style, epic",
]


def get_random_sfw_prompt() -> tuple[str, str, str]:
    """Get a random SFW prompt from a random category."""
    category = random.choice(list(SFW_CATEGORIES.keys()))
    base_prompt = random.choice(SFW_CATEGORIES[category])
    animation_prompt = random.choice(ANIMATION_PROMPTS)
    
    # Enhance with quality tags
    full_prompt = f"{base_prompt}, masterpiece, highly detailed, professional photography, 8k uhd, cinematic lighting, safe for work"
    
    return category, full_prompt, animation_prompt


def create_t2i_workflow(prompt: str, seed: int = None) -> dict:
    """Create a T2I workflow with the given prompt."""
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    
    workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {
                "ckpt_name": "dreamshaperXL_lightningDPMSDE.safetensors"
            }
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["1", 1],
                "text": prompt
            }
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["1", 1],
                "text": "ugly, deformed, blurry, low quality, bad anatomy, watermark, text, nsfw, nude, naked"
            }
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": 1024,
                "height": 1024,
                "batch_size": 1
            }
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": seed,
                "steps": 8,  # Lightning model is fast
                "cfg": 2.0,  # Lower CFG for lightning
                "sampler_name": "dpmpp_sde",
                "scheduler": "karras",
                "denoise": 1.0
            }
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["5", 0],
                "vae": ["1", 2]
            }
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["6", 0],
                "filename_prefix": "sfw_test"
            }
        }
    }
    
    return workflow, seed


def queue_prompt(workflow: dict) -> str:
    """Queue a workflow and return the prompt_id."""
    payload = {"prompt": workflow}
    
    try:
        response = requests.post(f"{COMFYUI_URL}/prompt", json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("prompt_id")
    except requests.exceptions.RequestException as e:
        print(f"❌ Error queueing prompt: {e}")
        return None


def wait_for_completion(prompt_id: str, timeout: int = 300) -> bool:
    """Wait for workflow to complete."""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
            if response.status_code == 200:
                history = response.json()
                if prompt_id in history:
                    return True
        except requests.exceptions.RequestException:
            pass
        
        time.sleep(2)
        print("⏳ Waiting for generation...", end="\r")
    
    return False


def get_output_images(prompt_id: str) -> list[str]:
    """Get output image paths from completed workflow."""
    try:
        response = requests.get(f"{COMFYUI_URL}/history/{prompt_id}")
        if response.status_code == 200:
            history = response.json()
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                images = []
                for node_id, node_output in outputs.items():
                    if "images" in node_output:
                        for img in node_output["images"]:
                            images.append(img.get("filename"))
                return images
    except requests.exceptions.RequestException as e:
        print(f"❌ Error getting outputs: {e}")
    
    return []


def main():
    print("=" * 60)
    print("🎨 SFW Content Generation Test")
    print("=" * 60)
    
    # Get random prompt
    category, prompt, animation = get_random_sfw_prompt()
    
    print(f"\n📂 Category: {category}")
    print(f"📝 Prompt: {prompt[:100]}...")
    print(f"🎬 Animation: {animation}")
    
    # Create workflow
    workflow, seed = create_t2i_workflow(prompt)
    print(f"🎲 Seed: {seed}")
    
    # Check ComfyUI is running
    try:
        response = requests.get(f"{COMFYUI_URL}/system_stats")
        if response.status_code != 200:
            print("❌ ComfyUI not responding")
            return
        print("✅ ComfyUI is running")
    except requests.exceptions.RequestException:
        print("❌ Cannot connect to ComfyUI at", COMFYUI_URL)
        return
    
    # Queue the workflow
    print("\n🚀 Queueing T2I workflow...")
    prompt_id = queue_prompt(workflow)
    
    if not prompt_id:
        print("❌ Failed to queue workflow")
        return
    
    print(f"📋 Prompt ID: {prompt_id}")
    
    # Wait for completion
    print("\n⏳ Generating image...")
    start_time = time.time()
    
    if wait_for_completion(prompt_id):
        duration = time.time() - start_time
        print(f"\n✅ Generation complete in {duration:.1f}s")
        
        # Get output images
        images = get_output_images(prompt_id)
        if images:
            print(f"🖼️ Output images: {images}")
            
            # Save test info
            test_info = {
                "timestamp": datetime.now().isoformat(),
                "category": category,
                "prompt": prompt,
                "animation_prompt": animation,
                "seed": seed,
                "duration_seconds": duration,
                "output_images": images,
                "prompt_id": prompt_id,
            }
            
            output_dir = Path("/home/flip/oelala/generated/sfw_test")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            test_file = output_dir / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(test_file, "w") as f:
                json.dump(test_info, f, indent=2)
            
            print(f"📄 Test info saved to: {test_file}")
            print("\n" + "=" * 60)
            print("✅ TEST SUCCESSFUL")
            print("=" * 60)
            print("\nNext step: Run I2V on this image")
            print(f"Image location: ComfyUI/output/{images[0]}")
        else:
            print("⚠️ No output images found")
    else:
        print("\n❌ Generation timed out")


if __name__ == "__main__":
    main()
