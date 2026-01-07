#!/usr/bin/env python3
"""
Batch SFW Video Generator
Generates 100 diverse SFW videos for the frontpage gallery.

Pipeline:
1. Generate SFW image with DreamShaper XL Lightning
2. Convert to video with Wan 2.2 14B (DisTorch2)
3. Save metadata for upload

Usage:
    python scripts/generate_sfw_batch.py --count 100
    python scripts/generate_sfw_batch.py --count 10 --dry-run
"""

import argparse
import json
import random
import requests
import time
import shutil
from pathlib import Path
from datetime import datetime
from typing import Optional

COMFYUI_URL = "http://localhost:8188"
OUTPUT_DIR = Path("/home/flip/oelala/generated/sfw_batch")
COMFYUI_OUTPUT = Path("/home/flip/oelala/ComfyUI/output")
COMFYUI_INPUT = Path("/home/flip/oelala/ComfyUI/input")

# ============================================================================
# PROMPT LIBRARY - 10 categories × 10 prompts = 100 unique scenes
# ============================================================================

SFW_PROMPTS = {
    "nature": [
        "majestic mountain range at golden hour, snow-capped peaks, dramatic clouds, alpine meadow",
        "serene ocean sunset, waves gently lapping shore, vibrant orange and purple sky, peaceful beach",
        "ancient redwood forest, misty atmosphere, sunbeams through trees, ferns and moss",
        "aurora borealis over frozen lake, green and purple lights dancing, starry night sky",
        "cherry blossom garden in spring, pink petals floating in wind, traditional Japanese bridge",
        "thundering waterfall in tropical rainforest, mist rising, lush green vegetation",
        "vast desert dunes at sunset, golden sand patterns, dramatic shadows, endless horizon",
        "autumn forest with colorful foliage, red and orange leaves, winding path, soft sunlight",
        "crystal clear mountain lake reflection, snow peaks mirrored, pristine wilderness",
        "rolling hills of lavender fields, purple waves, provence style, warm summer light",
    ],
    "animals": [
        "majestic eagle soaring over canyon, wings spread wide, clear blue sky, freedom",
        "pod of dolphins jumping through waves, sparkling ocean water, sunset backdrop",
        "colorful tropical fish in coral reef, underwater sunbeams, vibrant marine life",
        "fox in snowy winter forest, breath visible in cold air, pristine white landscape",
        "butterflies dancing around wildflowers, meadow in summer, soft golden light",
        "hummingbird hovering near flower, iridescent feathers, garden setting, sharp detail",
        "wolves running through snow, pack hunting, misty forest background, dynamic motion",
        "sea turtle gliding through crystal waters, coral reef below, peaceful underwater scene",
        "deer family in misty morning meadow, golden sunrise, peaceful pastoral scene",
        "owl in moonlit forest, wise gaze, ancient tree, mysterious night atmosphere",
    ],
    "urban": [
        "neon-lit Tokyo street at night, rain reflections on pavement, cyberpunk vibes",
        "historic European city square, cobblestone streets, golden hour, cafe culture",
        "futuristic city skyline at dusk, glass towers reflecting sunset, flying vehicles",
        "cozy coffee shop interior, warm lighting, rain on windows, books and plants",
        "grand library with towering bookshelves, reading nooks, soft ambient light",
        "busy street market in Morocco, colorful spices and textiles, lively atmosphere",
        "modern art museum interior, clean lines, dramatic architecture, sculptural lighting",
        "rooftop garden in metropolis, city lights backdrop, peaceful urban oasis",
        "vintage train station, steam locomotive, art deco architecture, nostalgic mood",
        "canal in Venice at twilight, gondolas, historic buildings, romantic atmosphere",
    ],
    "abstract": [
        "flowing liquid metal in slow motion, chrome and gold reflections, mesmerizing swirls",
        "geometric crystal formations, rainbow light refractions, prismatic beauty",
        "ink drops spreading in water, vibrant colors mixing, fluid art, organic patterns",
        "particle explosion in deep space, trails of light and color, cosmic energy",
        "mandala pattern made of pure light, intricate sacred geometry, spiritual art",
        "fractal landscape, infinite complexity, mathematical beauty, surreal colors",
        "aurora-like ribbons of light, flowing energy, ethereal atmosphere",
        "soap bubble macro, iridescent surface, rainbow colors, delicate beauty",
        "sound waves visualized as light, music made visible, rhythmic patterns",
        "magnetic field lines visualization, invisible forces revealed, scientific art",
    ],
    "space": [
        "spiral galaxy with billions of stars, cosmic dust clouds, deep space majesty",
        "ringed planet with multiple moons, nebula background, alien world",
        "asteroid field with distant sun, dramatic lighting, space exploration",
        "nebula nursery with newborn stars, colorful gas clouds, stellar birth",
        "earth from orbit, city lights visible, aurora at poles, pale blue dot",
        "supernova explosion, cosmic shockwave, stellar death and rebirth",
        "binary star system, two suns setting, alien landscape silhouette",
        "space station orbiting earth, technological marvel, humanity in space",
        "comet tail streaming through stars, ice and dust, celestial wanderer",
        "black hole with accretion disk, gravitational lensing, cosmic mystery",
    ],
    "weather": [
        "dramatic thunderstorm over prairie, lightning strikes, dark clouds, raw power",
        "peaceful snowfall in pine forest, fresh powder on trees, winter wonderland",
        "rainbow after storm over green valley, misty atmosphere, hope and renewal",
        "sandstorm approaching ancient ruins, dramatic sky, desert mystery",
        "fog rolling through mountain valley at dawn, layers of hills, ethereal mood",
        "tornado on horizon, storm chaser perspective, power of nature",
        "gentle rain on cherry blossoms, spring shower, romantic atmosphere",
        "sun rays breaking through storm clouds, dramatic crepuscular rays, divine light",
        "frost crystals on window, intricate ice patterns, winter morning",
        "monsoon rain on tropical beach, palm trees bending, dramatic weather",
    ],
    "water": [
        "massive waterfall cascading into jungle pool, mist and rainbows, tropical paradise",
        "crystal clear lake reflecting mountain peaks, perfect mirror, alpine serenity",
        "powerful ocean wave curling, spray and foam, surfer's perspective, raw power",
        "rain drops creating ripples on pond, lily pads floating, zen meditation",
        "underwater cave with light beams, crystal clear water, hidden world",
        "frozen waterfall in winter, ice sculptures, blue and white, nature's art",
        "river rapids through canyon, white water, adventure and energy",
        "bioluminescent waves on night beach, glowing blue, magical phenomenon",
        "koi fish in japanese garden pond, autumn leaves floating, peaceful scene",
        "glacier calving into ocean, massive ice chunks, climate drama",
    ],
    "fire_light": [
        "campfire under milky way, sparks floating up, cozy wilderness night",
        "fireworks over city skyline, celebration, colorful explosions, new year",
        "sun rays through ancient temple windows, dust particles, spiritual light",
        "lantern festival, thousands of paper lanterns rising into night sky, wishes",
        "northern lights reflected in still lake, dancing greens and purples",
        "molten lava flowing into ocean, fire meets water, primal elements",
        "candle-lit cathedral interior, gothic arches, spiritual atmosphere",
        "lightning storm over ocean, electric beauty, nature's fury",
        "sunrise through fog, golden light diffusing, new beginning",
        "fireflies in summer meadow at dusk, magical lights, fairy tale scene",
    ],
    "plants": [
        "field of sunflowers at golden hour, endless rows, blue sky, joyful scene",
        "autumn forest canopy from below, red and orange leaves, sunlight filtering",
        "lotus flower in zen garden, morning dew drops, peaceful meditation",
        "bonsai tree collection, miniature forests, artistic cultivation, patience",
        "wild mushrooms in fairy forest, bioluminescent glow, magical realm",
        "vine-covered ancient ruins, nature reclaiming, mysterious atmosphere",
        "bamboo forest path, tall green stalks, filtered light, zen journey",
        "desert cacti in bloom, unexpected flowers, survival and beauty",
        "moss-covered stones in stream, emerald green, forest atmosphere",
        "giant sequoia trees, ancient giants, human scale for perspective",
    ],
    "technology": [
        "vintage clockwork mechanism, brass gears turning, intricate steampunk detail",
        "sports car on winding mountain road, motion blur, scenic overlook",
        "retro-futuristic control room, analog gauges, warm lighting, space age",
        "steam locomotive crossing bridge, dramatic steam plume, landscape",
        "observatory dome under milky way, telescope pointing at stars, discovery",
        "neon sign reflected in wet street, urban night, cinematic mood",
        "record player with vinyl, warm light, music nostalgia, analog beauty",
        "sailing ship on open ocean, full sails, adventure and exploration",
        "hot air balloons over valley, sunrise, colorful fleet, freedom",
        "lighthouse beam cutting through fog, coastal storm, guiding light",
    ],
}

# Animation prompts - will be randomly combined
ANIMATION_STYLES = [
    "gentle camera pan, smooth motion, cinematic quality",
    "slow zoom in, atmospheric, dreamlike movement",
    "subtle movement, natural sway, peaceful animation",
    "floating particles, ambient motion, serene atmosphere",
    "light rays moving slowly, dynamic shadows, ethereal",
    "gentle ripple effect, calming waves, meditative",
    "slow rotation revealing scene, majestic reveal",
    "drifting clouds or elements, time-lapse style",
    "subtle parallax depth, immersive 3D feel",
    "bokeh lights shifting, cinematic depth of field",
]


def get_prompt(index: int) -> tuple[str, str, str]:
    """Get a specific prompt by index (0-99) for reproducibility."""
    categories = list(SFW_PROMPTS.keys())
    cat_index = index // 10
    prompt_index = index % 10
    
    category = categories[cat_index]
    base_prompt = SFW_PROMPTS[category][prompt_index]
    animation = ANIMATION_STYLES[index % len(ANIMATION_STYLES)]
    
    full_prompt = f"{base_prompt}, masterpiece, highly detailed, professional photography, 8k uhd, cinematic lighting, safe for work"
    
    return category, full_prompt, animation


def create_t2i_workflow(prompt: str, seed: int, prefix: str) -> dict:
    """Create T2I workflow."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "dreamshaperXL_lightningDPMSDE.safetensors"}
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["1", 1], "text": prompt}
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "clip": ["1", 1],
                "text": "ugly, deformed, blurry, low quality, bad anatomy, watermark, text, nsfw, nude, naked, sexual"
            }
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 576, "height": 1024, "batch_size": 1}
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": ["4", 0],
                "seed": seed,
                "steps": 8,
                "cfg": 2.0,
                "sampler_name": "dpmpp_sde",
                "scheduler": "karras",
                "denoise": 1.0
            }
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": ["1", 2]}
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": prefix}
        }
    }


def create_i2v_workflow(image_filename: str, animation_prompt: str, seed: int, prefix: str) -> dict:
    """Load and customize the working I2V workflow with DisTorch2 + LoRAs."""
    import copy
    
    # Load the working workflow template
    workflow_path = Path("/home/flip/oelala/workflows/ImageToVideo/sfw_i2v_distorch2_api.json")
    with open(workflow_path) as f:
        workflow = json.load(f)
    
    # Customize for this generation
    workflow["7"]["inputs"]["text"] = animation_prompt  # Positive prompt
    workflow["10"]["inputs"]["noise_seed"] = seed  # Sampler 1 seed
    workflow["11"]["inputs"]["noise_seed"] = seed  # Sampler 2 seed
    workflow["13"]["inputs"]["filename_prefix"] = prefix  # Output filename
    workflow["18"]["inputs"]["image"] = image_filename  # Input image
    
    return workflow


def queue_and_wait(workflow: dict, timeout: int = 600) -> tuple[bool, Optional[str], list]:
    """Queue workflow and wait for completion."""
    try:
        resp = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow})
        if resp.status_code != 200:
            print(f"❌ Queue failed: {resp.text[:200]}")
            return False, None, []
        
        prompt_id = resp.json().get("prompt_id")
        start = time.time()
        
        while time.time() - start < timeout:
            time.sleep(5)
            hist = requests.get(f"{COMFYUI_URL}/history/{prompt_id}").json()
            
            if prompt_id in hist:
                outputs = hist[prompt_id].get("outputs", {})
                files = []
                
                for node_id, out in outputs.items():
                    if "images" in out:
                        files.extend([img.get("filename") for img in out["images"]])
                    if "gifs" in out:
                        files.extend([g.get("filename") for g in out["gifs"]])
                
                return True, prompt_id, files
        
        return False, prompt_id, []
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, None, []


def generate_single(index: int, dry_run: bool = False) -> dict:
    """Generate a single T2I + I2V pipeline."""
    category, prompt, animation = get_prompt(index)
    seed = 1000000 + index  # Reproducible seeds
    
    result = {
        "index": index,
        "category": category,
        "prompt": prompt,
        "animation": animation,
        "seed": seed,
        "t2i_image": None,
        "i2v_video": None,
        "t2i_time": 0,
        "i2v_time": 0,
        "success": False,
    }
    
    prefix = f"sfw_batch_{index:04d}"
    
    if dry_run:
        print(f"  [DRY RUN] Would generate: {category} - {prompt[:50]}...")
        result["success"] = True
        return result
    
    # Step 1: T2I
    print(f"  📸 T2I: {category}...", end=" ", flush=True)
    t2i_start = time.time()
    t2i_workflow = create_t2i_workflow(prompt, seed, f"{prefix}_t2i")
    success, _, files = queue_and_wait(t2i_workflow, timeout=120)
    
    if not success or not files:
        print("❌")
        return result
    
    result["t2i_time"] = time.time() - t2i_start
    result["t2i_image"] = files[0]
    print(f"✅ ({result['t2i_time']:.0f}s)")
    
    # Copy image to input folder
    src = COMFYUI_OUTPUT / files[0]
    dst = COMFYUI_INPUT / files[0]
    shutil.copy(src, dst)
    
    # Step 2: I2V
    print(f"  🎬 I2V: animating...", end=" ", flush=True)
    i2v_start = time.time()
    i2v_workflow = create_i2v_workflow(files[0], animation, seed, f"{prefix}_i2v")
    success, _, files = queue_and_wait(i2v_workflow, timeout=600)
    
    if not success or not files:
        print("❌")
        return result
    
    result["i2v_time"] = time.time() - i2v_start
    result["i2v_video"] = files[0]
    result["success"] = True
    print(f"✅ ({result['i2v_time']:.0f}s)")
    
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate SFW video batch")
    parser.add_argument("--count", type=int, default=100, help="Number of videos")
    parser.add_argument("--start", type=int, default=0, help="Starting index")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually generate")
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎬 SFW Batch Video Generator")
    print("=" * 60)
    print(f"Generating {args.count} videos starting at index {args.start}")
    print(f"Estimated time: {args.count * 3} minutes ({args.count * 3 / 60:.1f} hours)")
    print()
    
    if not args.dry_run:
        # Check ComfyUI
        try:
            resp = requests.get(f"{COMFYUI_URL}/system_stats")
            if resp.status_code != 200:
                print("❌ ComfyUI not responding")
                return
        except:
            print("❌ Cannot connect to ComfyUI")
            return
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results = []
    start_time = time.time()
    
    for i in range(args.start, args.start + args.count):
        print(f"\n[{i+1-args.start}/{args.count}] Generating video {i:04d}")
        
        result = generate_single(i, args.dry_run)
        results.append(result)
        
        if result["success"]:
            print(f"  ✅ Total: {result['t2i_time'] + result['i2v_time']:.0f}s")
        else:
            print(f"  ❌ Failed")
    
    # Summary
    elapsed = time.time() - start_time
    successes = sum(1 for r in results if r["success"])
    
    print("\n" + "=" * 60)
    print("📊 BATCH COMPLETE")
    print("=" * 60)
    print(f"Total time: {elapsed/60:.1f} minutes")
    print(f"Success: {successes}/{len(results)}")
    print(f"Average per video: {elapsed/max(len(results),1):.0f}s")
    
    # Save results
    results_file = OUTPUT_DIR / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "count": args.count,
            "start": args.start,
            "elapsed_seconds": elapsed,
            "successes": successes,
            "results": results,
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
