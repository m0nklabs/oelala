# Wan2.2 Image-to-Video Generator Demo
# Oelala Project - AI Video Generation Pipeline

#!/usr/bin/env python3
"""
Simple demo script for Wan2.2 Image-to-Video generation
"""

import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from wan2_generator import Wan2VideoGenerator

def demo_wan2_generation():
    """Demo function showing how to use Wan2.2 generator"""

    print("🎬 Wan2.2 Image-to-Video Demo")
    print("=" * 50)

    # Initialize generator
    generator = Wan2VideoGenerator()

    # Load model
    print("Loading Wan2.2 model...")
    if not generator.load_model():
        print("❌ Failed to load model")
        return

    # Check for sample image
    sample_images = ["sample.jpg", "person.jpg", "test_image.jpg"]

    image_path = None
    for img in sample_images:
        if os.path.exists(img):
            image_path = img
            break

    if image_path is None:
        print("⚠️ No sample image found. Creating a placeholder...")
        print("💡 To test with a real image:")
        print("1. Place an image file (person.jpg, sample.jpg, etc.) in this directory")
        print("2. Or modify the image_path variable below")
        print("3. Run this script again")

        # Create a simple colored image as placeholder
        from PIL import Image
        import numpy as np

        # Create a 512x512 colored image
        img_array = np.zeros((512, 512, 3), dtype=np.uint8)
        img_array[:, :256] = [255, 0, 0]  # Red left half
        img_array[:, 256:] = [0, 255, 0]  # Green right half

        placeholder = Image.fromarray(img_array)
        placeholder.save("placeholder_image.jpg")
        image_path = "placeholder_image.jpg"
        print(f"📸 Created placeholder image: {image_path}")

    # Generate video
    prompt = "A person dancing gracefully with smooth movements"
    output_path = "wan2_demo_output.mp4"

    print(f"🎭 Generating video from: {image_path}")
    print(f"📝 Prompt: {prompt}")

    result = generator.generate_video_from_image(
        image_path=image_path,
        prompt=prompt,
        output_path=output_path,
        num_frames=16
    )

    if result:
        print(f"✅ Demo completed! Video saved to: {result}")
        print("🎉 Wan2.2 Image-to-Video generation is working!")
    else:
        print("❌ Demo failed. Check the error messages above.")

if __name__ == "__main__":
    demo_wan2_generation()
