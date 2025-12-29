#!/usr/bin/env python3
"""
Gradio UI for WAN2.2 Video Generation
Simple interface for testing text-to-video and image-to-video
"""

import gradio as gr
import requests
import os
from pathlib import Path

# Backend URL
BACKEND_URL = "http://127.0.0.1:7995"

def generate_text_to_video(prompt, num_frames, model_type):
    """Generate video from text prompt"""
    try:
        # Prepare form data
        data = {
            "prompt": prompt,
            "num_frames": num_frames,
            "model_type": model_type
        }
        
        # Call backend API
        response = requests.post(
            f"{BACKEND_URL}/generate-text",
            files=data,
            timeout=300  # 5 minutes timeout
        )
        
        if response.status_code == 200:
            result = response.json()
            video_path = result.get("video_path")
            
            if video_path and os.path.exists(video_path):
                return video_path, f"✅ Video generated: {result.get('message', 'Success')}"
            else:
                return None, f"❌ Video file not found: {video_path}"
        else:
            return None, f"❌ Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return None, f"❌ Exception: {str(e)}"

def generate_image_to_video(image, prompt, num_frames):
    """Generate video from image + prompt"""
    try:
        if image is None:
            return None, "❌ Please upload an image"
        
        # Save uploaded image temporarily
        temp_image_path = "/tmp/gradio_upload.png"
        image.save(temp_image_path)
        
        # Prepare multipart form data
        with open(temp_image_path, 'rb') as f:
            files = {'image': ('image.png', f, 'image/png')}
            data = {
                'prompt': prompt,
                'num_frames': str(num_frames)
            }
            
            # Call backend API
            response = requests.post(
                f"{BACKEND_URL}/generate",
                files=files,
                data=data,
                timeout=300  # 5 minutes timeout
            )
        
        if response.status_code == 200:
            result = response.json()
            video_path = result.get("video_path")
            
            if video_path and os.path.exists(video_path):
                return video_path, f"✅ Video generated: {result.get('message', 'Success')}"
            else:
                return None, f"❌ Video file not found: {video_path}"
        else:
            return None, f"❌ Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return None, f"❌ Exception: {str(e)}"

def check_backend_status():
    """Check if backend is running"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return f"✅ Backend: {data.get('status', 'unknown')}\n📍 Model loaded: {data.get('model_loaded', False)}"
        else:
            return f"⚠️ Backend responded with status {response.status_code}"
    except Exception as e:
        return f"❌ Backend not reachable: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="WAN2.2 Video Generator") as demo:
    gr.Markdown("""
    # 🎬 WAN2.2 Video Generator
    Generate videos from text prompts or images using AI
    """)
    
    # Backend status
    status_text = gr.Textbox(label="Backend Status", value=check_backend_status(), interactive=False)
    refresh_btn = gr.Button("🔄 Refresh Status")
    refresh_btn.click(fn=check_backend_status, outputs=status_text)
    
    # Text-to-Video tab
    with gr.Tab("📝 Text-to-Video"):
        gr.Markdown("### Generate video from text description")
        
        t2v_prompt = gr.Textbox(
            label="Prompt",
            placeholder="Describe the video you want to generate (e.g., 'a cat running in a field')",
            lines=3
        )
        
        with gr.Row():
            t2v_frames = gr.Slider(
                label="Number of Frames",
                minimum=8,
                maximum=32,
                value=16,
                step=1
            )
            t2v_model = gr.Dropdown(
                label="Model Type",
                choices=["light", "wan2.2", "svd"],
                value="light"
            )
        
        t2v_generate_btn = gr.Button("🎬 Generate Video", variant="primary")
        
        t2v_output_video = gr.Video(label="Generated Video")
        t2v_output_text = gr.Textbox(label="Status")
        
        t2v_generate_btn.click(
            fn=generate_text_to_video,
            inputs=[t2v_prompt, t2v_frames, t2v_model],
            outputs=[t2v_output_video, t2v_output_text]
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["a cat running in a field", 16, "light"],
                ["astronaut riding a horse on mars", 16, "light"],
                ["waves crashing on a beach at sunset", 16, "light"],
                ["a robot dancing in a futuristic city", 16, "light"],
            ],
            inputs=[t2v_prompt, t2v_frames, t2v_model],
        )
    
    # Image-to-Video tab
    with gr.Tab("🖼️ Image-to-Video"):
        gr.Markdown("### Generate video from image + text description")
        
        i2v_image = gr.Image(
            label="Input Image",
            type="pil",
            sources=["upload", "clipboard"]
        )
        
        i2v_prompt = gr.Textbox(
            label="Motion Prompt",
            placeholder="Describe the motion/animation (e.g., 'person walking forward', 'zoom in slowly')",
            lines=2
        )
        
        i2v_frames = gr.Slider(
            label="Number of Frames",
            minimum=8,
            maximum=32,
            value=16,
            step=1
        )
        
        i2v_generate_btn = gr.Button("🎬 Generate Video", variant="primary")
        
        i2v_output_video = gr.Video(label="Generated Video")
        i2v_output_text = gr.Textbox(label="Status")
        
        i2v_generate_btn.click(
            fn=generate_image_to_video,
            inputs=[i2v_image, i2v_prompt, i2v_frames],
            outputs=[i2v_output_video, i2v_output_text]
        )
    
    # Info footer
    gr.Markdown("""
    ---
    **💡 Tips:**
    - Text-to-Video uses lightweight model (fast, ~3.7GB VRAM)
    - Image-to-Video uses WAN2.2 I2V model (slower, ~15GB VRAM, lazy loaded)
    - First generation may be slow due to model loading
    - Backend runs on: http://127.0.0.1:7995
    """)

if __name__ == "__main__":
    print("🚀 Starting Gradio UI for WAN2.2 Video Generator...")
    print(f"📡 Backend URL: {BACKEND_URL}")
    print(f"🌐 Frontend will be available at: http://127.0.0.1:7860")
    print(f"📱 Or access from network: http://192.168.1.2:7860")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
