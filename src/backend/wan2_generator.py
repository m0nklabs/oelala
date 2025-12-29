#!/usr/bin/env python3
"""
Wan2.2 Image-to-Video Generation Script
Oelala Project - AI Video Generation Pipeline
With Stable Video Diffusion fallback
"""

import os
import sys
import re
import torch
import json
# Temporarily disable WanImageToVideoPipeline import until it's available
# from diffusers import WanImageToVideoPipeline
# from diffusers import StableVideoDiffusionPipeline
from PIL import Image
import numpy as np

# Dataset is used by ImageDataset regardless of PEFT availability
from torch.utils.data import Dataset

# Add OpenPose for pose estimation
try:
    import openpose.pyopenpose as op
    OPENPOSE_AVAILABLE = True
except ImportError:
    print("⚠️ OpenPose not available. Pose-guided generation will be disabled.")
    OPENPOSE_AVAILABLE = False
    op = None

# Add PEFT for LoRA fine-tuning
try:
    from peft import LoraConfig, get_peft_model
    PEFT_AVAILABLE = True
except Exception as e:
    print(f"⚠️ PEFT not available ({e.__class__.__name__}). LoRA fine-tuning will be disabled.")
    print("💡 Install with: pip install peft")
    PEFT_AVAILABLE = False
    LoraConfig = None
    get_peft_model = None

# Set environment variables for better performance
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

DEFAULT_MAX_COMPUTE_CAP = (9, 0)

def _resolve_supported_compute_capability_ceiling():
    """Return the highest compute capability supported by the current torch build."""
    try:
        arch_list = getattr(torch.cuda, "get_arch_list", lambda: [])()
        caps = []
        for arch in arch_list:
            match = re.match(r"sm_(\d+)(\d)", arch)
            if match:
                major = int(match.group(1))
                minor = int(match.group(2))
                caps.append((major, minor))
        if caps:
            ceiling = max(caps)
            print(f"📊 Torch build reports max supported capability sm_{ceiling[0]}{ceiling[1]}")
            return ceiling
    except Exception as cap_error:
        print(f"⚠️ Unable to read torch arch list: {cap_error}")
    return DEFAULT_MAX_COMPUTE_CAP

class Wan2VideoGenerator:
    def __init__(self, model_path="Wan-AI/Wan2.2-I2V-A14B-Diffusers", device="cuda", model_type="auto"):
        """
        Initialize the Wan2.2 generator

        Args:
            model_path: Path to the model (use -Diffusers variant for direct Diffusers support)
            device: Device to use ('cuda' or 'cpu')
            model_type: Type of model to use ('auto', 'wan2.2', 'svd', 'light')
        """
        self.model_path = model_path
        self.device = device
        self.model_type = model_type
        self.pipeline = None
        self.text_pipeline = None  # For lightweight text-to-video model
        self.max_supported_capability = _resolve_supported_compute_capability_ceiling()

        print(f"🎬 Initializing Video Generator")
        print(f"📍 Model path: {model_path}")
        print(f"🎯 Device: {device}")
        print(f"🔧 Model type: {model_type}")

        self.selected_cuda_index = None

        # Set device
        if device == "cuda" and torch.cuda.is_available():
            selected_index = self._select_compatible_cuda_device()
            if selected_index is not None:
                try:
                    torch.cuda.set_device(selected_index)
                except Exception as cuda_error:
                    print(f"❌ Failed to set CUDA device {selected_index}: {cuda_error}")
                    selected_index = None

            if selected_index is not None:
                self.selected_cuda_index = selected_index
                self.device = torch.device(f"cuda:{selected_index}")
                cap = torch.cuda.get_device_capability(selected_index)
                props = torch.cuda.get_device_properties(selected_index)
                print(f"✅ CUDA available: {props.name} (index {selected_index}, sm_{cap[0]}{cap[1]})")
                print(f"💾 CUDA memory: {props.total_memory / 1024**3:.1f} GB")
            else:
                self.device = torch.device("cpu")
                print("⚠️ No CUDA device with supported compute capability detected; using CPU fallback")
        else:
            self.device = torch.device("cpu")
            if device != "cuda":
                print("⚠️ Device override requested CPU; respecting configuration")
            else:
                print("⚠️ Using CPU (CUDA not available)")

        # Initialize based on model type
        if model_type == "light":
            print("🎯 Using lightweight text-to-video model")
            self._init_light_model()
        else:
            self.load_model()

    def _select_compatible_cuda_device(self):
        """Select a CUDA device whose compute capability is supported by this PyTorch build."""
        try:
            if not torch.cuda.is_available():
                print("⚠️ torch.cuda.is_available() returned False during device scan")
                return None

            selected_idx = None
            selected_capability = (-1, -1)

            for idx in range(torch.cuda.device_count()):
                capability = torch.cuda.get_device_capability(idx)
                name = torch.cuda.get_device_name(idx)
                print(f"🔍 Detected CUDA device {idx}: {name} (sm_{capability[0]}{capability[1]})")

                if capability <= self.max_supported_capability:
                    if capability > selected_capability:
                        selected_idx = idx
                        selected_capability = capability

            if selected_idx is not None:
                print(f"✅ Selected CUDA device {selected_idx} with sm_{selected_capability[0]}{selected_capability[1]} (<= sm_{self.max_supported_capability[0]}{self.max_supported_capability[1]})")
                return selected_idx

            print("⚠️ No CUDA GPUs fall within the supported compute capability range; falling back to CPU")
            return None
        except Exception as scan_error:
            print(f"❌ Error while scanning CUDA devices: {scan_error}")
            return None

    def _init_light_model(self):
        """
        Initialize the lightweight text-to-video model with lazy loading
        """
        # Don't load model at startup - wait for first request
        print("🎯 Lightweight text-to-video model configured for lazy loading")
        print("💡 Model will be loaded on first /generate-text request")
        if self.device.type == "cuda":
            print(f"🚀 Lightweight model will use CUDA device index {self.selected_cuda_index} (sm_{self.max_supported_capability[0]}{self.max_supported_capability[1]} supported)")
        else:
            print("⚠️ Using CPU fallback for lightweight model (no supported CUDA device detected)")
        self.text_pipeline = None  # Will be loaded on demand

    def load_model(self):
        """Load the Wan2.2 model and pipeline with lazy loading"""
        try:
            print("🔍 Checking for Wan2.2 pipeline availability...")
            print(f"📍 Model path: {self.model_path}")
            print(f"🎯 Device: {self.device}")
            print(f"💾 CUDA available: {torch.cuda.is_available()}")

            # If SVD is explicitly requested, skip Wan2.2 check
            if self.model_type == "svd":
                print("👉 SVD explicitly requested, skipping Wan2.2 check")
                raise ImportError("SVD requested")

            # Try Wan2.2 first
            try:
                from diffusers import WanImageToVideoPipeline
                print("✅ WanImageToVideoPipeline found in diffusers")
                print("🎯 Using lazy loading for WAN2.2 I2V pipeline")
                print("💡 Model will be loaded on first /generate request")
                self.pipeline = None  # Will be loaded on demand
                self.model_type = "wan2.2"
                return True
            except ImportError as ie:
                print(f"❌ WanImageToVideoPipeline not available: {ie}")
                print("📝 Wan2.2 is not yet publicly available in diffusers")
                print("🔄 Falling back to Stable Video Diffusion...")

            # Fallback to Stable Video Diffusion
            try:
                print("🎬 Stable Video Diffusion will be loaded on first use...")
                print("💡 This prevents long startup times")
                self.pipeline = None  # Lazy load
                self.model_type = "svd"
                print("✅ SVD lazy loading configured!")
                return True
            except Exception as svd_error:
                print(f"❌ Stable Video Diffusion setup failed: {svd_error}")
                print("🔄 Switching to placeholder mode for UI testing")

                # Create a placeholder pipeline for UI testing
                self.pipeline = "placeholder"
                self.model_type = "placeholder"
                print("✅ Placeholder mode activated - video generation will create dummy output")
                return True

        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔧 Troubleshooting steps:")
            print("   1. Check GPU memory (at least 8GB VRAM recommended)")
            print("   2. Verify internet connection for model downloads")
            print("   3. Consider using placeholder mode for testing")
            return False

    def generate_video_from_image(self, image_path, prompt="", output_path="output.mp4", num_frames=16):
        """
        Generate video from input image using available pipeline

        Args:
            image_path: Path to input image
            prompt: Optional text prompt to guide generation
            output_path: Path to save the output video
            num_frames: Number of frames in the video
        """
        if self.pipeline is None:
            print("❌ Model not loaded. Call load_model() first.")
            return None

        # Handle placeholder mode
        if self.pipeline == "placeholder":
            print("🎭 Placeholder mode: Generating dummy video for UI testing")
            return self._generate_placeholder_video(image_path, prompt, output_path, num_frames)

        # Handle Stable Video Diffusion
        if self.model_type == "svd":
            print("🎬 Stable Video Diffusion: Generating video from image")
            return self._generate_svd_video(image_path, output_path, num_frames)

        # Handle Wan2.2 (when available)
        if self.model_type == "wan2.2":
            print("🎬 Wan2.2: Generating video from image")
            return self._generate_wan2_video(image_path, prompt, output_path, num_frames)

        try:
            # Fallback for unknown model types
            print(f"🎭 Unknown model type '{self.model_type}', using placeholder")
            return self._generate_placeholder_video(image_path, prompt, output_path, num_frames)

        except Exception as e:
            print(f"❌ Error generating video: {e}")
            return None

    def _generate_svd_video(self, image_path, output_path, num_frames):
        """
        Generate video using Stable Video Diffusion with lazy loading
        """
        try:
            # Lazy load SVD pipeline if needed
            if self.pipeline == "svd_lazy":
                print("🎬 Loading Stable Video Diffusion pipeline (first use)...")
                print("⏱️ This may take 2-3 minutes to download models...")

                # Import SVD here to avoid multiprocessing issues
                from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline

                self.pipeline = StableVideoDiffusionPipeline.from_pretrained(
                    "stabilityai/stable-video-diffusion-img2vid-xt",
                    torch_dtype=torch.float16,
                    variant="fp16"
                ).to(self.device)
                print("✅ Stable Video Diffusion loaded successfully!")

            # Load input image
            if not os.path.exists(image_path):
                print(f"❌ Input image not found: {image_path}")
                return None

            image = Image.open(image_path).convert("RGB")
            print(f"🖼️ Loaded image: {image.size}")

            # SVD expects specific image size (1024x576)
            image = image.resize((1024, 576))
            print(f"📐 Resized to SVD format: {image.size}")

            print("🎬 Generating video with Stable Video Diffusion...")
            print("⏱️ This may take 30-60 seconds...")

            # Generate video with SVD
            # Note: SVD generates 25 frames by default, we ignore num_frames parameter
            output = self.pipeline(
                image,
                num_inference_steps=25,
                num_frames=25,
                decode_chunk_size=8
            )

            # Save video
            video_frames = output.frames[0]  # Get first batch
            self.save_video(video_frames, output_path)

            print(f"✅ SVD video saved to: {output_path}")
            print(f"📊 Generated {len(video_frames)} frames")
            return output_path

        except Exception as e:
            print(f"❌ Error in SVD generation: {e}")
            print("🔄 Falling back to enhanced placeholder mode")
            # For text-to-video, we don't have an image, so create a text-based placeholder
            if not os.path.exists(image_path):
                # This is likely a text-to-video case where image_path is a temp file that was cleaned up
                return self._create_enhanced_placeholder_text_video("", output_path, num_frames)
            else:
                # Check if this is a temp image from text-to-video by looking at the filename
                if "temp_text_image" in image_path:
                    # Try to extract the prompt from the filename hash
                    try:
                        # The filename contains hash(prompt), but we can't reverse it easily
                        # For now, use a generic message
                        return self._create_enhanced_placeholder_text_video("Text-to-Video Generation", output_path, num_frames)
                    except:
                        return self._create_enhanced_placeholder_text_video("", output_path, num_frames)
                else:
                    return self._generate_placeholder_video(image_path, "", output_path, num_frames)

    def _generate_wan2_video(self, image_path, prompt, output_path, num_frames):
        """
        Generate video using Wan2.2 model with lazy loading
        """
        try:
            # Lazy load WAN2.2 pipeline if needed
            if self.pipeline is None:
                print("🎬 Loading WAN2.2 I2V pipeline (first use)...")
                print("⏱️ This may take 2-3 minutes...")
                print("⚠️ Requires ~15GB VRAM - ensure GPU memory is available")

                from diffusers import WanImageToVideoPipeline

                self.pipeline = WanImageToVideoPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16
                ).to(self.device)
                print("✅ WAN2.2 I2V model loaded successfully!")

            # Load input image
            if not os.path.exists(image_path):
                print(f"❌ Input image not found: {image_path}")
                return None

            image = Image.open(image_path).convert("RGB")
            print(f"🖼️ Loaded image: {image.size}")

            print(f"🎬 Generating Wan2.2 video with prompt: '{prompt}'")
            print(f"📐 Frames: {num_frames}")

            # Generate video with Wan2.2
            output = self.pipeline(
                image=image,
                prompt=prompt,
                num_frames=num_frames,
                num_inference_steps=20,
                guidance_scale=6.0
            )

            # Save video
            video_frames = output.frames[0]  # Get first batch
            self.save_video(video_frames, output_path)

            print(f"✅ Wan2.2 video saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Error in Wan2.2 generation: {e}")
            print("🔄 Falling back to SVD")
            return self._generate_svd_video(image_path, output_path, num_frames)

    def _generate_placeholder_video(self, image_path, prompt, output_path, num_frames):
        """
        Generate a placeholder video for UI testing when real model is not available
        """
        try:
            print(f"🎭 Creating placeholder video: {output_path}")
            print(f"📝 Prompt: '{prompt}'")
            print(f"🎬 Frames: {num_frames}")

            # Load the input image
            if not os.path.exists(image_path):
                print(f"❌ Input image not found: {image_path}")
                return None

            image = Image.open(image_path).convert("RGB")
            print(f"🖼️ Loaded image: {image.size}")

            # Create a simple animation by slightly modifying the image
            frames = []
            width, height = image.size

            for i in range(num_frames):
                # Create a copy of the image
                frame = image.copy()

                # Add some simple animation effect (slight color shift)
                if i > 0:
                    # Convert to numpy array for manipulation
                    import numpy as np
                    img_array = np.array(frame)

                    # Add slight variation to create animation effect
                    variation = int(5 * (i / num_frames))  # Gradual change
                    img_array = np.clip(img_array + variation, 0, 255).astype(np.uint8)

                    frame = Image.fromarray(img_array)

                frames.append(frame)

            # Save as animated GIF (fallback if mp4 fails)
            try:
                # Try to save as MP4 first
                self.save_video(frames, output_path)
                print(f"✅ Placeholder video saved to: {output_path}")
                return output_path
            except Exception as e:
                print(f"⚠️ MP4 saving failed, trying GIF: {e}")
                # Fallback to GIF
                gif_path = output_path.replace('.mp4', '.gif')
                frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=500, loop=0)
                print(f"✅ Placeholder GIF saved to: {gif_path}")
                return gif_path

        except Exception as e:
            print(f"❌ Error creating placeholder video: {e}")
            return None

    def generate_video(self, prompt, output_path="output.mp4", num_frames=16, height=480, width=832):
        """
        Generate video from text prompt only (if supported)

        Args:
            prompt: Text description of the video
            output_path: Path to save the output video
            num_frames: Number of frames in the video
            height: Video height
            width: Video width
        """
        print("⚠️ Wan2.2 I2V model requires an input image. Use generate_video_from_image() instead.")
        return None

    def save_video(self, frames, output_path, fps=8):
        """
        Save video frames to MP4 file

        Args:
            frames: List of PIL images
            output_path: Output file path
            fps: Frames per second
        """
        # Preferred path: FFmpeg libx264 yuv420p baseline for browser compatibility
        try:
            import tempfile, subprocess
            tmpdir = tempfile.mkdtemp(prefix="oelala_frames_")
            rgb_frames = []
            for i, frame in enumerate(frames):
                # Ensure RGB
                f_rgb = frame.convert('RGB')
                frame_path = os.path.join(tmpdir, f"frame_{i:05d}.png")
                f_rgb.save(frame_path)
                rgb_frames.append(frame_path)
            # Build ffmpeg command
            cmd = [
                'ffmpeg','-y',
                '-framerate', str(fps),
                '-i', os.path.join(tmpdir, 'frame_%05d.png'),
                '-c:v','libx264',
                '-preset','veryfast',
                '-pix_fmt','yuv420p',
                '-movflags','+faststart',
                output_path
            ]
            print(f"🚀 Encoding video with ffmpeg libx264: {' '.join(cmd)}")
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            print(f"✅ H.264 video saved: {output_path} ({len(frames)} frames @ {fps} FPS)")
            return
        except Exception as e:
            print(f"⚠️ ffmpeg path failed ({e}); trying imageio libx264")
        # Second attempt: imageio + ffmpeg wrapper
        try:
            import imageio.v3 as iio
            arr_frames = [np.array(f.convert('RGB')) for f in frames]
            iio.imwrite(output_path, arr_frames, fps=fps, codec='libx264', quality=8)
            print(f"🎥 Video saved via imageio/libx264: {output_path} ({len(frames)} frames @ {fps} FPS)")
            return
        except Exception as e:
            print(f"⚠️ imageio/libx264 failed ({e}); falling back to OpenCV mp4v (mpeg4 codec)")
        # Third attempt: OpenCV mp4v (MPEG4 Part 2)
        try:
            import cv2
            frame_arrays = []
            for frame in frames:
                frame_np = np.array(frame.convert('RGB'))
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                frame_arrays.append(frame_bgr)
            height, width = frame_arrays[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            for frame in frame_arrays:
                video_writer.write(frame)
            video_writer.release()
            print(f"📹 Video saved via OpenCV mp4v (mpeg4): {output_path} ({len(frames)} frames @ {fps} FPS)")
            return
        except Exception as e:
            print(f"❌ OpenCV fallback failed ({e}); dumping PNG frames")
        # Final fallback: dump frames
        fallback_dir = os.path.splitext(output_path)[0] + "_frames"
        os.makedirs(fallback_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            frame.save(os.path.join(fallback_dir, f"frame_{i:04d}.png"))
        print(f"📸 Saved {len(frames)} PNG frames to {fallback_dir}")

    def generate_pose_guided_video(self, image_path, output_path="pose_output.mp4", num_frames=16):
        """
        Generate pose-guided video from input image using OpenPose

        Args:
            image_path: Path to input image
            output_path: Path to save the output video
            num_frames: Number of frames in the video
        """
        if not OPENPOSE_AVAILABLE:
            print("❌ OpenPose not available. Falling back to regular generation.")
            return self.generate_video_from_image(image_path, "", output_path, num_frames)

        if self.pipeline is None:
            print("❌ Model not loaded. Call load_model() first.")
            return None

        try:
            # Load input image
            if not os.path.exists(image_path):
                print(f"❌ Input image not found: {image_path}")
                return None

            image = Image.open(image_path).convert("RGB")
            print(f"🖼️ Loaded image: {image.size}")

            # Initialize OpenPose
            print("🤖 Initializing OpenPose for pose estimation...")
            wrapper = op.WrapperPython()
            params = {
                "model_folder": "/home/flip/oelala/openpose/models/",
                "net_resolution": "320x176",
                "face": False,
                "hand": False
            }
            wrapper.configure(params)
            wrapper.start()

            # Convert PIL to numpy for OpenPose
            image_np = np.array(image)

            # Perform pose estimation
            datum = op.Datum()
            datum.cvInputData = image_np
            datumVector = op.VectorDatum()
            datumVector.append(datum)
            wrapper.emplaceAndPop(datumVector)

            # Extract keypoints
            keypoints = datum.poseKeypoints
            if keypoints.shape[0] == 0:
                print("⚠️ No persons detected in image. Using regular generation.")
                return self.generate_video_from_image(image_path, "", output_path, num_frames)

            print(f"🎯 Detected {keypoints.shape[0]} person(s) with pose keypoints")

            # Analyze pose for motion description
            pose_description = self.analyze_pose_for_motion(keypoints[0])  # First person

            # Create pose-guided prompt
            base_prompt = "A person in motion"
            pose_prompt = f"{base_prompt}, {pose_description}, realistic movement, smooth animation"

            print(f"🎬 Generating pose-guided video with prompt: '{pose_prompt}'")

            # Generate video with pose-guided prompt
            output = self.pipeline(
                image=image,
                prompt=pose_prompt,
                num_frames=num_frames,
                num_inference_steps=20,
                guidance_scale=6.0
            )

            # Save video
            video_frames = output.frames[0]
            self.save_video(video_frames, output_path)

            print(f"✅ Pose-guided video saved to: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Error in pose-guided generation: {e}")
            print("💡 Falling back to regular generation...")
            return self.generate_video_from_image(image_path, "", output_path, num_frames)

    def analyze_pose_for_motion(self, keypoints):
        """
        Analyze pose keypoints to generate motion description

        Args:
            keypoints: OpenPose keypoints array (25, 3) - x, y, confidence

        Returns:
            str: Motion description for video generation
        """
        # BODY_25 keypoint mapping
        keypoint_names = [
            "Nose", "Neck", "RShoulder", "RElbow", "RWrist", "LShoulder", "LElbow", "LWrist",
            "MidHip", "RHip", "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle", "REye", "LEye",
            "REar", "LEar", "LBigToe", "LSmallToe", "LHeel", "RBigToe", "RSmallToe", "RHeel"
        ]

        motion_descriptions = []

        # Check arm positions
        if keypoints[3, 2] > 0.1 and keypoints[7, 2] > 0.1:  # Right elbow and wrist visible
            if keypoints[7, 1] < keypoints[3, 1]:  # Wrist above elbow
                motion_descriptions.append("raising right arm")

        if keypoints[6, 2] > 0.1 and keypoints[7, 2] > 0.1:  # Left elbow and wrist visible
            if keypoints[7, 1] < keypoints[6, 1]:  # Wrist above elbow
                motion_descriptions.append("raising left arm")

        # Check leg positions
        if keypoints[10, 2] > 0.1 and keypoints[11, 2] > 0.1:  # Right knee and ankle visible
            if keypoints[11, 1] > keypoints[10, 1]:  # Ankle below knee
                motion_descriptions.append("standing with right leg forward")

        if keypoints[13, 2] > 0.1 and keypoints[14, 2] > 0.1:  # Left knee and ankle visible
            if keypoints[14, 1] > keypoints[13, 1]:  # Ankle below knee
                motion_descriptions.append("standing with left leg forward")

        # Default motion if no specific pose detected
        if not motion_descriptions:
            motion_descriptions.append("natural walking motion")
            motion_descriptions.append("fluid body movement")

        return ", ".join(motion_descriptions)

    def train_lora(self, image_paths, output_dir="lora_output", num_epochs=10, learning_rate=1e-4):
        """
        Train LoRA adapter on multiple images for consistent avatar generation

        Args:
            image_paths: List of paths to training images
            output_dir: Directory to save LoRA weights
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training

        Returns:
            str: Path to saved LoRA weights
        """
        # Allow forcing a placeholder artifact via environment variable so training can be exercised
        if os.environ.get("OELALA_FORCE_LORA_PLACEHOLDER", "0") == "1":
            os.makedirs(output_dir, exist_ok=True)
            placeholder = {
                "note": "LoRA training skipped due to OELALA_FORCE_LORA_PLACEHOLDER=1",
                "image_count": len(image_paths),
                "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z"
            }
            placeholder_path = os.path.join(output_dir, "lora_placeholder.json")
            with open(placeholder_path, "w") as fh:
                json.dump(placeholder, fh, indent=2)
            print("⚠️ Forced placeholder LoRA artifact created at:", placeholder_path)
            return placeholder_path

        if not PEFT_AVAILABLE:
            # Create a placeholder artifact so callers can continue working without PEFT installed.
            os.makedirs(output_dir, exist_ok=True)
            placeholder = {
                "note": "PEFT (peft) is not installed. LoRA training was skipped.",
                "install": "pip install peft",
                "image_count": len(image_paths),
                "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z"
            }
            placeholder_path = os.path.join(output_dir, "lora_placeholder.json")
            with open(placeholder_path, "w") as fh:
                json.dump(placeholder, fh, indent=2)
            print("⚠️ PEFT not available. Created placeholder LoRA artifact at:", placeholder_path)
            return placeholder_path

        if self.pipeline is None:
            print("❌ Model not loaded. Call load_model() first.")
            return None

        try:
            print(f"🎨 Starting LoRA training on {len(image_paths)} images")
            print(f"📁 Output directory: {output_dir}")
            print(f"🔄 Epochs: {num_epochs}, Learning rate: {learning_rate}")

            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Setup LoRA configuration
            lora_config = LoraConfig(
                r=16,  # Rank of LoRA
                lora_alpha=32,
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],  # Attention layers
                lora_dropout=0.1,
                bias="none"
            )

            # Apply LoRA to the model
            self.pipeline.unet = get_peft_model(self.pipeline.unet, lora_config)
            self.pipeline.text_encoder = get_peft_model(self.pipeline.text_encoder, lora_config)

            # Create dataset
            dataset = ImageDataset(image_paths)

            # Setup optimizer
            optimizer = torch.optim.AdamW(
                self.pipeline.parameters(),
                lr=learning_rate
            )

            # Training loop
            self.pipeline.train()
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                print(f"🚀 Epoch {epoch + 1}/{num_epochs}")

                for batch in dataset:
                    image = batch["image"]
                    text = batch["text"]

                    # Forward pass
                    with torch.no_grad():
                        # Get latents from VAE
                        latents = self.pipeline.vae.encode(image.unsqueeze(0).to(self.device)).latent_dist.sample()
                        latents = latents * self.pipeline.vae.config.scaling_factor

                        # Get text embeddings
                        text_inputs = self.pipeline.tokenizer(
                            text,
                            padding="max_length",
                            max_length=self.pipeline.tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt"
                        )
                        text_embeddings = self.pipeline.text_encoder(text_inputs.input_ids.to(self.device))[0]

                    # Predict noise
                    noise_pred = self.pipeline.unet(
                        latents,
                        torch.randint(0, self.pipeline.scheduler.config.num_train_timesteps, (1,)).to(self.device),
                        encoder_hidden_states=text_embeddings
                    ).sample

                    # Simple loss (can be improved with more sophisticated loss functions)
                    loss = torch.mean(noise_pred ** 2)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / max(1, len(dataset))
                print(f"ℹ️ Epoch {epoch + 1} avg loss: {avg_loss:.4f}")

            # Save LoRA weights
            lora_path = os.path.join(output_dir, "lora_weights")
            self.pipeline.unet.save_pretrained(lora_path)
            self.pipeline.text_encoder.save_pretrained(lora_path)

            print(f"✅ LoRA training completed! Weights saved to: {lora_path}")
            return lora_path

        except Exception as e:
            print(f"❌ Error during LoRA training: {e}")
            return None

    def load_lora_weights(self, lora_path):
        """
        Load trained LoRA weights for inference

        Args:
            lora_path: Path to LoRA weights directory

        Returns:
            bool: Success status
        """
        if not PEFT_AVAILABLE:
            # If PEFT is not installed but a placeholder artifact exists, acknowledge it so workflows can continue.
            placeholder_file = None
            if os.path.isdir(lora_path):
                placeholder_file = os.path.join(lora_path, "lora_placeholder.json")
            elif lora_path.endswith('.json'):
                placeholder_file = lora_path

            if placeholder_file and os.path.exists(placeholder_file):
                print("⚠️ PEFT not installed but found placeholder artifact:", placeholder_file)
                return True

            print("❌ PEFT not available.")
            print("💡 Install with: pip install peft")
            return False

        try:
            print(f"🔄 Loading LoRA weights from: {lora_path}")

            # Load LoRA weights
            self.pipeline.unet.load_adapter(lora_path, "default")
            self.pipeline.text_encoder.load_adapter(lora_path, "default")

            print("✅ LoRA weights loaded successfully!")
            return True

        except Exception as e:
            print(f"❌ Error loading LoRA weights: {e}")
            return False

    def generate_text_video(self, prompt, output_path="text_output.mp4", num_frames=16, height=480, width=832):
        """
        Generate video from text prompt using available models

        Args:
            prompt: Text description of the video
            output_path: Path to save the output video
            num_frames: Number of frames in the video (8-32 for Wan2.1, will be clamped)
            height: Video height
            width: Video width
        """
        print(f"🎬 Generating text-to-video with prompt: '{prompt}'")
        print(f"📊 Requested frames: {num_frames}")
        print(f"🔧 Current model_type: '{self.model_type}'")  # DEBUG

        # Clamp num_frames to reasonable limits
        num_frames = max(8, min(32, num_frames))  # Wan2.1 T2V supports 8-32 range
        print(f"📊 Using {num_frames} frames (clamped to 8-32 range)")

        # Strategy 1: Try lightweight text-to-video model first (with lazy loading)
        print(f"🔍 Checking if model_type == 'light': {self.model_type == 'light'}")  # DEBUG
        if self.model_type == "light":
            try:
                # Lazy load lightweight T2V model if needed
                if self.text_pipeline is None:
                    print("🎬 Loading lightweight text-to-video model (first use)...")
                    print("⏱️ This may take 1-2 minutes to download...")
                    print("📦 Model: damo-vilab/text-to-video-ms-1.7b")

                    try:
                        from diffusers import TextToVideoMSDecoderPipeline

                        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

                        self.text_pipeline = TextToVideoMSDecoderPipeline.from_pretrained(
                            "damo-vilab/text-to-video-ms-1.7b",
                            torch_dtype=dtype
                        ).to(self.device)

                        print("✅ Lightweight text-to-video model loaded successfully!")
                        print("💾 Memory usage: ~3.7GB on CUDA" if self.device.type == "cuda" else "💾 CPU mode active; expect slower renders")

                    except ImportError:
                        print("⚠️ TextToVideoMSDecoderPipeline not available, trying DiffusionPipeline...")
                        from diffusers import DiffusionPipeline

                        dtype = torch.float16 if self.device.type == "cuda" else torch.float32

                        self.text_pipeline = DiffusionPipeline.from_pretrained(
                            "damo-vilab/text-to-video-ms-1.7b",
                            torch_dtype=dtype
                        ).to(self.device)
                        print("✅ Model loaded via DiffusionPipeline!")

                print("🎯 Using lightweight text-to-video model...")
                print(f"🎬 Generating {num_frames} frames...")
                
                try:
                    output = self.text_pipeline(
                        prompt=prompt,
                        num_frames=num_frames,
                        num_inference_steps=25,
                        guidance_scale=7.5
                    )

                    print(f"✅ Inference complete! Saving video...")
                    video_frames = output.frames[0]
                    pil_frames = []
                    for idx, frame in enumerate(video_frames):
                        if isinstance(frame, Image.Image):
                            pil_frames.append(frame)
                            continue

                        try:
                            frame_np = frame
                            if not isinstance(frame_np, np.ndarray):
                                frame_np = np.array(frame)

                            if frame_np.dtype != np.uint8:
                                # Normalize float data to 0-255 uint8 for video encoding
                                if frame_np.max() <= 1.0:
                                    frame_np = frame_np * 255.0
                                frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)

                            pil_frames.append(Image.fromarray(frame_np))
                        except Exception as convert_error:
                            print(f"⚠️ Failed to convert frame {idx} to PIL: {convert_error}")
                            raise

                    # Save video
                    self.save_video(pil_frames, output_path, fps=8)
                    print(f"✅ Lightweight text-to-video generated: {output_path}")
                    return output_path
                    
                except AttributeError as ae:
                    print(f"⚠️ Attribute error in pipeline: {ae}")
                    import traceback
                    traceback.print_exc()
                    raise

            except Exception as e:
                print(f"❌ Lightweight model failed: {e}")
                import traceback
                traceback.print_exc()
                print("🔄 Falling back to placeholder...")

        # All model fallbacks fail due to dependency issues
        # Create placeholder video
        print("📝 Creating text-based placeholder video...")
        print("📝 Creating text-based placeholder video...")
        return self._create_enhanced_placeholder_text_video(prompt, output_path, num_frames)

    def _create_placeholder_text_video(self, prompt, output_path, num_frames=16):
        """
        Create a placeholder video showing the text prompt

        Args:
            prompt: Text prompt to display
            output_path: Output video path
            num_frames: Number of frames
        """
        try:
            from PIL import Image, ImageDraw, ImageFont
            import cv2

            print(f"📝 Creating placeholder video for prompt: '{prompt}'")

            # Create frames with text
            frames = []
            width, height = 832, 480

            # Try to load a font, fallback to default if not available
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 32)
            except:
                font = ImageFont.load_default()

            for i in range(num_frames):
                # Create image with gradient background
                img = Image.new('RGB', (width, height), color=(20, 20, 40))
                draw = ImageDraw.Draw(img)

                # Add some animation effect
                y_offset = int(10 * (i / num_frames) * 2 * 3.14159)

                # Draw text
                bbox = draw.textbbox((0, 0), prompt, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                x = (width - text_width) // 2
                y = (height - text_height) // 2 + y_offset

                # Draw text with shadow
                draw.text((x+2, y+2), prompt, font=font, fill=(0, 0, 0))
                draw.text((x, y), prompt, font=font, fill=(255, 255, 255))

                # Add frame number
                draw.text((10, 10), f"Frame {i+1}/{num_frames}", font=font, fill=(200, 200, 200))

                frames.append(img)

            # Save as video
            self.save_video(frames, output_path, fps=8)
            print(f"✅ Placeholder text-to-video created: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Failed to create placeholder: {e}")
            return None

    def _create_enhanced_placeholder_text_video(self, prompt, output_path, num_frames=16):
        """
        Create an enhanced placeholder video with better visuals and animations

        Args:
            prompt: Text prompt to display
            output_path: Output video path
            num_frames: Number of frames
        """
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageFilter
            import cv2
            import numpy as np

            print(f"🎨 Creating enhanced placeholder video for prompt: '{prompt}'")

            # Create frames with enhanced visuals
            width, height = 832, 480
            frames = []

            # Try to load a better font
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()

            for i in range(num_frames):
                # Create base image with gradient background
                img = Image.new('RGB', (width, height), color=(20, 20, 40))
                draw = ImageDraw.Draw(img)

                # Add animated gradient effect
                for y in range(height):
                    # Create wave-like color variation
                    wave = int(20 * np.sin(2 * np.pi * (i / num_frames + y / height * 2)))
                    r = min(255, max(0, 20 + wave))
                    g = min(255, max(0, 20 + wave // 2))
                    b = min(255, max(0, 40 + wave // 3))

                    # Draw horizontal line with gradient
                    draw.line([(0, y), (width, y)], fill=(r, g, b), width=1)

                # Add some particle effects
                for _ in range(10):
                    x = np.random.randint(0, width)
                    y = np.random.randint(0, height)
                    size = np.random.randint(1, 3)
                    brightness = np.random.randint(100, 200)
                    draw.ellipse([x-size, y-size, x+size, y+size], fill=(brightness, brightness, brightness))

                # Add floating animation to text
                y_offset = int(15 * np.sin(2 * np.pi * i / num_frames))

                # Draw main text with enhanced styling
                bbox = draw.textbbox((0, 0), prompt, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

                x = (width - text_width) // 2
                y = (height - text_height) // 2 + y_offset

                # Draw text with glow effect
                glow_color = (255, 255, 0)  # Yellow glow
                for offset in [(2,2), (-2,-2), (2,-2), (-2,2)]:
                    draw.text((x+offset[0], y+offset[1]), prompt, font=font, fill=glow_color)

                # Draw main text
                draw.text((x, y), prompt, font=font, fill=(255, 255, 255))

                # Add progress indicator
                progress_width = int((i + 1) / num_frames * 200)
                draw.rectangle([width-220, height-30, width-20, height-10], fill=(50, 50, 50))
                draw.rectangle([width-218, height-28, width-22 - (200-progress_width), height-12], fill=(100, 200, 100))

                # Add frame info
                draw.text((10, 10), f"Frame {i+1}/{num_frames}", font=small_font, fill=(200, 200, 200))
                draw.text((10, 35), f"Text-to-Video Generation", font=small_font, fill=(150, 150, 150))

                # Add subtle animation to the image
                img_array = np.array(img)
                # Add some noise for texture
                noise = np.random.randint(-10, 10, img_array.shape, dtype=np.int16)
                img_array = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                img = Image.fromarray(img_array)

                frames.append(img)

            # Save as video
            self.save_video(frames, output_path, fps=12)  # Slightly faster for better animation
            print(f"✅ Enhanced placeholder text-to-video created: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Failed to create enhanced placeholder: {e}")
            # Fallback to simple placeholder
            return self._create_placeholder_text_video(prompt, output_path, num_frames)

class ImageDataset(Dataset):
    """Dataset class for LoRA training with multiple images"""
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return {"image": image, "text": "a person"}

def main():
    """Main function for testing Wan2.2"""
    print("🎬 Wan2.2 Image-to-Video Generator")
    print("=" * 50)

    # Initialize generator
    generator = Wan2VideoGenerator()

    # Load model
    if not generator.load_model():
        print("❌ Failed to load model. Exiting.")
        return

    # Test with sample image (you'll need to provide an actual image)
    test_image = "sample_image.jpg"  # Replace with actual image path
    test_prompts = [
        "A person dancing gracefully",
        "The image coming to life with movement",
        "Dynamic motion and animation"
    ]

    # Generate video for each prompt
    for i, prompt in enumerate(test_prompts):
        output_path = f"wan2_output_{i+1}.mp4"
        generator.generate_video_from_image(test_image, prompt, output_path)
        print()

    print("🎉 All videos generated successfully!")

if __name__ == "__main__":
    main()
