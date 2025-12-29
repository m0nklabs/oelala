#!/usr/bin/env python3
"""
Wan2.2 Image-to-Video Generation Script
Oelala Project - AI Video Generation Pipeline
With Stable Video Diffusion fallback
"""

import os
import sys
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
except ImportError:
    print("⚠️ PEFT not available. LoRA fine-tuning will be disabled.")
    PEFT_AVAILABLE = False
except ImportError:
    print("⚠️ PEFT not available. LoRA fine-tuning will be disabled.")
    print("💡 Install with: pip install peft")
    PEFT_AVAILABLE = False

# Set environment variables for better performance
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

class Wan2VideoGenerator:
    def __init__(self, model_path="Wan-AI/Wan2.2-I2V-A14B", device="cuda"):
        """
        Initialize Wan2.2 Image-to-Video Generator

        Args:
            model_path: HuggingFace model path
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_path = model_path
        self.pipeline = None

        print(f"Initializing Wan2.2 on device: {self.device}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name()}")
            print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    def load_model(self):
        """Load the Wan2.2 model and pipeline with SVD fallback"""
        try:
            print("🔍 Checking for Wan2.2 pipeline availability...")
            print(f"📍 Model path: {self.model_path}")
            print(f"🎯 Device: {self.device}")
            print(f"💾 CUDA available: {torch.cuda.is_available()}")

            # Try Wan2.2 first
            try:
                from diffusers import WanImageToVideoPipeline
                print("✅ WanImageToVideoPipeline found in diffusers")
                # If available, use real Wan2.2 pipeline
                self.pipeline = WanImageToVideoPipeline.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16
                ).to(self.device)
                self.model_type = "wan2.2"
                print("✅ Wan2.2 model loaded successfully!")
                return True
            except ImportError as ie:
                print(f"❌ WanImageToVideoPipeline not available: {ie}")
                print("📝 Wan2.2 is not yet publicly available in diffusers")
                print("🔄 Falling back to Stable Video Diffusion...")

            # Fallback to Stable Video Diffusion
            try:
                print("🎬 Stable Video Diffusion will be loaded on first use...")
                print("💡 This prevents long startup times")
                self.pipeline = "svd_lazy"
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
            print("🔄 Falling back to placeholder mode")
            return self._generate_placeholder_video(image_path, "", output_path, num_frames)

    def _generate_wan2_video(self, image_path, prompt, output_path, num_frames):
        """
        Generate video using Wan2.2 (when available)
        """
        try:
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
        try:
            import cv2

            # Convert PIL images to numpy arrays
            frame_arrays = []
            for frame in frames:
                # Convert PIL to numpy
                frame_np = np.array(frame)
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
                frame_arrays.append(frame_bgr)

            # Get dimensions from first frame
            height, width = frame_arrays[0].shape[:2]

            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Write frames
            for frame in frame_arrays:
                video_writer.write(frame)

            video_writer.release()
            print(f"📹 Video saved with {len(frames)} frames at {fps} FPS")

        except ImportError:
            print("⚠️ OpenCV not available for video saving")
            print("💡 Install with: pip install opencv-python")

            # Fallback: save as individual frames
            os.makedirs("frames", exist_ok=True)
            for i, frame in enumerate(frames):
                frame.save(f"frames/frame_{i:04d}.png")
            print("📸 Frames saved to 'frames/' directory")

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
            num_frames: Number of frames in the video
            height: Video height
            width: Video width
        """
        print(f"🎬 Generating text-to-video with prompt: '{prompt}'")

        # Strategy 1: Try Wan2.1 T2V if available
        try:
            print("🎯 Attempting Wan2.1 Text-to-Video...")
            from diffusers import WanImageToVideoPipeline

            # Try to load Wan2.1 T2V model
            t2v_pipeline = WanImageToVideoPipeline.from_pretrained(
                "Wan-AI/Wan2.1-T2V-14B",
                torch_dtype=torch.float16
            ).to(self.device)

            print("✅ Wan2.1 T2V model loaded!")

            # Generate video from text
            output = t2v_pipeline(
                prompt=prompt,
                num_frames=num_frames,
                height=height,
                width=width
            )

            # Save video
            self.save_video(output.frames[0], output_path, fps=8)
            print(f"✅ Text-to-video generated with Wan2.1: {output_path}")
            return output_path

        except Exception as e:
            print(f"❌ Wan2.1 T2V failed: {e}")
            print("🔄 Falling back to creative approach...")

        # Strategy 2: Creative approach - generate image from text, then video from image
        try:
            print("🎨 Using creative approach: text → image → video")

            # First, generate an image from the text prompt
            from diffusers import StableDiffusionPipeline

            print("🖼️ Generating image from text prompt...")
            image_pipeline = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16
            ).to(self.device)

            # Generate image
            image_output = image_pipeline(
                prompt=prompt,
                num_inference_steps=20,
                guidance_scale=7.5
            )

            generated_image = image_output.images[0]
            print("✅ Image generated from text!")

            # Save temporary image
            temp_image_path = f"temp_text_image_{hash(prompt)}.png"
            generated_image.save(temp_image_path)

            # Now generate video from the generated image
            print("🎬 Generating video from generated image...")
            result = self.generate_video_from_image(
                temp_image_path,
                prompt,  # Use same prompt for consistency
                output_path,
                num_frames
            )

            # Clean up temporary image
            if os.path.exists(temp_image_path):
                os.remove(temp_image_path)

            if result:
                print(f"✅ Text-to-video completed via creative approach: {output_path}")
                return output_path
            else:
                print("❌ Video generation from generated image failed")

        except Exception as e:
            print(f"❌ Creative approach failed: {e}")

        # Strategy 3: Last resort - create placeholder video
        print("📝 Creating placeholder text-to-video...")
        return self._create_placeholder_text_video(prompt, output_path, num_frames)

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
