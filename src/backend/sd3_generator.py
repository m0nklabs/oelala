import os
import torch
import gc
from diffusers import StableDiffusion3Pipeline, SD3Transformer2DModel
from transformers import BitsAndBytesConfig, T5EncoderModel
import logging
import os

logger = logging.getLogger(__name__)

class SD3ImageGenerator:
    def __init__(self):
        self.pipe = None
        self.device_map = None
        
        # Ensure CUDA devices are visible if not already set
        # Note: This might need to be set before process start in production
        if "CUDA_VISIBLE_DEVICES" not in os.environ:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
            
        self._load_model()

    def _load_model(self):
        # Primary choice: SD3.5 Large INT8. If it fails due to config/weight mismatch, fall back to SD3 Medium.
        candidates = [
            {
                "name": "SD3.5 Large INT8",
                "repo": "stabilityai/stable-diffusion-3.5-large",
                "notes": "preferred",
            },
            {
                "name": "SD3 Medium INT8 (fallback)",
                "repo": "stabilityai/stable-diffusion-3-medium-diffusers",
                "notes": "fallback for incompatible checkpoints",
            },
        ]

        last_error: Exception | None = None
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")

        for cand in candidates:
            try:
                logger.info(f"🚀 Loading {cand['name']} from {cand['repo']} (INT8 multi-GPU)...")

                quantization_config = BitsAndBytesConfig(load_in_8bit=True)

                # 1. Load Transformer (INT8) on GPU 0
                logger.info("   - Loading Transformer (INT8) on GPU 0...")
                transformer = SD3Transformer2DModel.from_pretrained(
                    cand["repo"],
                    subfolder="transformer",
                    quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16,
                    token=hf_token,
                )

                # 2. Load T5 Encoder (INT8) on GPU 1
                logger.info("   - Loading T5 Encoder (INT8) on GPU 1...")
                text_encoder_3_raw = T5EncoderModel.from_pretrained(
                    cand["repo"],
                    subfolder="text_encoder_3",
                    quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16,
                    device_map={"": 1},  # Force to GPU 1
                    token=hf_token,
                )

                # Monkey patch forward for device awareness
                text_encoder_3 = text_encoder_3_raw
                original_forward = text_encoder_3.forward

                def device_aware_forward(input_ids, **kwargs):
                    input_ids = input_ids.to(text_encoder_3.device)
                    return original_forward(input_ids, **kwargs)

                text_encoder_3.forward = device_aware_forward

                # 3. Assemble Pipeline
                logger.info("   - Assembling Pipeline...")
                self.pipe = StableDiffusion3Pipeline.from_pretrained(
                    cand["repo"],
                    transformer=transformer,
                    text_encoder_3=text_encoder_3,
                    torch_dtype=torch.bfloat16,
                    token=hf_token,
                )

                # 4. Move small components to GPU 0
                logger.info("   - Moving small components to GPU 0...")
                self.pipe.vae.to("cuda:0")
                self.pipe.text_encoder.to("cuda:0")
                self.pipe.text_encoder_2.to("cuda:0")

                logger.info(f"✅ {cand['name']} loaded successfully!")
                return

            except Exception as e:
                last_error = e
                logger.error(f"❌ Failed to load {cand['name']}: {e}")
                import traceback
                traceback.print_exc()
                # Try next candidate

        # If we exit the loop, all candidates failed
        self.pipe = None
        raise RuntimeError(
            f"SD3 pipeline failed to load after attempts: {[c['name'] for c in candidates]}; last error: {last_error}"
        )

    def generate(self, prompt, negative_prompt="", width=1024, height=1024, num_inference_steps=28, guidance_scale=4.5, seed=None, callback=None, callback_steps=1):
        if not self.pipe:
            raise RuntimeError("Model not initialized")

        if seed is not None:
            generator = torch.Generator(device="cuda:0").manual_seed(seed)
        else:
            generator = None

        logger.info(f"🎨 Generating image for prompt: {prompt[:50]}...")
        
        with torch.inference_mode():
            # SD3 pipeline in this version does not support callback; we ignore callback params to avoid errors
            image = self.pipe(  # type: ignore[call-arg]
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
            ).images[0]
            
        return image
