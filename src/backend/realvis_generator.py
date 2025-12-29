import os
import logging
import torch
from diffusers import StableDiffusionXLPipeline

logger = logging.getLogger(__name__)


class RealVisXLImageGenerator:
    """Text-to-image generator using RealVisXL V5.0 (SDXL) on a single GPU."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.model_id = "SG161222/RealVisXL_V5.0"
        self.pipe: StableDiffusionXLPipeline | None = None
        self._load_model()

    def _load_model(self):
        hf_token = os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
        try:
            logger.info(f"🚀 Loading RealVisXL V5.0 on {self.device}...")
            self.pipe = StableDiffusionXLPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.float16,
                use_safetensors=True,
                token=hf_token,
            )
            logger.info("   - Moving pipeline to GPU...")
            self.pipe.to(self.device)
            self.pipe.enable_vae_tiling()
            self.pipe.enable_vae_slicing()
            logger.info("✅ RealVisXL pipeline ready")
        except Exception as exc:
            logger.error(f"❌ Failed to load RealVisXL: {exc}")
            self.pipe = None
            raise RuntimeError("RealVisXL pipeline failed to load") from exc

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int | None = None,
    ):
        if not self.pipe:
            raise RuntimeError("Model not initialized")

        generator = torch.Generator(device=self.device).manual_seed(seed) if seed is not None else None
        logger.info(f"🎨 [RealVisXL] Generating image for prompt: {prompt[:80]}...")

        with torch.inference_mode():
            image = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            ).images[0]

        return image
