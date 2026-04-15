"""
Wan2.2 Local I2V variants — each delegates to a comfyui_client builder.

- Q6: Standard Q6_K GGUF dual-pass (build_q6_workflow)
- Q8 DisTorch2: Q8_0 GGUF with DisTorch2 multi-GPU (build_distorch2_q8_workflow)
- Q8 BlockSwap: Q8_0 GGUF with BlockSwap VRAM optimization (build_blockswap_q8_workflow)
- Ultra Q8: Maximum quality Q8 workflow (build_ultra_q8_workflow)
"""

from __future__ import annotations

from generation.adapters.local.i2v_wan22_base import QuantConfig, Wan22LocalI2VBase


class Wan22LocalI2VQ6Adapter(Wan22LocalI2VBase):
    """Wan2.2 I2V Q6_K — standard dual-pass, memory-efficient."""

    name = "wan22-local-i2v-q6"

    def _get_quant_config(self) -> QuantConfig:
        return QuantConfig(
            name="Q6_K",
            builder_method="build_q6_workflow",
            default_steps=6,
            default_cfg=1.0,
            default_sampler="uni_pc",
            default_scheduler="normal",
            max_frames=321,
            resolution_presets=["480p", "576p", "720p"],
        )


class Wan22LocalI2VDisTorch2Adapter(Wan22LocalI2VBase):
    """Wan2.2 I2V Q8_0 — DisTorch2 multi-GPU distribution."""

    name = "wan22-local-i2v-distorch2"

    def _get_quant_config(self) -> QuantConfig:
        return QuantConfig(
            name="Q8_0_distorch2",
            builder_method="build_distorch2_q8_workflow",
            default_steps=6,
            default_cfg=1.0,
            default_sampler="uni_pc",
            default_scheduler="normal",
            max_frames=321,
            resolution_presets=["480p", "576p", "720p"],
        )


class Wan22LocalI2VBlockSwapAdapter(Wan22LocalI2VBase):
    """Wan2.2 I2V Q8_0 — BlockSwap VRAM optimization + NAG + Lightning LoRA."""

    name = "wan22-local-i2v-blockswap"

    def _get_quant_config(self) -> QuantConfig:
        return QuantConfig(
            name="Q8_0_blockswap",
            builder_method="build_blockswap_q8_workflow",
            default_steps=8,
            default_cfg=1.0,
            default_sampler="uni_pc",
            default_scheduler="normal",
            max_frames=161,
            resolution_presets=["480p", "576p", "720p"],
            supports_extra_params=True,
        )


class Wan22LocalI2VUltraAdapter(Wan22LocalI2VBase):
    """Wan2.2 I2V Q8_0 Ultra — maximum quality with all enhancements."""

    name = "wan22-local-i2v-ultra"

    def _get_quant_config(self) -> QuantConfig:
        return QuantConfig(
            name="Q8_0_ultra",
            builder_method="build_ultra_q8_workflow",
            default_steps=8,
            default_cfg=1.0,
            default_sampler="uni_pc",
            default_scheduler="normal",
            max_frames=161,
            resolution_presets=["480p", "576p", "720p"],
            supports_extra_params=True,
        )
