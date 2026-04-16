"""
Wan2.2 I2V Lightning adapter — Q4KM GGUF with Lightning defaults.

Uses build_enhanced_workflow() — fast 4-step generation.
"""

from __future__ import annotations

from generation.adapters.local.i2v_wan22_base import QuantConfig, Wan22LocalI2VBase


class Wan22LocalI2VLightningAdapter(Wan22LocalI2VBase):
    """Wan2.2 I2V Q4KM Lightning — ultra-fast 4-step generation."""

    name = "wan22-local-i2v-lightning"

    def _get_quant_config(self) -> QuantConfig:
        return QuantConfig(
            name="Q4KM_lightning",
            builder_method="build_enhanced_workflow",
            default_steps=4,
            default_cfg=1.0,
            default_sampler="uni_pc",
            default_scheduler="simple",
            max_frames=41,
            resolution_presets=["480p", "576p", "720p"],
        )
