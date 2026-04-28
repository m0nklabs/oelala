"""Compatibility exports for the Qwen Image Edit cloud adapter."""

from __future__ import annotations

from .cloud_i2i import I2IEditCloudAdapter, build_i2i_edit_workflow


class QwenEditCloudAdapter(I2IEditCloudAdapter):
    """Legacy public adapter name for Qwen Image Edit."""

    name = "qwen-cloud-edit"
    model_family = "qwen_image_edit"


build_qwen_edit_workflow = build_i2i_edit_workflow
