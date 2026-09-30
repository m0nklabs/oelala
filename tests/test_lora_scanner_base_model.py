"""Tests for lora_scanner base-model derivation (drives the compat filter)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "backend"))

from lora_scanner import _derive_base_model  # noqa: E402


def test_krea2_subdirectory():
    assert _derive_base_model("krea2/KNP_000003000.safetensors") == "krea2"
    assert _derive_base_model("krea2/snofs_krea_v1_4.safetensors") == "krea2"


def test_krea2_filename_marker():
    assert _derive_base_model("some_krea_model_v2.safetensors") == "krea2"


def test_existing_mappings_unchanged():
    assert _derive_base_model("wan/foo.safetensors") == "wan2.2"
    assert _derive_base_model("ltx/bar.safetensors") == "ltx"
    assert _derive_base_model("minimax-h3/baz.safetensors") == "minimax_h3"
    assert _derive_base_model("totally-unknown.safetensors") == ""
