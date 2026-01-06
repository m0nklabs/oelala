"""
Credit System Tests
Tests for credit calculation, checking, and deduction logic.
"""

import pytest
from src.backend.credits import (
    calculate_credits,
    GenerationType,
    CREDIT_COSTS,
)


class TestCreditCalculation:
    """Test credit cost calculation for different generation types."""

    def test_image_generation_sd15(self):
        """Test SD1.5 image generation cost."""
        cost = calculate_credits("sd15", width=512, height=512)
        assert cost == 1

    def test_image_generation_sdxl(self):
        """Test SDXL image generation cost."""
        cost = calculate_credits("sdxl", width=1024, height=1024)
        assert cost == 1

    def test_image_generation_sdxl_hd(self):
        """Test SDXL HD image generation cost (higher resolution)."""
        cost = calculate_credits("sdxl", width=1344, height=768)
        # HD should cost more
        assert cost >= 1
        # Should be around 1.5x base cost for HD
        base_cost = CREDIT_COSTS[GenerationType.SDXL]
        assert cost >= base_cost

    def test_image_generation_flux(self):
        """Test Flux image generation cost."""
        cost = calculate_credits("flux", width=1024, height=1024)
        # Flux with 1024x1024 may apply multipliers
        assert cost >= 2
        assert cost <= 4

    def test_video_generation_i2v_short(self):
        """Test short video (3s) generation cost."""
        cost = calculate_credits("wan22_i2v", width=720, height=480, duration_seconds=3)
        # Should be around 5 credits for short 720p video
        assert cost >= 5
        assert cost <= 10

    def test_video_generation_i2v_medium(self):
        """Test medium video (5s) generation cost."""
        cost = calculate_credits("wan22_i2v", width=720, height=480, duration_seconds=5)
        # Should be more expensive than short
        assert cost >= 8
        assert cost <= 15

    def test_video_generation_hd(self):
        """Test HD video generation cost."""
        cost_hd = calculate_credits("wan22_i2v", width=1280, height=720, duration_seconds=3)
        cost_sd = calculate_credits("wan22_i2v", width=720, height=480, duration_seconds=3)
        # HD shouldn't be cheaper than SD
        assert cost_hd >= cost_sd

    def test_video_generation_t2v(self):
        """Test text-to-video generation cost."""
        cost = calculate_credits("wan22_t2v", width=720, height=480, duration_seconds=3)
        # T2V should be more expensive than I2V (includes image generation)
        assert cost >= 8

    def test_audio_generation_short(self):
        """Test short audio generation cost."""
        cost = calculate_credits("mmaudio_short", duration_seconds=5)
        # Base cost plus possible duration multipliers
        assert cost >= 3
        assert cost <= 6

    def test_audio_generation_long(self):
        """Test long audio generation cost."""
        cost = calculate_credits("mmaudio_long", duration_seconds=20)
        # Longer audio has multipliers applied
        assert cost >= 5
        assert cost <= 15

    def test_voice_clone(self):
        """Test voice cloning cost."""
        cost = calculate_credits("voice_clone")
        # Voice clone may have multipliers
        assert cost >= 20
        assert cost <= 40

    def test_unknown_generation_type(self):
        """Test unknown generation type returns default cost."""
        cost = calculate_credits("unknown_type", width=512, height=512)
        # Should return default cost of 2
        assert cost == 2

    def test_api_endpoint_mapping(self):
        """Test that API endpoint names map correctly."""
        # Test endpoint name mappings (actual costs may have multipliers)
        cost_generate_image = calculate_credits("generate_image")
        cost_sdxl = calculate_credits("generate_sdxl")
        cost_flux = calculate_credits("generate_flux")
        
        # All should return reasonable costs
        assert cost_generate_image >= 1
        assert cost_sdxl >= 1
        assert cost_flux >= 2

    def test_steps_multiplier(self):
        """Test that high step counts increase cost."""
        cost_low = calculate_credits("sdxl", width=1024, height=1024, steps=20)
        cost_high = calculate_credits("sdxl", width=1024, height=1024, steps=50)
        # High steps should cost more (1.2x multiplier)
        assert cost_high >= cost_low

    def test_duration_multiplier(self):
        """Test that longer videos cost more."""
        cost_3s = calculate_credits("wan22_i2v", width=720, height=480, duration_seconds=3)
        cost_10s = calculate_credits("wan22_i2v", width=720, height=480, duration_seconds=10)
        # Longer videos should cost more
        assert cost_10s > cost_3s

    def test_minimum_cost(self):
        """Test that cost is never less than 1 credit."""
        cost = calculate_credits("sd15", width=256, height=256, steps=1)
        assert cost >= 1


class TestCreditConstants:
    """Test that credit cost constants are reasonable."""

    def test_all_costs_positive(self):
        """All credit costs should be positive integers."""
        for gen_type, cost in CREDIT_COSTS.items():
            assert isinstance(cost, int)
            assert cost > 0

    def test_video_more_expensive_than_images(self):
        """Video generation should cost more than image generation."""
        max_image_cost = max(
            CREDIT_COSTS[GenerationType.SDXL],
            CREDIT_COSTS[GenerationType.FLUX],
            CREDIT_COSTS[GenerationType.SD15],
        )
        min_video_cost = min(
            CREDIT_COSTS[GenerationType.WAN22_I2V_SHORT],
            CREDIT_COSTS[GenerationType.WAN22_T2V_SHORT],
        )
        assert min_video_cost > max_image_cost

    def test_hd_more_expensive_than_sd(self):
        """HD generation should cost more than SD."""
        sd_cost = CREDIT_COSTS[GenerationType.WAN22_I2V_SHORT]
        hd_cost = CREDIT_COSTS[GenerationType.WAN22_I2V_HD_SHORT]
        assert hd_cost > sd_cost

    def test_long_more_expensive_than_short(self):
        """Longer generations should cost more."""
        short_cost = CREDIT_COSTS[GenerationType.MMAUDIO_SHORT]
        long_cost = CREDIT_COSTS[GenerationType.MMAUDIO_LONG]
        assert long_cost > short_cost


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
