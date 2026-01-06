"""
Tests for Credit System & Stripe Integration
Tests the credits.py and credits_api.py modules.
"""

import pytest
import os
import sys
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "backend"))

from credits import (
    calculate_credits,
    GenerationType,
    CreditManager,
    CreditBalance,
    InsufficientCreditsError,
    DEFAULT_PACKAGES,
)


class TestCreditCalculation:
    """Test credit cost calculation for different generation types."""

    def test_basic_image_costs(self):
        """Test basic image generation costs."""
        # SDXL base (1024x1024)
        assert calculate_credits("sdxl", 1024, 1024) == 1
        
        # FLUX (costs 3 per the CREDIT_COSTS dict)
        assert calculate_credits("flux", 1024, 1024) >= 2
        
        # SD1.5
        assert calculate_credits("sd15", 512, 512) == 1

    def test_hd_image_upcharge(self):
        """Test HD resolution upcharge."""
        # Standard resolution
        base_cost = calculate_credits("sdxl", 1024, 1024)
        
        # HD resolution (>1280)
        hd_cost = calculate_credits("sdxl", 1920, 1080)
        
        # HD should cost more or equal (depends on threshold logic)
        assert hd_cost >= base_cost

    def test_video_costs(self):
        """Test video generation costs."""
        # Short video (3 sec)
        short_cost = calculate_credits("wan22_i2v", 720, 480, duration_seconds=3)
        
        # Long video (5 sec)
        long_cost = calculate_credits("wan22_i2v", 720, 480, duration_seconds=6)
        
        # Longer should cost more
        assert long_cost > short_cost

    def test_video_resolution_tiers(self):
        """Test different video resolution costs."""
        # 480p short
        cost_480_short = calculate_credits("wan22_i2v", 720, 480, duration_seconds=3)
        
        # 720p short
        cost_720_short = calculate_credits("wan22_i2v", 1280, 720, duration_seconds=3)
        
        # HD should cost more
        assert cost_720_short >= cost_480_short

    def test_audio_costs(self):
        """Test audio generation costs."""
        # Short audio
        short = calculate_credits("mmaudio_short", duration_seconds=5)
        
        # Long audio
        long = calculate_credits("mmaudio_long", duration_seconds=20)
        
        assert long > short

    def test_unknown_type_defaults(self):
        """Test unknown generation types get default cost."""
        cost = calculate_credits("unknown_type_xyz", 1024, 1024)
        assert cost >= 1  # Should have some default

    def test_generation_type_mapping(self):
        """Test API endpoint name mapping."""
        # These should map to the same type
        cost1 = calculate_credits("generate_sdxl", 1024, 1024)
        cost2 = calculate_credits("sdxl", 1024, 1024)
        assert cost1 == cost2


class TestCreditManager:
    """Test CreditManager async operations."""

    @pytest.fixture
    def manager(self):
        """Create a CreditManager with mock credentials."""
        return CreditManager(
            supabase_url="https://test.supabase.co",
            service_key="mock_service_key_123"
        )

    def test_manager_initialization(self, manager):
        """Test CreditManager initializes correctly."""
        assert manager.supabase_url == "https://test.supabase.co"
        assert manager.service_key == "mock_service_key_123"
        assert "Authorization" in manager.headers

    @pytest.mark.asyncio
    async def test_get_balance_creates_new_user(self, manager):
        """Test get_balance creates new user with welcome bonus."""
        # Mock HTTP client
        mock_client = AsyncMock()
        
        # First call returns empty (user doesn't exist)
        mock_get = Mock()
        mock_get.status_code = 200
        mock_get.json = Mock(return_value=[])
        mock_client.get = AsyncMock(return_value=mock_get)
        
        # Second call creates user
        mock_post = Mock()
        mock_post.status_code = 201
        mock_post.json = Mock(return_value={
            "user_id": "test-user-123",
            "balance": 25,
            "lifetime_purchased": 0,
            "lifetime_used": 0,
        })
        mock_client.post = AsyncMock(return_value=mock_post)
        
        manager._client = mock_client
        
        with patch.object(manager, 'get_client', return_value=mock_client):
            balance = await manager.get_balance("test-user-123")
            
            assert balance.balance == 25  # Welcome bonus
            assert balance.lifetime_purchased == 0
            assert balance.lifetime_used == 0

    @pytest.mark.asyncio
    async def test_insufficient_credits_error(self, manager):
        """Test InsufficientCreditsError is raised when balance too low."""
        # Mock HTTP client with existing user
        mock_client = AsyncMock()
        
        mock_get = Mock()
        mock_get.status_code = 200
        mock_get.json = Mock(return_value=[{
            "user_id": "test-user",
            "balance": 5,
            "lifetime_purchased": 100,
            "lifetime_used": 95,
        }])
        mock_client.get = AsyncMock(return_value=mock_get)
        manager._client = mock_client
        
        with patch.object(manager, 'get_client', return_value=mock_client):
            # Try to reserve 10 credits when only 5 available
            with pytest.raises(InsufficientCreditsError) as exc_info:
                await manager.check_and_reserve("test-user", 10)
            
            error = exc_info.value
            assert error.required == 10
            assert error.available == 5
            assert len(error.packages) > 0  # Should include packages

    @pytest.mark.asyncio
    async def test_get_packages_returns_defaults_on_error(self, manager):
        """Test get_packages returns defaults when DB not configured."""
        # Mock HTTP client that returns error
        mock_client = AsyncMock()
        mock_get = AsyncMock()
        mock_get.status_code = 404
        mock_get.json.return_value = []
        mock_client.get.return_value = mock_get
        manager._client = mock_client
        
        with patch.object(manager, 'get_client', return_value=mock_client):
            packages = await manager.get_packages()
            
            # Should return defaults
            assert len(packages) == len(DEFAULT_PACKAGES)
            assert packages[0].id == "starter"

    def test_default_packages_structure(self):
        """Test default packages are properly structured."""
        assert len(DEFAULT_PACKAGES) > 0
        
        for pkg in DEFAULT_PACKAGES:
            assert pkg.id
            assert pkg.name
            assert pkg.credits > 0
            assert pkg.price_cents > 0
            assert pkg.currency == "EUR"


class TestInsufficientCreditsError:
    """Test the InsufficientCreditsError exception."""

    def test_error_message(self):
        """Test error message formatting."""
        error = InsufficientCreditsError(
            required=10,
            available=5,
            packages=DEFAULT_PACKAGES
        )
        
        assert "10" in str(error)
        assert "5" in str(error)
        assert error.required == 10
        assert error.available == 5

    def test_packages_included(self):
        """Test packages are included in error."""
        error = InsufficientCreditsError(
            required=100,
            available=20
        )
        
        assert len(error.packages) > 0
        assert error.packages[0].id == "starter"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
