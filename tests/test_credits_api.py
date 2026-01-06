"""
Tests for Credits API Endpoints
Tests the FastAPI endpoints in credits_api.py
"""

import pytest
import os
import sys
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "backend"))


class TestCreditsAPIEndpoints:
    """Test credits API endpoints."""

    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user."""
        user = Mock()
        user.id = "test-user-123"
        user.email = "test@example.com"
        return user

    @pytest.fixture
    def mock_credit_manager(self):
        """Mock CreditManager."""
        from credits import CreditBalance, CreditPackageResponse
        
        manager = AsyncMock()
        
        # Mock get_balance
        manager.get_balance.return_value = CreditBalance(
            balance=100,
            lifetime_purchased=500,
            lifetime_used=400,
        )
        
        # Mock get_packages
        manager.get_packages.return_value = [
            CreditPackageResponse(
                id="starter",
                name="Starter",
                credits=100,
                price_cents=500,
                currency="EUR",
            ),
            CreditPackageResponse(
                id="pro",
                name="Pro",
                credits=1500,
                price_cents=5000,
                currency="EUR",
            ),
        ]
        
        return manager

    def test_estimate_request_structure(self):
        """Test EstimateRequest model."""
        from credits_api import EstimateRequest
        
        req = EstimateRequest(
            generation_type="sdxl",
            width=1024,
            height=1024,
            steps=20,
        )
        
        assert req.generation_type == "sdxl"
        assert req.width == 1024
        assert req.height == 1024
        assert req.steps == 20

    def test_purchase_request_structure(self):
        """Test PurchaseRequest model."""
        from credits_api import PurchaseRequest
        
        req = PurchaseRequest(
            package_id="pro",
            success_url="http://localhost/success",
            cancel_url="http://localhost/cancel",
        )
        
        assert req.package_id == "pro"
        assert req.success_url == "http://localhost/success"

    @pytest.mark.asyncio
    async def test_get_balance_endpoint(self, mock_user, mock_credit_manager):
        """Test GET /api/credits endpoint."""
        from credits_api import get_balance
        
        with patch('credits_api.get_credit_manager', return_value=mock_credit_manager):
            result = await get_balance(user=mock_user)
            
            assert result.balance == 100
            assert result.lifetime_purchased == 500
            assert result.lifetime_used == 400
            
            # Verify manager was called with user ID
            mock_credit_manager.get_balance.assert_called_once_with(mock_user.id)

    @pytest.mark.asyncio
    async def test_get_packages_endpoint(self, mock_credit_manager):
        """Test GET /api/credits/packages endpoint."""
        from credits_api import get_packages
        
        with patch('credits_api.get_credit_manager', return_value=mock_credit_manager):
            packages = await get_packages()
            
            assert len(packages) == 2
            assert packages[0].id == "starter"
            assert packages[1].id == "pro"

    @pytest.mark.asyncio
    async def test_estimate_cost_endpoint(self, mock_user, mock_credit_manager):
        """Test POST /api/credits/estimate endpoint."""
        from credits_api import estimate_cost, EstimateRequest
        
        request = EstimateRequest(
            generation_type="sdxl",
            width=1024,
            height=1024,
            steps=20,
        )
        
        with patch('credits_api.get_credit_manager', return_value=mock_credit_manager):
            result = await estimate_cost(request, user=mock_user)
            
            assert result.estimated_credits > 0
            assert result.current_balance == 100
            assert result.sufficient is True  # Has 100 credits
            assert "base_type" in result.breakdown

    @pytest.mark.asyncio
    async def test_estimate_insufficient_credits(self, mock_user, mock_credit_manager):
        """Test estimate shows insufficient when balance too low."""
        from credits_api import estimate_cost, EstimateRequest
        from credits import CreditBalance
        
        # Mock low balance
        mock_credit_manager.get_balance.return_value = CreditBalance(
            balance=1,  # Very low balance
            lifetime_purchased=100,
            lifetime_used=99,
        )
        
        request = EstimateRequest(
            generation_type="wan22_i2v_hd_medium",  # Expensive
            width=1920,
            height=1080,
            duration_seconds=6,
            steps=30,
        )
        
        with patch('credits_api.get_credit_manager', return_value=mock_credit_manager):
            result = await estimate_cost(request, user=mock_user)
            
            assert result.sufficient is False  # Not enough credits
            assert result.current_balance == 1

    @pytest.mark.asyncio
    async def test_check_credits_helper(self, mock_user, mock_credit_manager):
        """Test check_credits helper function."""
        from credits_api import check_credits
        from credits import CreditBalance
        from fastapi import HTTPException
        
        # Mock sufficient balance
        mock_credit_manager.get_balance.return_value = CreditBalance(
            balance=100,
            lifetime_purchased=100,
            lifetime_used=0,
        )
        
        with patch('credits_api.get_credit_manager', return_value=mock_credit_manager):
            # Should pass with enough credits
            result = await check_credits(mock_user, required=50)
            assert result is True

    @pytest.mark.asyncio
    async def test_check_credits_insufficient(self, mock_user, mock_credit_manager):
        """Test check_credits raises 402 when insufficient."""
        from credits_api import check_credits
        from credits import CreditBalance
        from fastapi import HTTPException
        
        # Mock insufficient balance
        mock_credit_manager.get_balance.return_value = CreditBalance(
            balance=10,
            lifetime_purchased=100,
            lifetime_used=90,
        )
        
        with patch('credits_api.get_credit_manager', return_value=mock_credit_manager):
            # Should raise HTTPException with 402
            with pytest.raises(HTTPException) as exc_info:
                await check_credits(mock_user, required=50)
            
            assert exc_info.value.status_code == 402
            assert "insufficient_credits" in str(exc_info.value.detail).lower()


class TestStripeWebhook:
    """Test Stripe webhook handling."""

    @pytest.mark.asyncio
    async def test_webhook_validates_signature(self):
        """Test webhook validates Stripe signature."""
        from credits_api import stripe_webhook
        from fastapi import Request, HTTPException
        
        # Mock request with invalid signature
        mock_request = AsyncMock(spec=Request)
        mock_request.body.return_value = b'{"type": "checkout.session.completed"}'
        mock_request.headers.get.return_value = "invalid_signature"
        
        with patch.dict(os.environ, {
            "STRIPE_SECRET_KEY": "sk_test_123",
            "STRIPE_WEBHOOK_SECRET": "whsec_123",
        }):
            with patch('stripe.Webhook.construct_event') as mock_construct:
                # Make it raise a generic ValueError which the code catches
                mock_construct.side_effect = ValueError("Invalid signature")
                
                with pytest.raises(HTTPException) as exc_info:
                    await stripe_webhook(mock_request)
                
                assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_webhook_adds_credits_on_completion(self):
        """Test webhook adds credits when payment completes."""
        from credits_api import stripe_webhook
        from fastapi import Request
        
        # Mock successful payment event
        mock_event = {
            "type": "checkout.session.completed",
            "data": {
                "object": {
                    "payment_status": "paid",
                    "client_reference_id": "user-123",
                    "metadata": {
                        "package_id": "pro",
                        "credits": "1500",
                    },
                    "payment_intent": "pi_123",
                    "id": "cs_123",
                }
            }
        }
        
        mock_request = AsyncMock(spec=Request)
        mock_request.body.return_value = b'{"type": "checkout.session.completed"}'
        mock_request.headers.get.return_value = "t=123,v1=sig"
        
        mock_manager = AsyncMock()
        
        with patch.dict(os.environ, {
            "STRIPE_SECRET_KEY": "sk_test_123",
            "STRIPE_WEBHOOK_SECRET": "whsec_123",
        }):
            # Patch at the module level where it's imported
            import sys
            stripe_module = sys.modules.get('stripe')
            if not stripe_module:
                import stripe
                stripe_module = stripe
            
            with patch.object(stripe_module.Webhook, 'construct_event', return_value=mock_event):
                with patch('credits_api.get_credit_manager', return_value=mock_manager):
                    result = await stripe_webhook(mock_request)
                    
                    assert result["status"] == "ok"
                    
                    # Verify credits were added
                    mock_manager.add_credits.assert_called_once()
                    call_args = mock_manager.add_credits.call_args
                    assert call_args.kwargs["user_id"] == "user-123"
                    assert call_args.kwargs["amount"] == 1500
                    assert call_args.kwargs["type"] == "purchase"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
