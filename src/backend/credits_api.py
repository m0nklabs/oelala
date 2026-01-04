"""
Oelala Credits API Routes
FastAPI endpoints for credit management.
"""

import os
import logging
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Depends, Request
from pydantic import BaseModel

from auth import get_current_user, User
from credits import (
    CreditManager,
    get_credit_manager,
    CreditBalance,
    CreditPackageResponse,
    CreditEstimate,
    CreditTransaction,
    InsufficientCreditsError,
    calculate_credits,
    DEFAULT_PACKAGES,
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/credits", tags=["credits"])

# Debug flag
DEBUG = os.getenv("OELALA_DEBUG", "0") == "1"


def debug_log(msg: str):
    if DEBUG:
        logger.info(f"💰 CREDITS-API: {msg}")


# =============================================================================
# Pydantic Models for Requests
# =============================================================================

class EstimateRequest(BaseModel):
    """Request body for cost estimation."""
    generation_type: str
    width: int = 1024
    height: int = 1024
    duration_seconds: Optional[int] = None
    steps: int = 20


class PurchaseRequest(BaseModel):
    """Request body for initiating purchase."""
    package_id: str
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None


class PurchaseResponse(BaseModel):
    """Response with Stripe checkout URL."""
    checkout_url: str
    session_id: str


# =============================================================================
# Endpoints
# =============================================================================

@router.get("", response_model=CreditBalance)
async def get_balance(user: User = Depends(get_current_user)):
    """
    Get current user's credit balance.
    
    Returns balance, lifetime purchased, and lifetime used.
    """
    manager = get_credit_manager()
    try:
        balance = await manager.get_balance(user.id)
        debug_log(f"Balance for {user.id}: {balance.balance}")
        return balance
    except Exception as e:
        logger.error(f"Error getting balance: {e}")
        raise HTTPException(status_code=500, detail="Failed to get credit balance")


@router.get("/packages", response_model=List[CreditPackageResponse])
async def get_packages():
    """
    Get available credit packages for purchase.
    
    Returns list of packages with credits and prices.
    """
    manager = get_credit_manager()
    try:
        packages = await manager.get_packages()
        return packages
    except Exception as e:
        logger.error(f"Error getting packages: {e}")
        # Return defaults on error
        return [
            CreditPackageResponse(
                id=p.id,
                name=p.name,
                credits=p.credits,
                price_cents=p.price_cents,
                currency=p.currency,
            )
            for p in DEFAULT_PACKAGES
        ]


@router.post("/estimate", response_model=CreditEstimate)
async def estimate_cost(
    request: EstimateRequest,
    user: User = Depends(get_current_user),
):
    """
    Estimate credit cost for a generation job.
    
    Helps users understand how many credits will be used before generating.
    """
    manager = get_credit_manager()
    
    # Calculate cost
    estimated = calculate_credits(
        generation_type=request.generation_type,
        width=request.width,
        height=request.height,
        duration_seconds=request.duration_seconds,
        steps=request.steps,
    )
    
    # Get current balance
    balance = await manager.get_balance(user.id)
    
    # Build breakdown for transparency
    breakdown = {
        "base_type": request.generation_type,
        "base_cost": estimated,
    }
    
    if request.width > 1280 or request.height > 720:
        breakdown["hd_upcharge"] = True
    if request.duration_seconds and request.duration_seconds > 5:
        breakdown["duration_multiplier"] = True
    
    return CreditEstimate(
        estimated_credits=estimated,
        breakdown=breakdown,
        current_balance=balance.balance,
        sufficient=balance.balance >= estimated,
    )


@router.get("/history", response_model=List[CreditTransaction])
async def get_transaction_history(
    limit: int = 50,
    offset: int = 0,
    user: User = Depends(get_current_user),
):
    """
    Get user's credit transaction history.
    
    Returns recent transactions with type, amount, and description.
    """
    manager = get_credit_manager()
    
    if limit > 100:
        limit = 100  # Cap at 100
    
    try:
        transactions = await manager.get_transactions(user.id, limit, offset)
        return transactions
    except Exception as e:
        logger.error(f"Error getting transactions: {e}")
        return []


@router.post("/purchase", response_model=PurchaseResponse)
async def initiate_purchase(
    request: PurchaseRequest,
    user: User = Depends(get_current_user),
):
    """
    Initiate a credit purchase via Stripe Checkout.
    
    Returns a checkout URL to redirect the user to Stripe.
    """
    import stripe
    
    STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
    if not STRIPE_SECRET_KEY:
        raise HTTPException(
            status_code=503, 
            detail="Payment system not configured"
        )
    
    stripe.api_key = STRIPE_SECRET_KEY
    
    # Find the package
    manager = get_credit_manager()
    packages = await manager.get_packages()
    package = next((p for p in packages if p.id == request.package_id), None)
    
    if not package:
        raise HTTPException(status_code=404, detail="Package not found")
    
    # Default URLs
    base_url = os.getenv("FRONTEND_URL", "http://localhost:5174")
    success_url = request.success_url or f"{base_url}/credits?success=true"
    cancel_url = request.cancel_url or f"{base_url}/credits?cancelled=true"
    
    try:
        # Create Stripe Checkout session
        session = stripe.checkout.Session.create(
            mode="payment",
            payment_method_types=["card", "ideal"],  # Cards + iDEAL for NL
            line_items=[
                {
                    "price_data": {
                        "currency": package.currency.lower(),
                        "unit_amount": package.price_cents,
                        "product_data": {
                            "name": f"Oelala {package.name} Pack",
                            "description": f"{package.credits} generation credits",
                        },
                    },
                    "quantity": 1,
                },
            ],
            client_reference_id=user.id,  # To identify user in webhook
            metadata={
                "package_id": package.id,
                "credits": str(package.credits),
                "user_id": user.id,
            },
            success_url=success_url + "&session_id={CHECKOUT_SESSION_ID}",
            cancel_url=cancel_url,
        )
        
        debug_log(f"Created checkout session {session.id} for user {user.id}")
        
        return PurchaseResponse(
            checkout_url=session.url,
            session_id=session.id,
        )
        
    except stripe.error.StripeError as e:
        logger.error(f"Stripe error: {e}")
        raise HTTPException(status_code=500, detail="Payment system error")


# =============================================================================
# Webhook for Stripe (no auth required)
# =============================================================================

@router.post("/webhook/stripe")
async def stripe_webhook(request: Request):
    """
    Handle Stripe webhook events.
    
    Called by Stripe when payment completes. Adds credits to user account.
    """
    import stripe
    
    STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
    STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    
    if not STRIPE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Payment system not configured")
    
    stripe.api_key = STRIPE_SECRET_KEY
    
    # Get the raw body
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature", "")
    
    try:
        if STRIPE_WEBHOOK_SECRET:
            event = stripe.Webhook.construct_event(
                payload, sig_header, STRIPE_WEBHOOK_SECRET
            )
        else:
            # No webhook secret configured - parse directly (not recommended for production)
            import json
            event = json.loads(payload)
            logger.warning("⚠️ Stripe webhook secret not configured - signature not verified")
    except ValueError as e:
        logger.error(f"Invalid payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload")
    except stripe.error.SignatureVerificationError as e:
        logger.error(f"Invalid signature: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")
    
    # Handle checkout completion
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        
        user_id = session.get("client_reference_id")
        metadata = session.get("metadata", {})
        package_id = metadata.get("package_id")
        credits = int(metadata.get("credits", 0))
        payment_intent = session.get("payment_intent")
        
        if not user_id or not credits:
            logger.error(f"Missing data in webhook: user_id={user_id}, credits={credits}")
            return {"status": "error", "message": "Missing user or credits"}
        
        # Add credits to user account
        manager = get_credit_manager()
        try:
            await manager.add_credits(
                user_id=user_id,
                amount=credits,
                type="purchase",
                description=f"Purchased {package_id} package ({credits} credits)",
                reference_id=payment_intent,
                metadata={"package_id": package_id, "session_id": session["id"]},
            )
            logger.info(f"✅ Added {credits} credits to user {user_id} (payment: {payment_intent})")
        except Exception as e:
            logger.error(f"Failed to add credits: {e}")
            # Stripe will retry if we return error
            raise HTTPException(status_code=500, detail="Failed to add credits")
    
    return {"status": "ok"}


# =============================================================================
# Credit Check Middleware Helper
# =============================================================================

async def check_credits(user: User, required: int) -> bool:
    """
    Check if user has enough credits.
    
    Raises HTTPException with 402 if insufficient.
    Returns True if sufficient.
    """
    manager = get_credit_manager()
    balance = await manager.get_balance(user.id)
    
    if balance.balance < required:
        packages = await manager.get_packages()
        raise HTTPException(
            status_code=402,
            detail={
                "error": "insufficient_credits",
                "required": required,
                "available": balance.balance,
                "packages": [p.dict() for p in packages],
            },
        )
    
    return True


async def deduct_credits(
    user: User,
    amount: int,
    job_id: str,
    generation_type: str,
) -> bool:
    """
    Deduct credits after successful generation.
    
    Call this after job completes successfully.
    """
    manager = get_credit_manager()
    try:
        await manager.deduct(
            user_id=user.id,
            amount=amount,
            reference_id=job_id,
            description=f"Generated {generation_type}",
            metadata={"generation_type": generation_type, "job_id": job_id},
        )
        return True
    except Exception as e:
        logger.error(f"Failed to deduct credits: {e}")
        return False


async def refund_credits(
    user: User,
    amount: int,
    job_id: str,
    reason: str = "Generation failed",
) -> bool:
    """
    Refund credits for failed generation.
    """
    manager = get_credit_manager()
    try:
        await manager.refund(
            user_id=user.id,
            amount=amount,
            reference_id=job_id,
            reason=reason,
        )
        logger.info(f"🔄 Refunded {amount} credits to user {user.id} for job {job_id}")
        return True
    except Exception as e:
        logger.error(f"Failed to refund credits: {e}")
        return False
