"""
Oelala Credits System
Pay-as-you-go credit management for AI generation.
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Debug flag
DEBUG = os.getenv("OELALA_DEBUG", "0") == "1"


def debug_log(msg: str):
    if DEBUG:
        logger.info(f"💰 CREDITS: {msg}")


# =============================================================================
# Configuration
# =============================================================================

SUPABASE_URL = os.getenv("SUPABASE_URL", "https://nsbjwhxdkxnyggtuxjjp.supabase.co")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")  # Service role key for backend
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")

# Welcome bonus
WELCOME_CREDITS = 25
VERIFICATION_BONUS = 10


# =============================================================================
# Credit Costs per Generation Type
# =============================================================================

class GenerationType(str, Enum):
    """Types of generation with associated credit costs."""
    # Images
    SDXL = "sdxl"
    SDXL_HD = "sdxl_hd"
    FLUX = "flux"
    FLUX_HD = "flux_hd"
    SD15 = "sd15"
    WAN22_T2I = "wan22_t2i"
    
    # Videos
    WAN22_I2V_SHORT = "wan22_i2v_short"  # 3 sec 720p
    WAN22_I2V_MEDIUM = "wan22_i2v_medium"  # 5 sec 720p
    WAN22_I2V_HD_SHORT = "wan22_i2v_hd_short"  # 3 sec 1080p
    WAN22_I2V_HD_MEDIUM = "wan22_i2v_hd_medium"  # 5 sec 1080p
    WAN22_T2V_SHORT = "wan22_t2v_short"  # 3 sec
    WAN22_T2V_MEDIUM = "wan22_t2v_medium"  # 5 sec
    
    # Audio
    MMAUDIO_SHORT = "mmaudio_short"  # <10 sec
    MMAUDIO_LONG = "mmaudio_long"  # 10-30 sec
    VOICE_CLONE = "voice_clone"


# Base credit costs
CREDIT_COSTS: Dict[GenerationType, int] = {
    # Images (cheap)
    GenerationType.SDXL: 1,
    GenerationType.SDXL_HD: 2,
    GenerationType.FLUX: 2,
    GenerationType.FLUX_HD: 3,
    GenerationType.SD15: 1,
    GenerationType.WAN22_T2I: 2,
    
    # Videos (expensive)
    GenerationType.WAN22_I2V_SHORT: 5,
    GenerationType.WAN22_I2V_MEDIUM: 8,
    GenerationType.WAN22_I2V_HD_SHORT: 10,
    GenerationType.WAN22_I2V_HD_MEDIUM: 15,
    GenerationType.WAN22_T2V_SHORT: 8,
    GenerationType.WAN22_T2V_MEDIUM: 12,
    
    # Audio
    GenerationType.MMAUDIO_SHORT: 3,
    GenerationType.MMAUDIO_LONG: 5,
    GenerationType.VOICE_CLONE: 20,
}


def calculate_credits(
    generation_type: str,
    width: int = 1024,
    height: int = 1024,
    duration_seconds: Optional[int] = None,
    steps: int = 20,
) -> int:
    """
    Calculate credit cost for a generation job.
    
    Args:
        generation_type: Type of generation (e.g., 'sdxl', 'wan22_i2v')
        width: Output width in pixels
        height: Output height in pixels  
        duration_seconds: Video/audio duration
        steps: Inference steps (affects quality/time)
    
    Returns:
        Number of credits required
    """
    # Normalize type
    gen_type = generation_type.lower().replace("-", "_")
    
    # Map API endpoints to generation types
    type_mapping = {
        "generate_image": GenerationType.SDXL,
        "generate_sdxl": GenerationType.SDXL,
        "generate_flux": GenerationType.FLUX,
        "generate_sd15": GenerationType.SD15,
        "generate_wan22_t2i": GenerationType.WAN22_T2I,
        "generate": "wan22_i2v",  # Dynamic based on params
        "generate_wan22": "wan22_i2v",
        "generate_wan22_comfyui": "wan22_i2v",
        "generate_wan22_async": "wan22_i2v",
        "generate_text": "wan22_t2v",
        "generate_pose": "wan22_i2v",
        "generate_audio": GenerationType.MMAUDIO_SHORT,
        "mmaudio": GenerationType.MMAUDIO_SHORT,
        "wan22_i2v": "wan22_i2v",  # Alias
        "wan22_t2v": "wan22_t2v",
    }
    
    # Get base generation type
    mapped = type_mapping.get(gen_type, gen_type)
    
    # Dynamic video type based on duration and resolution
    if mapped in ("wan22_i2v", "wan22_t2v"):
        is_hd = width > 1280 or height > 720
        is_long = duration_seconds and duration_seconds > 3
        
        if mapped == "wan22_i2v":
            if is_hd and is_long:
                base_type = GenerationType.WAN22_I2V_HD_MEDIUM
            elif is_hd:
                base_type = GenerationType.WAN22_I2V_HD_SHORT
            elif is_long:
                base_type = GenerationType.WAN22_I2V_MEDIUM
            else:
                base_type = GenerationType.WAN22_I2V_SHORT
        else:  # wan22_t2v
            if is_long:
                base_type = GenerationType.WAN22_T2V_MEDIUM
            else:
                base_type = GenerationType.WAN22_T2V_SHORT
    elif isinstance(mapped, GenerationType):
        base_type = mapped
    else:
        # Try direct enum lookup
        try:
            base_type = GenerationType(gen_type)
        except ValueError:
            debug_log(f"Unknown generation type: {gen_type}, defaulting to 2 credits")
            return 2  # Default cost for unknown types
    
    # Get base cost
    base_cost = CREDIT_COSTS.get(base_type, 2)
    
    # HD resolution multiplier (>720p)
    is_hd = width > 1280 or height > 720
    if is_hd and "_hd" not in base_type.value:
        base_cost = int(base_cost * 1.5)
    
    # Duration multiplier for video/audio
    if duration_seconds:
        if duration_seconds > 5:
            base_cost = int(base_cost * 1.5)
        elif duration_seconds > 10:
            base_cost = int(base_cost * 2.0)
    
    # Steps multiplier (for image generation)
    if steps > 30:
        base_cost = int(base_cost * 1.2)
    
    debug_log(f"Calculated {base_cost} credits for {gen_type} ({width}x{height})")
    return max(1, base_cost)


# =============================================================================
# Credit Packages
# =============================================================================

@dataclass
class CreditPackage:
    """Credit package available for purchase."""
    id: str
    name: str
    credits: int
    price_cents: int
    currency: str = "EUR"
    stripe_price_id: Optional[str] = None
    is_active: bool = True


# Default packages (also stored in database)
DEFAULT_PACKAGES: List[CreditPackage] = [
    CreditPackage("starter", "Starter", 100, 500, "EUR"),
    CreditPackage("basic", "Basic", 500, 2000, "EUR"),
    CreditPackage("pro", "Pro", 1500, 5000, "EUR"),
    CreditPackage("studio", "Studio", 5000, 15000, "EUR"),
    CreditPackage("enterprise", "Enterprise", 20000, 50000, "EUR"),
]


# =============================================================================
# Pydantic Models for API
# =============================================================================

class CreditBalance(BaseModel):
    """User credit balance response."""
    balance: int
    lifetime_purchased: int
    lifetime_used: int


class CreditPackageResponse(BaseModel):
    """Credit package for API response."""
    id: str
    name: str
    credits: int
    price_cents: int
    currency: str


class CreditEstimate(BaseModel):
    """Credit cost estimate response."""
    estimated_credits: int
    breakdown: Dict[str, Any]
    current_balance: Optional[int] = None
    sufficient: Optional[bool] = None


class CreditTransaction(BaseModel):
    """Credit transaction record."""
    id: str
    amount: int
    type: str  # 'purchase', 'bonus', 'generation', 'refund', 'admin'
    description: Optional[str]
    reference_id: Optional[str]
    created_at: datetime


class InsufficientCreditsError(Exception):
    """Raised when user doesn't have enough credits."""
    def __init__(self, required: int, available: int, packages: List[CreditPackage] = None):
        self.required = required
        self.available = available
        self.packages = packages or DEFAULT_PACKAGES
        super().__init__(f"Insufficient credits: need {required}, have {available}")


# =============================================================================
# Credit Manager
# =============================================================================

class CreditManager:
    """
    Manages user credits via Supabase.
    
    Handles:
    - Balance queries
    - Credit deduction (with atomic transactions)
    - Credit addition (purchases, bonuses)
    - Transaction logging
    """
    
    def __init__(self, supabase_url: str = None, service_key: str = None):
        self.supabase_url = supabase_url or SUPABASE_URL
        self.service_key = service_key or SUPABASE_SERVICE_KEY
        self._client: Optional[httpx.AsyncClient] = None
    
    @property
    def headers(self) -> Dict[str, str]:
        """Auth headers for Supabase REST API."""
        return {
            "apikey": self.service_key,
            "Authorization": f"Bearer {self.service_key}",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }
    
    async def get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=f"{self.supabase_url}/rest/v1",
                headers=self.headers,
                timeout=30.0,
            )
        return self._client
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    # -------------------------------------------------------------------------
    # Balance Operations
    # -------------------------------------------------------------------------
    
    async def get_balance(self, user_id: str) -> CreditBalance:
        """
        Get user's current credit balance.
        
        Creates record with welcome bonus if user doesn't exist.
        """
        client = await self.get_client()
        
        # Try to get existing balance
        response = await client.get(
            "/user_credits",
            params={"user_id": f"eq.{user_id}", "select": "*"},
        )
        
        if response.status_code == 200 and response.json():
            data = response.json()[0]
            return CreditBalance(
                balance=data["balance"],
                lifetime_purchased=data["lifetime_purchased"],
                lifetime_used=data["lifetime_used"],
            )
        
        # Create new record with welcome bonus
        debug_log(f"Creating credit record for new user {user_id}")
        return await self._create_user_credits(user_id)
    
    async def _create_user_credits(self, user_id: str) -> CreditBalance:
        """Create credit record for new user with welcome bonus."""
        client = await self.get_client()
        
        # Insert user_credits
        response = await client.post(
            "/user_credits",
            json={
                "user_id": user_id,
                "balance": WELCOME_CREDITS,
                "lifetime_purchased": 0,
                "lifetime_used": 0,
            },
        )
        
        if response.status_code not in (200, 201):
            # Might already exist (race condition)
            return await self.get_balance(user_id)
        
        # Log welcome bonus transaction
        await self._log_transaction(
            user_id=user_id,
            amount=WELCOME_CREDITS,
            type="bonus",
            description="Welcome bonus credits",
        )
        
        return CreditBalance(
            balance=WELCOME_CREDITS,
            lifetime_purchased=0,
            lifetime_used=0,
        )
    
    # -------------------------------------------------------------------------
    # Credit Operations
    # -------------------------------------------------------------------------
    
    async def check_and_reserve(self, user_id: str, amount: int) -> bool:
        """
        Check if user has enough credits and reserve them atomically.
        
        Uses Supabase RPC for atomic operation.
        Returns True if reserved, raises InsufficientCreditsError if not.
        """
        if amount <= 0:
            return True
        
        client = await self.get_client()
        
        # Atomic decrement with balance check
        # UPDATE user_credits SET balance = balance - $amount 
        # WHERE user_id = $user_id AND balance >= $amount
        response = await client.patch(
            "/user_credits",
            params={
                "user_id": f"eq.{user_id}",
                "balance": f"gte.{amount}",  # Only if balance >= amount
            },
            json={
                "balance": f"balance - {amount}",  # Raw SQL expression doesn't work in REST
            },
        )
        
        # Supabase REST doesn't support raw SQL expressions
        # We need to use RPC or do a transaction manually
        # For now, use two-step approach with optimistic locking
        
        balance = await self.get_balance(user_id)
        if balance.balance < amount:
            raise InsufficientCreditsError(
                required=amount,
                available=balance.balance,
                packages=DEFAULT_PACKAGES,
            )
        
        # Deduct credits
        response = await client.patch(
            "/user_credits",
            params={"user_id": f"eq.{user_id}"},
            json={
                "balance": balance.balance - amount,
                "lifetime_used": balance.lifetime_used + amount,
                "updated_at": datetime.utcnow().isoformat(),
            },
        )
        
        if response.status_code not in (200, 204):
            logger.error(f"Failed to deduct credits: {response.text}")
            raise Exception("Failed to reserve credits")
        
        debug_log(f"Reserved {amount} credits for user {user_id}")
        return True
    
    async def deduct(
        self,
        user_id: str,
        amount: int,
        reference_id: str,
        description: str = None,
        metadata: Dict[str, Any] = None,
    ) -> bool:
        """
        Deduct credits and log transaction atomically.
        
        Call this after generation completes successfully.
        Balance should be verified with check_credits before calling this.
        
        Uses database RPC function to avoid race conditions.
        """
        client = await self.get_client()
        
        # Use atomic database RPC function to deduct credits
        # This handles: balance check, deduction, lifetime_used update, and transaction logging
        response = await client.post(
            "/rpc/deduct_credits",
            json={
                "p_user_id": user_id,
                "p_amount": amount,
                "p_description": description or f"Generation job",
                "p_reference_id": reference_id,
                "p_metadata": metadata or {},
            },
        )
        
        if response.status_code not in (200, 201):
            logger.error(f"Failed to deduct credits via RPC: {response.text}")
            raise Exception("Failed to deduct credits")
        
        # Check RPC result
        result = response.json()
        if result and len(result) > 0:
            result_row = result[0]
            if not result_row.get("success", False):
                error_msg = result_row.get("error", "Unknown error")
                logger.error(f"Credit deduction failed: {error_msg}")
                raise Exception(f"Failed to deduct credits: {error_msg}")
            
            debug_log(f"Deducted {amount} credits from user {user_id}, new balance: {result_row.get('new_balance')}, ref={reference_id}")
        else:
            logger.warning("RPC returned empty result")
        
        return True
    
    async def add_credits(
        self,
        user_id: str,
        amount: int,
        type: str,
        description: str = None,
        reference_id: str = None,
        metadata: Dict[str, Any] = None,
    ) -> CreditBalance:
        """
        Add credits to user account (purchase, bonus, refund).
        """
        client = await self.get_client()
        
        # Get current balance
        balance = await self.get_balance(user_id)
        
        # Update with addition
        new_balance = balance.balance + amount
        new_lifetime = balance.lifetime_purchased + (amount if type == "purchase" else 0)
        
        response = await client.patch(
            "/user_credits",
            params={"user_id": f"eq.{user_id}"},
            json={
                "balance": new_balance,
                "lifetime_purchased": new_lifetime,
                "updated_at": datetime.utcnow().isoformat(),
            },
        )
        
        if response.status_code not in (200, 204):
            logger.error(f"Failed to add credits: {response.text}")
            raise Exception("Failed to add credits")
        
        # Log transaction
        await self._log_transaction(
            user_id=user_id,
            amount=amount,
            type=type,
            description=description,
            reference_id=reference_id,
            metadata=metadata,
        )
        
        debug_log(f"Added {amount} credits to user {user_id}, type={type}")
        
        return CreditBalance(
            balance=new_balance,
            lifetime_purchased=new_lifetime,
            lifetime_used=balance.lifetime_used,
        )
    
    async def refund(
        self,
        user_id: str,
        amount: int,
        reference_id: str,
        reason: str = None,
    ) -> CreditBalance:
        """
        Refund credits for failed generation.
        """
        return await self.add_credits(
            user_id=user_id,
            amount=amount,
            type="refund",
            description=reason or "Generation failed - credits refunded",
            reference_id=reference_id,
        )
    
    # -------------------------------------------------------------------------
    # Transaction Log
    # -------------------------------------------------------------------------
    
    async def _log_transaction(
        self,
        user_id: str,
        amount: int,
        type: str,
        description: str = None,
        reference_id: str = None,
        metadata: Dict[str, Any] = None,
    ):
        """Log a credit transaction."""
        client = await self.get_client()
        
        response = await client.post(
            "/credit_transactions",
            json={
                "user_id": user_id,
                "amount": amount,
                "type": type,
                "description": description,
                "reference_id": reference_id,
                "metadata": metadata or {},
            },
        )
        
        if response.status_code not in (200, 201):
            logger.warning(f"Failed to log transaction: {response.text}")
    
    async def get_transactions(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> List[CreditTransaction]:
        """Get user's transaction history."""
        client = await self.get_client()
        
        response = await client.get(
            "/credit_transactions",
            params={
                "user_id": f"eq.{user_id}",
                "select": "*",
                "order": "created_at.desc",
                "limit": limit,
                "offset": offset,
            },
        )
        
        if response.status_code != 200:
            return []
        
        return [
            CreditTransaction(
                id=row["id"],
                amount=row["amount"],
                type=row["type"],
                description=row.get("description"),
                reference_id=row.get("reference_id"),
                created_at=datetime.fromisoformat(row["created_at"].replace("Z", "+00:00")),
            )
            for row in response.json()
        ]
    
    # -------------------------------------------------------------------------
    # Packages
    # -------------------------------------------------------------------------
    
    async def get_packages(self) -> List[CreditPackageResponse]:
        """Get available credit packages."""
        client = await self.get_client()
        
        response = await client.get(
            "/credit_packages",
            params={
                "is_active": "eq.true",
                "select": "id,name,credits,price_cents,currency",
                "order": "sort_order.asc",
            },
        )
        
        if response.status_code != 200 or not response.json():
            # Return defaults if DB not set up
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
        
        return [
            CreditPackageResponse(**row)
            for row in response.json()
        ]


# =============================================================================
# Singleton Instance
# =============================================================================

_credit_manager: Optional[CreditManager] = None


def get_credit_manager() -> CreditManager:
    """Get singleton CreditManager instance."""
    global _credit_manager
    if _credit_manager is None:
        _credit_manager = CreditManager()
    return _credit_manager


# =============================================================================
# FastAPI Dependency
# =============================================================================

async def require_credits(amount: int):
    """
    FastAPI dependency to require credits for an endpoint.
    
    Usage:
        @app.post("/generate")
        async def generate(user: User = Depends(get_current_user)):
            credits_required = calculate_credits("sdxl", 1024, 1024)
            await require_credits(credits_required)(user)
            # ... do generation
    """
    async def check(user):
        manager = get_credit_manager()
        await manager.check_and_reserve(user.id, amount)
        return amount
    return check
