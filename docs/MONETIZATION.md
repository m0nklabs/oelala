# Oelala Monetization Plan

> **Last Updated**: 2026-01-04  
> **Status**: Implementation Phase  
> **Model**: Credit-based (Pay-as-you-go)

## Business Model: Credits-Based Generation

### Waarom Credits?

Credits bieden flexibiliteit voor gebruikers die niet elke maand dezelfde hoeveelheid genereren:
- **Geen waste**: Je betaalt alleen voor wat je gebruikt
- **Geen commitment**: Geen maandelijkse abonnementen nodig
- **Schaalbaar**: Kleine gebruikers betalen weinig, power users betalen meer
- **Predictable costs**: Elke generatie heeft een vaste credit cost

---

## Credit Pricing

### Credit Packages

| Package | Credits | Price | Per Credit | Savings |
|---------|---------|-------|------------|---------|
| **Starter** | 100 | €5 | €0.050 | - |
| **Basic** | 500 | €20 | €0.040 | 20% |
| **Pro** | 1500 | €50 | €0.033 | 33% |
| **Studio** | 5000 | €150 | €0.030 | 40% |
| **Enterprise** | 20000 | €500 | €0.025 | 50% |

### Welcome Bonus
- **Nieuwe gebruikers**: 25 gratis credits bij registratie
- **Verificatie bonus**: +10 credits na e-mail verificatie

---

## Generation Costs (Credits)

### Image Generation

| Type | Resolution | Credits | Notes |
|------|------------|---------|-------|
| **SDXL** | 1024x1024 | 1 | Standard quality |
| **SDXL** | 1536x1536 | 2 | Higher resolution |
| **Flux.1** | 1024x1024 | 2 | Better quality |
| **Flux.1** | 1536x1536 | 3 | HD quality |
| **Wan2.2 T2I** | 1280x720 | 2 | Video model T2I |

### Video Generation

| Type | Duration | Resolution | Credits |
|------|----------|------------|---------|
| **Wan2.2 I2V** | 3 sec | 720p | 5 |
| **Wan2.2 I2V** | 5 sec | 720p | 8 |
| **Wan2.2 I2V** | 3 sec | 1080p | 10 |
| **Wan2.2 I2V** | 5 sec | 1080p | 15 |
| **Wan2.2 T2V** | 3 sec | 720p | 8 |
| **Wan2.2 T2V** | 5 sec | 720p | 12 |

### Audio Generation

| Type | Duration | Credits |
|------|----------|---------|
| **MMAudio** | <10 sec | 3 |
| **MMAudio** | 10-30 sec | 5 |
| **Voice Clone** | per voice | 20 |

---

## Credit System Architecture

### Database Schema (Supabase)

```sql
-- User credit balance
CREATE TABLE user_credits (
    user_id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    balance INTEGER NOT NULL DEFAULT 0,
    lifetime_purchased INTEGER NOT NULL DEFAULT 0,
    lifetime_used INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Credit transactions log
CREATE TABLE credit_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    amount INTEGER NOT NULL,  -- Positive = add, negative = deduct
    type TEXT NOT NULL CHECK (type IN ('purchase', 'bonus', 'generation', 'refund', 'admin')),
    description TEXT,
    reference_id TEXT,  -- Stripe payment ID or job ID
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Credit packages for purchase
CREATE TABLE credit_packages (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    credits INTEGER NOT NULL,
    price_cents INTEGER NOT NULL,
    currency TEXT DEFAULT 'EUR',
    stripe_price_id TEXT,
    is_active BOOLEAN DEFAULT true,
    sort_order INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_credit_transactions_user ON credit_transactions(user_id);
CREATE INDEX idx_credit_transactions_created ON credit_transactions(created_at DESC);
CREATE INDEX idx_credit_transactions_type ON credit_transactions(type);

-- RLS Policies
ALTER TABLE user_credits ENABLE ROW LEVEL SECURITY;
ALTER TABLE credit_transactions ENABLE ROW LEVEL SECURITY;

-- Users can only read their own credits
CREATE POLICY "Users can view own credits" ON user_credits
    FOR SELECT USING (auth.uid() = user_id);

-- Users can only read their own transactions
CREATE POLICY "Users can view own transactions" ON credit_transactions
    FOR SELECT USING (auth.uid() = user_id);

-- Credit packages are public (read-only)
CREATE POLICY "Anyone can view packages" ON credit_packages
    FOR SELECT USING (is_active = true);

-- Trigger to auto-create user_credits on signup
CREATE OR REPLACE FUNCTION create_user_credits()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO public.user_credits (user_id, balance)
    VALUES (NEW.id, 25)  -- 25 welcome credits
    ON CONFLICT (user_id) DO NOTHING;
    
    -- Log welcome bonus
    INSERT INTO public.credit_transactions (user_id, amount, type, description)
    VALUES (NEW.id, 25, 'bonus', 'Welcome bonus credits');
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION create_user_credits();

-- Insert default packages
INSERT INTO credit_packages (id, name, credits, price_cents, currency, sort_order) VALUES
    ('starter', 'Starter', 100, 500, 'EUR', 1),
    ('basic', 'Basic', 500, 2000, 'EUR', 2),
    ('pro', 'Pro', 1500, 5000, 'EUR', 3),
    ('studio', 'Studio', 5000, 15000, 'EUR', 4),
    ('enterprise', 'Enterprise', 20000, 50000, 'EUR', 5)
ON CONFLICT (id) DO UPDATE SET
    credits = EXCLUDED.credits,
    price_cents = EXCLUDED.price_cents;
```

---

## API Endpoints

### Credit Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/credits` | Get current balance |
| GET | `/api/credits/packages` | List available packages |
| POST | `/api/credits/purchase` | Initiate Stripe checkout |
| GET | `/api/credits/history` | Transaction history |
| POST | `/api/credits/estimate` | Estimate generation cost |

### Response Examples

```json
// GET /api/credits
{
    "balance": 150,
    "lifetime_purchased": 500,
    "lifetime_used": 350
}

// GET /api/credits/packages
{
    "packages": [
        {"id": "starter", "name": "Starter", "credits": 100, "price_cents": 500, "currency": "EUR"},
        {"id": "basic", "name": "Basic", "credits": 500, "price_cents": 2000, "currency": "EUR"}
    ]
}

// POST /api/credits/estimate
// Request:
{
    "generation_type": "wan22_i2v",
    "params": {"duration_seconds": 5, "width": 1920, "height": 1080}
}
// Response:
{
    "estimated_credits": 15,
    "breakdown": {
        "base": 10,
        "hd_multiplier": 1.5
    },
    "current_balance": 150,
    "sufficient": true
}
```

---

## Credit Deduction Flow

```
1. User initiates generation
          ↓
2. Calculate required credits
          ↓
3. Check balance (user_credits.balance >= required)
          ↓ (insufficient?)
          → Return 402 Payment Required
          ↓ (sufficient)
4. Atomic credit reserve (UPDATE ... SET balance = balance - X WHERE balance >= X)
          ↓ (race condition / insufficient)
          → Return 402
          ↓ (reserved)
5. Start generation job
          ↓
6. On completion:
          ↓ (success)
          → Log transaction (type='generation')
          ↓ (failure)
          → Refund credits + log (type='refund')
```

---

## Payment Integration (Stripe)

### Checkout Flow

1. User clicks "Buy Credits" → selects package
2. Frontend: `POST /api/credits/purchase` with `package_id`
3. Backend creates Stripe Checkout Session
4. User redirected to Stripe payment page
5. After payment → Stripe webhook → credits added
6. User redirected back with updated balance

### Webhook Handler

```python
@app.post("/webhooks/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")
    event = stripe.Webhook.construct_event(payload, sig_header, WEBHOOK_SECRET)
    
    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session["client_reference_id"]
        package_id = session["metadata"]["package_id"]
        
        package = await get_package(package_id)
        await credit_manager.add_credits(
            user_id=user_id,
            amount=package.credits,
            type="purchase",
            reference_id=session["payment_intent"],
            description=f"Purchased {package.name} package"
        )
    
    return {"status": "ok"}
```

---

## Free Tier & Limits

### Welcome Credits
| Trigger | Credits | Notes |
|---------|---------|-------|
| Account creation | 25 | Automatic |
| Email verification | 10 | Optional bonus |
| **Total** | **35** | Enough for ~5 videos |

### Rate Limits (Anti-abuse)

| Limit | Value | Reason |
|-------|-------|--------|
| Generations/minute | 5 | Prevent spam |
| Generations/hour | 30 | Fair usage |
| Concurrent jobs | 2 | Queue fairness |
| Max daily spend | 500 credits | Fraud protection |

---

## Implementation Status

### Phase 1: Core Credits
- [x] Database schema design (this doc)
- [ ] `src/backend/credits.py` - CreditManager module
- [ ] Credit API endpoints
- [ ] Generation cost calculation
- [ ] Credit deduction middleware

### Phase 2: Payments
- [ ] Stripe account setup
- [ ] Checkout flow
- [ ] Webhook handlers
- [ ] Receipt emails

### Phase 3: Frontend
- [ ] Credits display in UI
- [ ] Purchase modal
- [ ] Transaction history
- [ ] Low balance warnings

---

## Related Documents

- [ROADMAP.md](./ROADMAP.md) - Product roadmap
- [MEDIA_STORAGE.md](./MEDIA_STORAGE.md) - Storage architecture  
- [ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture
- [MONETIZATION_OLD.md](./MONETIZATION_OLD.md) - Previous subscription model (archived)
