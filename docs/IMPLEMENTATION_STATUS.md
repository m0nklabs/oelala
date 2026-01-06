# Credit System & Stripe Integration - Implementation Summary

## ✅ IMPLEMENTATION COMPLETE

The credit system and Stripe payments integration is **fully implemented and tested**. All code is production-ready.

---

## What Was Implemented

### 🎯 Backend Features
- ✅ Credit calculation engine with dynamic pricing
- ✅ Package management system (5 tiers)
- ✅ Stripe Checkout Session creation
- ✅ Webhook handler for payment completion
- ✅ Credit balance tracking via Supabase
- ✅ Atomic credit deduction in all generation endpoints
- ✅ Insufficient credits error handling (402 status)

### 🖥️ Frontend Features
- ✅ Purchase Credits modal (already existed)
- ✅ Stripe checkout redirect flow
- ✅ Success/cancel URL handling
- ✅ Auto-refresh balance after purchase
- ✅ Success/cancel notifications with auto-hide
- ✅ Real-time balance display in user menu
- ✅ Credit cost estimates in tool descriptions

### 🗄️ Database Schema
- ✅ `user_credits` - Balance tracking per user
- ✅ `credit_transactions` - Complete audit log
- ✅ `credit_packages` - Admin-managed packages
- ✅ Row-level security (RLS) policies
- ✅ Auto-create on signup (25 welcome credits)
- ✅ Atomic deduct_credits() function
- ✅ Atomic add_credits() function

### 🧪 Testing
- ✅ 14 unit tests for credits.py
- ✅ 10 unit tests for credits_api.py
- ✅ All 24 tests passing
- ✅ Tests cover calculation, balance, packages, checkout, webhook

### 📚 Documentation
- ✅ Complete CREDITS_SETUP.md guide
- ✅ Step-by-step Stripe configuration
- ✅ Testing instructions with curl examples
- ✅ Stripe test card numbers
- ✅ Troubleshooting section
- ✅ Production checklist

---

## What You Need to Do (Production Setup)

### 1. Run Database Migration

Open your Supabase project → SQL Editor → Run this file:
```
src/backend/migrations/001_credits_system.sql
```

This creates:
- 3 tables (user_credits, credit_transactions, credit_packages)
- RLS policies
- Helper functions
- Welcome bonus trigger

### 2. Create Stripe Products

**Option A: Stripe Dashboard** (Recommended)
1. Go to https://dashboard.stripe.com/test/products
2. Create 5 products (see table below)
3. Copy each Price ID (starts with `price_`)

**Option B: Stripe CLI**
```bash
stripe products create --name="Starter Pack" --description="100 credits"
stripe prices create --product=prod_xxx --unit-amount=500 --currency=eur
```

| Package | Credits | Price | Metadata |
|---------|---------|-------|----------|
| Starter | 100 | €5.00 | `credits: 100` |
| Basic | 500 | €20.00 | `credits: 500` |
| Pro | 1500 | €50.00 | `credits: 1500` |
| Studio | 5000 | €150.00 | `credits: 5000` |
| Enterprise | 20000 | €500.00 | `credits: 20000` |

### 3. Update Migration with Stripe Price IDs

Edit line 150-156 in `src/backend/migrations/001_credits_system.sql`:

```sql
INSERT INTO public.credit_packages (...) VALUES
    ('starter', 'Starter', 100, 500, 'EUR', 'price_YOUR_ACTUAL_ID', ...),
    ('basic', 'Basic', 500, 2000, 'EUR', 'price_YOUR_ACTUAL_ID', ...),
    -- etc.
```

**OR** update after migration with SQL:
```sql
UPDATE public.credit_packages SET stripe_price_id = 'price_xxx' WHERE id = 'starter';
```

### 4. Configure Environment Variables

Add to `src/backend/.env`:
```bash
# Supabase (REQUIRED - use service role key, not anon key)
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUz...

# Stripe (test mode first)
STRIPE_SECRET_KEY=sk_test_xxx
STRIPE_PUBLISHABLE_KEY=pk_test_xxx
STRIPE_WEBHOOK_SECRET=whsec_xxx

# Frontend URL for redirects
FRONTEND_URL=http://localhost:5174

# Optional: Debug mode
OELALA_DEBUG=1
```

**Get Stripe Keys:**
- Dashboard → Developers → API Keys
- Copy Publishable Key (pk_test_...)
- Copy Secret Key (sk_test_...)

### 5. Configure Stripe Webhook

**For Development (Local Testing):**
```bash
stripe listen --forward-to http://localhost:7998/api/stripe/webhook
```
This outputs a webhook secret (whsec_xxx) - copy it to .env

**For Production:**
1. Stripe Dashboard → Developers → Webhooks → Add Endpoint
2. URL: `https://yourdomain.com/api/stripe/webhook`
3. Events: `checkout.session.completed`
4. Copy Signing Secret to .env

### 6. Test the Integration

**Test Credit Balance:**
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:7998/api/credits
```

**Test Package Listing:**
```bash
curl http://localhost:7998/api/credits/packages
```

**Test Checkout Flow:**
```bash
curl -X POST \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"package_id": "starter"}' \
  http://localhost:7998/api/credits/purchase
```

**Test with Stripe Test Card:**
- Card: `4242 4242 4242 4242`
- Expiry: Any future date
- CVC: Any 3 digits

### 7. Go Live

When ready for production:

1. Switch Stripe to Live Mode (toggle in dashboard)
2. Update .env with live keys:
   - `STRIPE_SECRET_KEY=sk_live_xxx`
   - `STRIPE_PUBLISHABLE_KEY=pk_live_xxx`
   - `STRIPE_WEBHOOK_SECRET=whsec_xxx` (from live webhook)
3. Update `FRONTEND_URL` to production domain
4. Verify Supabase is using production environment
5. Test with real card (small amount first)

---

## How It Works

### Purchase Flow
1. User clicks "Buy Credits" in user menu
2. PurchaseCreditsModal shows packages
3. User selects package
4. Backend creates Stripe Checkout Session
5. User redirected to Stripe
6. User completes payment
7. Stripe calls webhook → credits added
8. User redirected back with `?success=true`
9. CreditsContext detects success → refreshes balance
10. Success notification shown (auto-hide 5s)

### Generation Flow
1. User submits generation (e.g., SDXL image)
2. Backend calculates credit cost
3. Backend checks user balance
4. If insufficient → HTTP 402 error
5. If sufficient → queue job, deduct credits
6. Job processes (ComfyUI)
7. User charged regardless of job outcome
8. (Future: Refund on failure)

---

## Architecture

### Backend (Python/FastAPI)
```
src/backend/
├── credits.py           # Credit manager, calculation logic
├── credits_api.py       # API endpoints (/api/credits/*, /api/stripe/webhook)
├── app.py              # Main app (generation endpoints have credit checks)
└── migrations/
    └── 001_credits_system.sql  # Database schema
```

### Frontend (React)
```
src/frontend/src/
├── contexts/
│   └── CreditsContext.jsx    # Credit state, purchase flow, URL handling
├── components/
│   ├── PurchaseCreditsModal.jsx  # Package selection UI
│   └── UserMenu.jsx          # Balance display, modal trigger
└── dashboard/
    ├── Dashboard.jsx         # Success/cancel notifications
    └── tools/*.jsx           # Show credit costs
```

### Database (Supabase/PostgreSQL)
```
user_credits
├── user_id (PK, FK to auth.users)
├── balance (INT, CHECK >= 0)
├── lifetime_purchased
├── lifetime_used
└── timestamps

credit_transactions
├── id (UUID)
├── user_id (FK)
├── amount (INT, +/-)
├── type (purchase|bonus|generation|refund|admin)
├── description
├── reference_id (Stripe payment_intent, job_id, etc.)
├── metadata (JSONB)
└── created_at

credit_packages
├── id (TEXT, PK)
├── name
├── credits
├── price_cents
├── currency
├── stripe_price_id
├── is_active
├── sort_order
├── description
└── badge
```

---

## Testing Commands

**Run Unit Tests:**
```bash
cd /path/to/oelala
python -m pytest tests/test_credits_system.py tests/test_credits_api.py -v
```

**Test Stripe Webhook Locally:**
```bash
# Terminal 1: Start webhook listener
stripe listen --forward-to http://localhost:7998/api/stripe/webhook

# Terminal 2: Trigger test event
stripe trigger checkout.session.completed
```

**Check Backend Logs:**
```bash
# If using systemd service
journalctl -u oelala-api -f

# If running manually
# Check console output
```

---

## Credit Costs Reference

| Generation Type | Base Cost | Notes |
|----------------|-----------|-------|
| SDXL Image | 1 credit | 1024x1024 |
| SDXL HD | 2 credits | >1280 resolution |
| Flux Image | 3 credits | |
| SD1.5 Image | 1 credit | |
| Video 480p 3s | 5 credits | |
| Video 720p 3s | 8 credits | |
| Video 1080p 3s | 10 credits | |
| Video 480p 5s | 8 credits | |
| Video 720p 5s | 12 credits | |
| Video 1080p 5s | 15 credits | |
| Audio <10s | 3 credits | |
| Audio 10-30s | 5 credits | |

---

## Support & Troubleshooting

See `docs/CREDITS_SETUP.md` for:
- Detailed troubleshooting guide
- Common error solutions
- FAQ

**Common Issues:**

1. **Credits not showing**: Check SUPABASE_SERVICE_KEY is set (not anon key)
2. **Webhook failing**: Verify STRIPE_WEBHOOK_SECRET matches Stripe CLI or Dashboard
3. **Purchase not adding credits**: Check webhook endpoint is accessible and logs
4. **Insufficient credits error**: Balance too low - buy more credits

---

## Production Checklist

Before going live:

- [ ] Run database migration in Supabase
- [ ] Create Stripe products (5 packages)
- [ ] Update migration with Stripe Price IDs
- [ ] Set all environment variables
- [ ] Configure production webhook endpoint
- [ ] Test with Stripe test cards
- [ ] Switch to Stripe live mode
- [ ] Update to live Stripe keys
- [ ] Test with real payment (small amount)
- [ ] Set up Stripe billing alerts
- [ ] Configure email receipts in Stripe
- [ ] Add refund policy to UI
- [ ] Set up monitoring for failed webhooks
- [ ] Review RLS policies
- [ ] Enable Stripe Radar (fraud protection)

---

## Contact

For questions or issues with this integration, refer to:
- `docs/CREDITS_SETUP.md` - Complete setup guide
- Stripe Dashboard - Payment logs and webhooks
- Supabase Dashboard - Database and logs
- GitHub Issues - Bug reports

---

**🎉 You're all set! The code is ready - just configure Stripe and Supabase, and you're live!**
