# Credit System & Stripe Payments Setup Guide

This guide walks you through setting up the Oelala credit system with Stripe payments integration.

## Prerequisites

- Supabase project with authentication enabled
- Stripe account (test mode for development)
- Backend environment variables configured

---

## Step 1: Database Setup (Supabase)

### Run the Migration

1. Open your Supabase project dashboard
2. Navigate to **SQL Editor**
3. Open the migration file: `src/backend/migrations/001_credits_system.sql`
4. Copy the entire contents
5. Paste into the SQL Editor
6. Click **Run** to execute

This creates:
- `user_credits` table - stores user balances
- `credit_transactions` table - audit log of all credit movements
- `credit_packages` table - available packages for purchase
- RLS policies for security
- Helper functions for atomic operations
- Auto-trigger for welcome bonus on signup

### Verify Tables Created

Run this query in SQL Editor:
```sql
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN ('user_credits', 'credit_transactions', 'credit_packages');
```

You should see all 3 tables listed.

---

## Step 2: Stripe Setup

### Create Stripe Account

1. Go to [stripe.com](https://stripe.com) and sign up
2. Complete business verification (can use test mode immediately)
3. Switch to **Test Mode** (toggle in dashboard top-right)

### Create Products & Prices

You can create products either via Dashboard or CLI:

#### Option A: Stripe Dashboard

1. Go to **Products** → **Add Product**
2. Create each package:

| Name | Description | Price | Metadata |
|------|-------------|-------|----------|
| Starter Pack | 100 generation credits | €5.00 | `credits: 100` |
| Basic Pack | 500 generation credits | €20.00 | `credits: 500` |
| Pro Pack | 1500 generation credits | €50.00 | `credits: 1500` |
| Studio Pack | 5000 generation credits | €150.00 | `credits: 5000` |
| Enterprise Pack | 20000 generation credits | €500.00 | `credits: 20000` |

3. After creating each product, copy the **Price ID** (starts with `price_xxx`)

#### Option B: Stripe CLI

Install Stripe CLI:
```bash
# Linux
wget https://github.com/stripe/stripe-cli/releases/latest/download/stripe_linux_amd64.tar.gz
tar -xvf stripe_linux_amd64.tar.gz
sudo mv stripe /usr/local/bin/
```

Login:
```bash
stripe login
```

Create products:
```bash
# Starter
PROD_STARTER=$(stripe products create --name="Starter Pack" --description="100 generation credits" -d "metadata[credits]=100" -d "metadata[package_id]=starter" --format=json | jq -r .id)
PRICE_STARTER=$(stripe prices create --product=$PROD_STARTER --unit-amount=500 --currency=eur --format=json | jq -r .id)
echo "Starter: $PRICE_STARTER"

# Basic
PROD_BASIC=$(stripe products create --name="Basic Pack" --description="500 generation credits" -d "metadata[credits]=500" -d "metadata[package_id]=basic" --format=json | jq -r .id)
PRICE_BASIC=$(stripe prices create --product=$PROD_BASIC --unit-amount=2000 --currency=eur --format=json | jq -r .id)
echo "Basic: $PRICE_BASIC"

# Pro
PROD_PRO=$(stripe products create --name="Pro Pack" --description="1500 generation credits" -d "metadata[credits]=1500" -d "metadata[package_id]=pro" --format=json | jq -r .id)
PRICE_PRO=$(stripe prices create --product=$PROD_PRO --unit-amount=5000 --currency=eur --format=json | jq -r .id)
echo "Pro: $PRICE_PRO"

# Studio
PROD_STUDIO=$(stripe products create --name="Studio Pack" --description="5000 generation credits" -d "metadata[credits]=5000" -d "metadata[package_id]=studio" --format=json | jq -r .id)
PRICE_STUDIO=$(stripe prices create --product=$PROD_STUDIO --unit-amount=15000 --currency=eur --format=json | jq -r .id)
echo "Studio: $PRICE_STUDIO"

# Enterprise
PROD_ENTERPRISE=$(stripe products create --name="Enterprise Pack" --description="20000 generation credits" -d "metadata[credits]=20000" -d "metadata[package_id]=enterprise" --format=json | jq -r .id)
PRICE_ENTERPRISE=$(stripe prices create --product=$PROD_ENTERPRISE --unit-amount=50000 --currency=eur --format=json | jq -r .id)
echo "Enterprise: $PRICE_ENTERPRISE"
```

### Update Database with Stripe Price IDs

Once you have the Price IDs, update the database:

```sql
-- Update packages with Stripe Price IDs
UPDATE public.credit_packages SET stripe_price_id = 'price_xxx' WHERE id = 'starter';
UPDATE public.credit_packages SET stripe_price_id = 'price_xxx' WHERE id = 'basic';
UPDATE public.credit_packages SET stripe_price_id = 'price_xxx' WHERE id = 'pro';
UPDATE public.credit_packages SET stripe_price_id = 'price_xxx' WHERE id = 'studio';
UPDATE public.credit_packages SET stripe_price_id = 'price_xxx' WHERE id = 'enterprise';
```

Replace `price_xxx` with your actual Price IDs.

### Get API Keys

1. Go to **Developers** → **API Keys**
2. Copy:
   - **Publishable key** (starts with `pk_test_`)
   - **Secret key** (starts with `sk_test_`)

---

## Step 3: Configure Webhooks

Stripe needs to notify your backend when payments succeed.

### Development (Local Testing)

Use Stripe CLI to forward webhooks:

```bash
stripe listen --forward-to http://localhost:7998/api/stripe/webhook
```

This will output a webhook secret like `whsec_xxx`. Copy it.

### Production

1. Go to **Developers** → **Webhooks** → **Add Endpoint**
2. Endpoint URL: `https://yourdomain.com/api/stripe/webhook`
3. Events to listen for:
   - `checkout.session.completed`
4. Click **Add Endpoint**
5. Copy the **Signing Secret** (starts with `whsec_`)

---

## Step 4: Environment Variables

Update your `.env` file (copy from `.env.example` if needed):

```bash
# Supabase (required for credits system)
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJhbG...  # Service role key, NOT anon key

# Stripe (test mode for development)
STRIPE_SECRET_KEY=sk_test_xxx
STRIPE_PUBLISHABLE_KEY=pk_test_xxx
STRIPE_WEBHOOK_SECRET=whsec_xxx

# Frontend URL (for Stripe redirect)
FRONTEND_URL=http://localhost:5174

# Optional: Debug mode
OELALA_DEBUG=1
```

**Important**:
- Use `SUPABASE_SERVICE_KEY` (service role), not anon key
- For production, use live keys: `sk_live_`, `pk_live_`, `whsec_` (live)

---

## Step 5: Install Dependencies

```bash
cd src/backend
pip install -r requirements.txt
```

This installs:
- `stripe` - Stripe Python SDK
- `httpx` - Async HTTP client (for Supabase)
- Other dependencies

---

## Step 6: Start the Backend

```bash
cd src/backend
uvicorn app:app --reload --host 0.0.0.0 --port 7998
```

Or if using systemd:
```bash
sudo systemctl restart oelala-api
```

Check logs:
```bash
journalctl -u oelala-api -f
```

---

## Step 7: Test the Integration

### Test 1: Check Balance

```bash
curl -X GET http://localhost:7998/api/credits \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

Expected response:
```json
{
  "balance": 25,
  "lifetime_purchased": 0,
  "lifetime_used": 0
}
```

### Test 2: Get Packages

```bash
curl -X GET http://localhost:7998/api/credits/packages
```

Expected response:
```json
[
  {
    "id": "starter",
    "name": "Starter",
    "credits": 100,
    "price_cents": 500,
    "currency": "EUR"
  },
  ...
]
```

### Test 3: Create Checkout Session

```bash
curl -X POST http://localhost:7998/api/credits/purchase \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"package_id": "starter"}'
```

Expected response:
```json
{
  "checkout_url": "https://checkout.stripe.com/c/pay/cs_test_xxx",
  "session_id": "cs_test_xxx"
}
```

Visit the checkout URL to test payment flow.

### Test 4: Test Webhook

Use Stripe CLI trigger:
```bash
stripe trigger checkout.session.completed
```

Check backend logs for:
```
✅ Added 100 credits to user xxx (payment: pi_xxx)
```

### Test 5: Test Card Payment

Use Stripe test card:
- Card number: `4242 4242 4242 4242`
- Expiry: Any future date
- CVC: Any 3 digits
- ZIP: Any 5 digits

Complete the checkout and verify:
1. Webhook received (check backend logs)
2. Credits added (check `/api/credits` endpoint)
3. Transaction logged (check `credit_transactions` table)

---

## Troubleshooting

### "Payment system not configured"

- Missing `STRIPE_SECRET_KEY` in `.env`
- Restart backend after adding env vars

### "Insufficient credits" for new users

- Check if user has welcome bonus (25 credits)
- Verify trigger is working: `SELECT * FROM user_credits;`
- Manually grant credits:
  ```sql
  INSERT INTO user_credits (user_id, balance)
  VALUES ('USER_UUID', 25)
  ON CONFLICT (user_id) DO UPDATE SET balance = 25;
  ```

### Webhook not receiving events

- Ensure Stripe CLI is running: `stripe listen --forward-to ...`
- Check webhook secret matches in `.env`
- Verify endpoint URL is correct
- Check backend logs for errors

### Database permissions error

- Ensure you're using `SUPABASE_SERVICE_KEY`, not anon key
- Verify RLS policies are set correctly
- Grant permissions:
  ```sql
  GRANT ALL ON public.user_credits TO service_role;
  GRANT ALL ON public.credit_transactions TO service_role;
  GRANT ALL ON public.credit_packages TO service_role;
  ```

---

## Next Steps

1. **Integrate credit deduction** in generation endpoints
2. **Update frontend** to show credit costs
3. **Add low balance warnings** in UI
4. **Set up monitoring** for failed payments
5. **Configure email receipts** via Stripe

See `docs/MONETIZATION.md` for business model details.

---

## Production Checklist

Before going live:

- [ ] Switch Stripe to live mode
- [ ] Update keys to live keys (`sk_live_`, `pk_live_`)
- [ ] Configure production webhook endpoint
- [ ] Test with real payment
- [ ] Set up Stripe billing alerts
- [ ] Configure email receipts
- [ ] Add refund policy to UI
- [ ] Set up monitoring & alerts
- [ ] Review RLS policies
- [ ] Enable Stripe Radar for fraud protection
