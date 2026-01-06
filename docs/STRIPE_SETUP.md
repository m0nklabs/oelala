# Stripe Payment Integration Setup

Complete guide for setting up Stripe payments for the Oelala credit system.

## Prerequisites

- Stripe account (create at https://stripe.com)
- Supabase project with database tables created
- Backend server with environment variables configured

## 1. Database Setup

First, ensure the credit system tables are created in Supabase.

### Run the Migration

1. Open your Supabase project dashboard
2. Navigate to **SQL Editor**
3. Copy the contents of `src/backend/migrations/001_credits_system.sql`
4. Execute the SQL migration

This creates the following tables:
- `user_credits` - User credit balances
- `credit_transactions` - Transaction audit log
- `credit_packages` - Available packages for purchase

### Verify Tables

```sql
-- Check that tables exist
SELECT tablename FROM pg_tables WHERE schemaname = 'public';

-- Check default packages were inserted
SELECT * FROM credit_packages ORDER BY sort_order;
```

## 2. Stripe Dashboard Setup

### Create Products and Prices

1. Log in to [Stripe Dashboard](https://dashboard.stripe.com)
2. Navigate to **Products** → **Add Product**

Create the following products (use **Test Mode** for development):

#### Starter Pack
- **Name**: Oelala Starter Pack
- **Description**: 100 generation credits
- **Price**: €5.00 (500 cents)
- **Currency**: EUR
- **Recurring**: No (one-time payment)

Copy the **Price ID** (starts with `price_...`)

#### Basic Pack
- **Name**: Oelala Basic Pack
- **Description**: 500 generation credits
- **Price**: €20.00 (2000 cents)
- **Currency**: EUR

#### Pro Pack
- **Name**: Oelala Pro Pack
- **Description**: 1500 generation credits
- **Price**: €50.00 (5000 cents)
- **Currency**: EUR
- **Badge**: Add tag "POPULAR"

#### Studio Pack
- **Name**: Oelala Studio Pack
- **Description**: 5000 generation credits
- **Price**: €150.00 (15000 cents)
- **Currency**: EUR
- **Badge**: Add tag "BEST VALUE"

#### Enterprise Pack
- **Name**: Oelala Enterprise Pack
- **Description**: 20000 generation credits
- **Price**: €500.00 (50000 cents)
- **Currency**: EUR

### Update Database with Stripe Price IDs

After creating products in Stripe, update the `credit_packages` table:

```sql
UPDATE public.credit_packages
SET stripe_price_id = 'price_xxx'  -- Replace with actual Price ID from Stripe
WHERE id = 'starter';

UPDATE public.credit_packages
SET stripe_price_id = 'price_yyy'
WHERE id = 'basic';

-- Repeat for all packages
```

## 3. Webhook Configuration

### Create Webhook Endpoint

1. In Stripe Dashboard, navigate to **Developers** → **Webhooks**
2. Click **Add Endpoint**
3. Set **Endpoint URL**: `https://your-domain.com/api/stripe/webhook`
4. Select **Events to listen to**:
   - `checkout.session.completed`
   - `payment_intent.succeeded` (optional)
   - `payment_intent.payment_failed` (optional)
5. Click **Add Endpoint**

### Get Webhook Signing Secret

1. Click on the webhook you just created
2. Under **Signing secret**, click **Reveal**
3. Copy the secret (starts with `whsec_...`)

## 4. Environment Variables

Add the following to your `.env` file:

```bash
# Stripe Configuration (Test Mode)
STRIPE_SECRET_KEY=sk_test_xxxxxxxxxxxxxxxxxxxxx
STRIPE_PUBLISHABLE_KEY=pk_test_xxxxxxxxxxxxxxxxxxxxx
STRIPE_WEBHOOK_SECRET=whsec_xxxxxxxxxxxxxxxxxxxxx

# Supabase Configuration
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
SUPABASE_JWT_SECRET=your-jwt-secret

# Frontend URL (for Stripe redirects)
FRONTEND_URL=http://localhost:5174
```

### Get Stripe Keys

1. In Stripe Dashboard, go to **Developers** → **API Keys**
2. Copy **Publishable key** and **Secret key**
3. Use **Test mode** keys for development

## 5. Testing Locally

### Install Stripe CLI

```bash
# macOS
brew install stripe/stripe-cli/stripe

# Linux
wget https://github.com/stripe/stripe-cli/releases/latest/download/stripe_linux_amd64.tar.gz
tar -xvf stripe_linux_amd64.tar.gz
sudo mv stripe /usr/local/bin/

# Windows
scoop install stripe
```

### Forward Webhooks to Local Server

```bash
stripe login

# Forward webhooks to local backend
stripe listen --forward-to localhost:7998/api/stripe/webhook

# You'll get a webhook signing secret - add it to .env
# whsec_xxx
```

### Test the Integration

1. Start your backend server:
   ```bash
   cd src/backend
   uvicorn app:app --reload --port 7998
   ```

2. Start the frontend:
   ```bash
   cd src/frontend
   npm run dev
   ```

3. Test a purchase:
   - Navigate to credits page
   - Click "Buy Credits"
   - Select a package
   - Use test card: `4242 4242 4242 4242`
   - Expiry: Any future date
   - CVC: Any 3 digits
   - ZIP: Any 5 digits

4. Check the webhook in Stripe CLI:
   ```
   webhook_event  checkout.session.completed
   webhook_status OK
   ```

5. Verify credits were added:
   ```sql
   SELECT * FROM user_credits WHERE user_id = 'your-user-id';
   SELECT * FROM credit_transactions WHERE user_id = 'your-user-id' ORDER BY created_at DESC;
   ```

## 6. Production Deployment

### Switch to Live Mode

1. In Stripe Dashboard, toggle from **Test Mode** to **Live Mode**
2. Create production products (same as test but in live mode)
3. Get live API keys
4. Update environment variables with live keys
5. Create production webhook endpoint

### Security Checklist

- [ ] Use HTTPS for webhook endpoint
- [ ] Verify webhook signatures (done automatically)
- [ ] Use Stripe's test mode for development
- [ ] Store API keys in environment variables (never commit)
- [ ] Enable RLS policies on Supabase tables
- [ ] Monitor webhook delivery in Stripe Dashboard
- [ ] Set up alerts for failed payments

## 7. Troubleshooting

### Webhook Not Receiving Events

1. Check webhook URL is correct and accessible
2. Verify webhook signing secret matches `.env`
3. Check Stripe Dashboard → Webhooks → Recent events
4. Look for delivery attempts and errors

### Credits Not Being Added

1. Check webhook logs in backend
2. Verify `SUPABASE_SERVICE_KEY` is set correctly
3. Check database RPC function `add_credits` exists
4. Look at backend logs for errors

### Test Cards

Use these for testing different scenarios:

- **Success**: `4242 4242 4242 4242`
- **Declined**: `4000 0000 0000 0002`
- **Insufficient Funds**: `4000 0000 0000 9995`
- **3D Secure**: `4000 0027 6000 3184`

## 8. API Reference

### Checkout Endpoint

```bash
POST /api/credits/purchase
Content-Type: application/json
Authorization: Bearer <jwt-token>

{
  "package_id": "starter"
}

# Response
{
  "checkout_url": "https://checkout.stripe.com/...",
  "session_id": "cs_test_xxx"
}
```

### Get Balance

```bash
GET /api/credits
Authorization: Bearer <jwt-token>

# Response
{
  "balance": 125,
  "lifetime_purchased": 100,
  "lifetime_used": 75
}
```

### Get Packages

```bash
GET /api/credits/packages

# Response
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

## Support

For issues with Stripe integration:
- Check [Stripe Documentation](https://stripe.com/docs)
- Review [Stripe API Reference](https://stripe.com/docs/api)
- Check webhook logs in Stripe Dashboard
- Review backend logs for errors

For database issues:
- Check [Supabase Documentation](https://supabase.com/docs)
- Verify RLS policies are correct
- Check SQL function definitions
