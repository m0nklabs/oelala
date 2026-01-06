# Credit System Usage Guide

## Overview

Oelala uses a credit-based system for AI generation. Credits are consumed when you create images, videos, or audio. You can purchase credits via Stripe.

## Credit Costs

### Image Generation

| Type | Resolution | Credits |
|------|-----------|---------|
| SD 1.5 | 512x768 | 1 credit |
| SDXL | 1024x1024 | 1-2 credits |
| SDXL HD | 1344x768+ | 2-3 credits |
| Flux | 1024x1024 | 2 credits |
| Flux HD | 1344x768+ | 3 credits |
| Wan2.2 T2I | 512x512 | 2 credits |

### Video Generation

| Type | Duration | Resolution | Credits |
|------|----------|-----------|---------|
| I2V Short | 3s | 480p | 5 credits |
| I2V Medium | 5s | 480p | 8 credits |
| I2V HD Short | 3s | 720p | 10 credits |
| I2V HD Medium | 5s | 720p | 15 credits |
| T2V Short | 3s | 480p | 8 credits |
| T2V Medium | 5s | 480p | 12 credits |

### Audio Generation

| Type | Duration | Credits |
|------|----------|---------|
| TTS | <10s | 3 credits |
| Music/SFX Short | <10s | 3 credits |
| Music/SFX Long | 10-30s | 5 credits |
| Voice Clone | Any | 20 credits |

## Credit Packages

| Package | Credits | Price | Per Credit |
|---------|---------|-------|------------|
| Starter | 100 | €5.00 | €0.050 |
| Basic | 500 | €20.00 | €0.040 |
| Pro | 1,500 | €50.00 | €0.033 |
| Studio | 5,000 | €150.00 | €0.030 |
| Enterprise | 20,000 | €500.00 | €0.025 |

## Welcome Bonus

New users receive **25 free credits** to get started!

## How to Purchase Credits

### Via Web Interface

1. Navigate to the **Credits** page
2. Click **Buy Credits**
3. Select a package
4. Complete payment via Stripe
5. Credits are added immediately

### Via API

```bash
# 1. Get available packages
curl https://api.oelala.com/api/credits/packages

# 2. Initiate purchase
curl -X POST https://api.oelala.com/api/credits/purchase \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"package_id": "pro"}'

# Response includes checkout_url - redirect user to complete payment
```

## Checking Your Balance

### Via Web Interface

Your current credit balance is displayed in the top-right corner of the interface.

### Via API

```bash
curl https://api.oelala.com/api/credits \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

Response:
```json
{
  "balance": 125,
  "lifetime_purchased": 1500,
  "lifetime_used": 1375
}
```

## Transaction History

View all credit transactions (purchases, usage, bonuses, refunds):

```bash
curl https://api.oelala.com/api/credits/history \
  -H "Authorization: Bearer YOUR_JWT_TOKEN"
```

## Credit Estimation

Before generating content, you can estimate the credit cost:

```bash
curl -X POST https://api.oelala.com/api/credits/estimate \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "generation_type": "wan22_i2v",
    "width": 720,
    "height": 480,
    "duration_seconds": 3,
    "steps": 20
  }'
```

Response:
```json
{
  "estimated_credits": 5,
  "breakdown": {
    "base_type": "wan22_i2v",
    "base_cost": 5
  },
  "current_balance": 125,
  "sufficient": true
}
```

## Insufficient Credits

If you don't have enough credits for a generation, you'll receive a `402 Payment Required` error:

```json
{
  "error": "insufficient_credits",
  "required": 10,
  "available": 5,
  "packages": [...]
}
```

The error includes available packages to purchase.

## Credit Refunds

Credits are automatically refunded if a generation fails:

- **Queue failure**: Full refund
- **Generation error**: Full refund
- **Partial completion**: No refund (credits consumed)

Refunds appear in your transaction history as `type: "refund"`.

## FAQ

### Do credits expire?

No! Credits never expire. Use them whenever you want.

### Can I get a refund for purchased credits?

Purchased credits are non-refundable. However, credits are automatically refunded if a generation fails.

### What happens if I cancel during generation?

Credits are deducted when the job is queued, not when it completes. Canceling won't refund credits.

### Can I share credits between users?

No. Credits are tied to your user account and cannot be transferred.

### Is there a monthly subscription?

Currently, we offer pay-as-you-go credits only. Subscriptions may be added in the future.

### What payment methods are supported?

We support all major credit cards and iDEAL via Stripe:
- Visa
- Mastercard
- American Express
- iDEAL (Netherlands)

### How secure is my payment?

Payments are processed by Stripe, a PCI-compliant payment processor. We never store your card details.

## Support

For billing issues:
- Email: billing@oelala.com
- Discord: https://discord.gg/oelala

For technical issues:
- GitHub Issues: https://github.com/m0nklabs/oelala/issues
- Discord: https://discord.gg/oelala
