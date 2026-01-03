# Oelala Monetization Plan

> **Last Updated**: 2026-01-03  
> **Status**: Planning Phase

## Business Model: Freemium SaaS

### Target Markets

| Segment | Description | Size |
|---------|-------------|------|
| **Hobbyists** | AI art enthusiasts, experimenters | Large |
| **Content Creators** | YouTubers, TikTokers, streamers | Medium |
| **Professionals** | Video editors, VFX artists | Small |
| **Studios** | Production companies, agencies | Very Small |

---

## Pricing Tiers

### Free Tier
**Price**: $0/month

| Feature | Limit |
|---------|-------|
| Generations | 50/month |
| Storage | 2 GB |
| Max resolution | 720p |
| Max video length | 6 seconds |
| Queue priority | Low |
| Watermark | Yes (small) |
| Models | Basic set only |
| API access | No |

### Creator Tier
**Price**: $19/month ($190/year = 2 months free)

| Feature | Limit |
|---------|-------|
| Generations | 500/month |
| Storage | 50 GB |
| Max resolution | 1080p |
| Max video length | 15 seconds |
| Queue priority | Normal |
| Watermark | No |
| Models | All public models |
| API access | Limited (100 req/day) |
| Voice cloning | 3 voices |
| Commercial use | Yes |

### Pro Tier
**Price**: $49/month ($490/year = 2 months free)

| Feature | Limit |
|---------|-------|
| Generations | 2000/month |
| Storage | 200 GB |
| Max resolution | 4K |
| Max video length | 30 seconds |
| Queue priority | High |
| Watermark | No |
| Models | All models + early access |
| API access | Full (1000 req/day) |
| Voice cloning | 10 voices |
| Custom LoRA upload | 5 LoRAs |
| Commercial use | Yes |
| Priority support | Email |

### Studio Tier
**Price**: $199/month ($1990/year = 2 months free)

| Feature | Limit |
|---------|-------|
| Generations | Unlimited* |
| Storage | 1 TB |
| Max resolution | 4K+ |
| Max video length | 60 seconds |
| Queue priority | Highest |
| Watermark | No |
| Models | All + private models |
| API access | Unlimited |
| Voice cloning | Unlimited |
| Custom LoRA upload | Unlimited |
| Custom model training | Yes |
| Commercial use | Yes + white-label |
| Priority support | Slack/Discord |
| SLA | 99.5% uptime |

*Fair use policy: ~10,000 generations/month

### Enterprise / Self-Hosted
**Price**: Custom (starting $999/month)

- On-premise deployment
- Dedicated GPU resources
- Custom integrations
- SSO/SAML
- Audit logs
- Custom SLA (99.9%+)
- Dedicated support

---

## Add-ons (À la carte)

| Add-on | Price | Description |
|--------|-------|-------------|
| Extra storage | $5/50GB/month | Additional storage |
| Extra generations | $10/500 | One-time credit pack |
| Priority queue boost | $5/month | Jump the queue |
| Custom voice clone | $10/voice | Additional voice slot |
| Custom LoRA training | $25/training | Train on your images |
| Remove watermark (Free tier) | $3/month | Keep free limits |
| API overage | $0.02/request | Beyond tier limit |

---

## Payment Processing

### Recommended: Stripe

| Feature | Status |
|---------|--------|
| Subscription billing | Required |
| Usage-based billing | Required |
| Invoicing | Required |
| Tax calculation | Required |
| Multi-currency | Nice to have |
| Crypto payments | Future |

### Alternative: Paddle
- Handles EU VAT automatically
- Acts as Merchant of Record
- Higher fees but less compliance burden

### Implementation Priority

1. **Phase 1**: Stripe Checkout (hosted page)
   - Quick to implement
   - Handles subscriptions
   - No PCI compliance needed

2. **Phase 2**: Stripe Elements (embedded)
   - Better UX
   - Custom checkout flow

3. **Phase 3**: Usage metering
   - Track generations
   - Overage billing

---

## Revenue Projections

### Conservative Scenario (Year 1)

| Month | Free | Creator | Pro | Studio | MRR |
|-------|------|---------|-----|--------|-----|
| M1 | 100 | 5 | 1 | 0 | $144 |
| M3 | 500 | 25 | 5 | 1 | $839 |
| M6 | 2000 | 100 | 20 | 3 | $3,477 |
| M12 | 10000 | 400 | 80 | 10 | $13,510 |

**Year 1 Total**: ~$80,000

### Moderate Scenario (Year 1)

| Month | Free | Creator | Pro | Studio | MRR |
|-------|------|---------|-----|--------|-----|
| M1 | 200 | 10 | 2 | 0 | $288 |
| M3 | 1000 | 75 | 15 | 2 | $2,158 |
| M6 | 5000 | 300 | 60 | 8 | $9,332 |
| M12 | 25000 | 1200 | 300 | 30 | $43,530 |

**Year 1 Total**: ~$250,000

---

## Cost Structure

### Infrastructure Costs (per user/month)

| Component | Free | Creator | Pro | Studio |
|-----------|------|---------|-----|--------|
| GPU compute | $0.50 | $3 | $8 | $25 |
| Storage | $0.02 | $0.50 | $2 | $10 |
| Bandwidth | $0.10 | $0.50 | $2 | $5 |
| **Total** | $0.62 | $4 | $12 | $40 |

### Gross Margins

| Tier | Price | Cost | Margin |
|------|-------|------|--------|
| Free | $0 | $0.62 | -$0.62 |
| Creator | $19 | $4 | 79% |
| Pro | $49 | $12 | 76% |
| Studio | $199 | $40 | 80% |

### Fixed Costs (Monthly)

| Item | Cost |
|------|------|
| GPU servers (base) | $500-2000 |
| Storage (base) | $100-500 |
| Payment processing | 2.9% + $0.30 |
| Domain/SSL | $10 |
| Monitoring | $50 |
| Backups | $50 |
| **Total** | ~$800-3000 |

---

## Conversion Funnel

```
Visitors → Sign-ups → Active → Paying → Retained
  100%       10%        30%       5%       80%

Example:
10,000 visitors/month
→ 1,000 sign-ups
→ 300 active users  
→ 15 paying customers
→ 12 retained next month
```

### Conversion Optimization

| Stage | Target | Strategy |
|-------|--------|----------|
| Visit → Signup | 15% | Landing page, social proof |
| Signup → Active | 50% | Onboarding, quick win |
| Active → Paid | 10% | Feature gating, trials |
| Paid → Retained | 90% | Quality, support, features |

---

## Marketing Channels

### Organic (Low Cost)

| Channel | Strategy |
|---------|----------|
| SEO | Blog, tutorials, comparisons |
| Social | Twitter/X, Reddit, Discord |
| YouTube | Tutorials, showcases |
| Product Hunt | Launch campaign |
| Hacker News | Show HN post |

### Paid (Growth Phase)

| Channel | CPA Target |
|---------|------------|
| Google Ads | $10-20 |
| Reddit Ads | $5-15 |
| Twitter Ads | $15-25 |
| Influencer | $5-10 |

---

## Competitive Pricing Analysis

| Competitor | Comparable Tier | Price | Notes |
|------------|-----------------|-------|-------|
| RunwayML | Standard | $15/mo | Limited seconds |
| Pika Labs | Pro | $58/mo | 700 credits |
| Luma AI | Pro | $29/mo | Limited features |
| ElevenLabs | Creator | $22/mo | Voice only |
| Midjourney | Standard | $30/mo | Images only |

**Positioning**: Competitive with more features (video + voice + image in one platform)

---

## Legal & Compliance

### Required

- [ ] Terms of Service
- [ ] Privacy Policy
- [ ] Cookie Policy
- [ ] Refund Policy
- [ ] Acceptable Use Policy
- [ ] DMCA Policy

### Regional

- [ ] GDPR compliance (EU)
- [ ] CCPA compliance (California)
- [ ] Age verification (13+/18+)

### Content

- [ ] NSFW policy (allowed/gated?)
- [ ] Copyright/IP policy
- [ ] AI-generated content disclosure

---

## Implementation Roadmap

### Phase 1: Free Launch (Month 1-2)
- [ ] Core generation features
- [ ] User accounts (basic)
- [ ] Usage tracking
- [ ] Waitlist/invite system

### Phase 2: Paid Launch (Month 3-4)
- [ ] Stripe integration
- [ ] Subscription management
- [ ] Feature gating by tier
- [ ] Basic analytics

### Phase 3: Growth (Month 5-8)
- [ ] Usage-based billing
- [ ] API access
- [ ] Referral program
- [ ] Team accounts

### Phase 4: Scale (Month 9-12)
- [ ] Enterprise tier
- [ ] Self-hosted option
- [ ] Marketplace (LoRAs, voices)
- [ ] White-label

---

## Key Metrics to Track

| Metric | Definition | Target |
|--------|------------|--------|
| MRR | Monthly recurring revenue | Growing |
| ARPU | Average revenue per user | $15+ |
| CAC | Customer acquisition cost | <$50 |
| LTV | Lifetime value | >$200 |
| Churn | Monthly cancellation rate | <5% |
| NPS | Net promoter score | >50 |

---

## Related Documents

- [ROADMAP.md](./ROADMAP.md) - Product roadmap
- [MEDIA_STORAGE.md](./MEDIA_STORAGE.md) - Storage architecture
- [../ARCHITECTURE.md](./ARCHITECTURE.md) - System architecture
