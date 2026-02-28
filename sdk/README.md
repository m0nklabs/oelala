# Oelala SDK & Developer Resources

Official SDKs, examples, and API reference for the [Oelala](https://oelala.xyz) AI generation platform.

## Contents

| Resource | Description |
|----------|-------------|
| [Python SDK](python/) | `pip install oelala` — sync & async clients |
| [JavaScript/TypeScript SDK](javascript/) | `npm install @oelala/sdk` — Node.js 18+ with full types |
| [Discord Bot Example](examples/discord_bot.py) | Ready-to-use Discord bot with `/imagine`, `/video`, `/animate` |
| [Postman Collection](postman_collection.json) | Import into Postman for interactive API testing |
| [OpenAPI Spec](openapi.json) | OpenAPI 3.1 specification for code generation |

## Quick Start

### Get your API key

1. Sign up at [oelala.xyz](https://oelala.xyz)
2. Go to **Settings → API Keys**
3. Create a new key — it starts with `oelala_`

### Python

```bash
pip install oelala
```

```python
from oelala import OelalaClient

client = OelalaClient(api_key="oelala_your_key")
job = client.text_to_image("a cat in space")
result = client.wait_for_job(job.job_id)
if result.succeeded:
    client.download(job.job_id, "cat.png")
```

### JavaScript / TypeScript

```bash
npm install @oelala/sdk
```

```typescript
import { OelalaClient } from "@oelala/sdk";

const client = new OelalaClient({ apiKey: "oelala_your_key" });
const job = await client.textToImage("a cat in space");
const result = await client.waitForJob(job.job_id);
```

### cURL

```bash
# Generate an image
curl -X POST https://api.oelala.xyz/api/v1/generate \
  -H "X-API-Key: oelala_your_key" \
  -H "Content-Type: application/json" \
  -d '{"type": "text-to-image", "prompt": "a beautiful sunset"}'

# Check status
curl https://api.oelala.xyz/api/v1/jobs/{job_id} \
  -H "X-API-Key: oelala_your_key"

# Download result
curl -o result.png https://api.oelala.xyz/api/v1/jobs/{job_id}/download \
  -H "X-API-Key: oelala_your_key"
```

## API Reference

Base URL: `https://api.oelala.xyz`

### Authentication

All API endpoints require an API key via the `X-API-Key` header:

```
X-API-Key: oelala_your_key_here
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/generate` | Submit a generation job |
| `GET` | `/api/v1/jobs/{job_id}` | Get job status |
| `GET` | `/api/v1/jobs/{job_id}/download` | Download completed result |
| `GET` | `/api/v1/credits` | Check credit balance |
| `GET` | `/api/v1/health` | Health check (no auth) |

### Generation Types

| Type | Credits | Description |
|------|---------|-------------|
| `text-to-image` | ~10 | Generate image from text prompt |
| `text-to-video` | 50-200 | Generate video from text prompt |
| `image-to-video` | 50-200 | Animate an existing image |

### Webhook Events

| Event | Description |
|-------|-------------|
| `job.queued` | Job entered the queue |
| `job.started` | Processing has begun |
| `job.completed` | Generation succeeded |
| `job.failed` | Generation failed |

Webhooks include HMAC-SHA256 signatures via the `X-Webhook-Signature` header for verification.

### Interactive API Docs

- **Swagger UI**: https://api.oelala.xyz/docs
- **ReDoc**: https://api.oelala.xyz/redoc

## Rate Limits

Rate limits are applied per API key. Current limits TBD — design your integration with exponential backoff.

## Support

- Issues: [GitHub](https://github.com/Oelala-xyz/oelala)
- Website: [oelala.xyz](https://oelala.xyz)
