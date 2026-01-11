# Oelala REST API v1

Public REST API for programmatic access to Oelala AI video generation.

## Authentication

The API uses API keys for authentication. You can create API keys from your account dashboard.

### Getting an API Key

1. Log in to [Oelala](https://oelala.xyz)
2. Go to Settings → API Keys
3. Click "Create New API Key"
4. Give it a name (e.g., "My Production App")
5. **Save the key immediately** - it's only shown once!

API keys look like this: `oelala_1234567890abcdef...`

### Using Your API Key

Include your API key in the `X-API-Key` header:

```bash
curl https://oelala.xyz/api/v1/... \
  -H "X-API-Key: oelala_your_key_here"
```

## Base URL

```
https://oelala.xyz/api/v1
```

## Endpoints

### Health Check

Check API status (no authentication required).

```
GET /api/v1/health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "timestamp": "2024-01-10T12:00:00Z"
}
```

---

### Generate Image/Video

Generate images or videos using AI.

```
POST /api/v1/generate
```

**Headers:**
- `X-API-Key: your_api_key` (required)
- `Content-Type: application/json`

**Request Body:**

```json
{
  "type": "text-to-image",
  "prompt": "a beautiful sunset over mountains",
  "negative_prompt": "ugly, blurry, low quality",
  "width": 1024,
  "height": 1024,
  "steps": 20,
  "cfg": 7.5,
  "seed": -1
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `type` | string | Yes | Generation type: `text-to-image`, `text-to-video`, `image-to-video` |
| `prompt` | string | Yes | Text prompt describing what to generate |
| `negative_prompt` | string | No | What to avoid in the generation |
| `width` | integer | No | Output width (256-2048, default: 1024) |
| `height` | integer | No | Output height (256-2048, default: 1024) |
| `steps` | integer | No | Inference steps (1-100, default: 20) |
| `cfg` | float | No | CFG scale (1.0-20.0, default: 7.5) |
| `seed` | integer | No | Random seed (-1 for random) |
| `duration_seconds` | integer | No | Video duration (1-30, for video types) |
| `image_url` | string | No | Source image URL (for image-to-video) |

**Response:**

```json
{
  "job_id": "abc-123-def-456",
  "status": "queued",
  "credits_used": 10,
  "estimated_time_seconds": 30
}
```

**Credit Costs:**
- Text-to-Image (SDXL): ~10 credits
- Text-to-Video (Wan2.2, 3s): ~50 credits
- Image-to-Video (Wan2.2, 3s): ~50 credits

---

### Get Job Status

Poll the status of a generation job.

```
GET /api/v1/jobs/{job_id}
```

**Headers:**
- `X-API-Key: your_api_key` (required)

**Response:**

```json
{
  "job_id": "abc-123-def-456",
  "status": "completed",
  "progress": 100,
  "created_at": "2024-01-10T12:00:00Z",
  "completed_at": "2024-01-10T12:00:30Z",
  "error": null,
  "result_url": "/api/v1/jobs/abc-123-def-456/download",
  "metadata": {
    "type": "text-to-image",
    "prompt": "a beautiful sunset"
  }
}
```

**Job Statuses:**
- `queued`: Job is waiting to start
- `running`: Job is currently processing
- `completed`: Job finished successfully
- `failed`: Job failed with error

**Polling:** Poll this endpoint every 5-10 seconds until status is `completed` or `failed`.

---

### Download Result

Download the generated image or video.

```
GET /api/v1/jobs/{job_id}/download
```

**Headers:**
- `X-API-Key: your_api_key` (required)

**Response:**
- Binary file (PNG, JPEG, or MP4)
- `Content-Type` header indicates file type

**Example:**

```bash
curl https://oelala.xyz/api/v1/jobs/abc-123/download \
  -H "X-API-Key: oelala_your_key_here" \
  -o result.mp4
```

---

### Get Credits Balance

Check your current credit balance.

```
GET /api/v1/credits
```

**Headers:**
- `X-API-Key: your_api_key` (required)

**Response:**

```json
{
  "balance": 150,
  "lifetime_purchased": 200,
  "lifetime_used": 50
}
```

## API Key Management

Manage your API keys through these endpoints (requires JWT authentication from web login):

### Create API Key

```
POST /api/keys
```

**Headers:**
- `Authorization: Bearer your_jwt_token` (required)
- `Content-Type: application/json`

**Request Body:**

```json
{
  "name": "My Production App",
  "expires_days": 90
}
```

**Response:**

```json
{
  "id": "key-uuid",
  "name": "My Production App",
  "api_key": "oelala_abc123...",
  "key_prefix": "oelala_abc123",
  "created_at": "2024-01-10T12:00:00Z",
  "expires_at": "2024-04-10T12:00:00Z"
}
```

**⚠️ Important:** The full `api_key` is only shown once! Save it securely.

### List API Keys

```
GET /api/keys
```

**Headers:**
- `Authorization: Bearer your_jwt_token` (required)

**Response:**

```json
[
  {
    "id": "key-uuid",
    "name": "My Production App",
    "key_prefix": "oelala_abc123",
    "is_active": true,
    "usage_count": 42,
    "last_used_at": "2024-01-10T11:00:00Z",
    "created_at": "2024-01-01T12:00:00Z",
    "expires_at": "2024-04-01T12:00:00Z"
  }
]
```

### Disable/Enable API Key

```
PATCH /api/keys/{key_id}
```

**Headers:**
- `Authorization: Bearer your_jwt_token` (required)
- `Content-Type: application/json`

**Request Body:**

```json
{
  "is_active": false
}
```

### Delete API Key

```
DELETE /api/keys/{key_id}
```

**Headers:**
- `Authorization: Bearer your_jwt_token` (required)

**Response:**

```json
{
  "message": "API key deleted successfully",
  "id": "key-uuid"
}
```

## Rate Limits

Rate limits are applied per API key:
- **TBD** (to be determined in future updates)

## Error Responses

All errors follow this format:

```json
{
  "detail": "Error message here"
}
```

**Common HTTP Status Codes:**

| Code | Meaning |
|------|---------|
| 200 | Success |
| 400 | Bad Request (invalid parameters) |
| 401 | Unauthorized (missing or invalid API key) |
| 403 | Forbidden (insufficient credits) |
| 404 | Not Found |
| 422 | Validation Error (invalid request body) |
| 503 | Service Unavailable (generation backend down) |

## Example Workflows

### Text-to-Image Generation

```bash
# 1. Generate image
RESPONSE=$(curl -s https://oelala.xyz/api/v1/generate \
  -H "X-API-Key: oelala_your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "text-to-image",
    "prompt": "a serene mountain landscape at sunset",
    "width": 1024,
    "height": 768
  }')

JOB_ID=$(echo $RESPONSE | jq -r '.job_id')
echo "Job ID: $JOB_ID"

# 2. Poll for completion
while true; do
  STATUS=$(curl -s https://oelala.xyz/api/v1/jobs/$JOB_ID \
    -H "X-API-Key: oelala_your_key" \
    | jq -r '.status')

  echo "Status: $STATUS"

  if [ "$STATUS" = "completed" ]; then
    break
  elif [ "$STATUS" = "failed" ]; then
    echo "Generation failed!"
    exit 1
  fi

  sleep 5
done

# 3. Download result
curl https://oelala.xyz/api/v1/jobs/$JOB_ID/download \
  -H "X-API-Key: oelala_your_key" \
  -o result.png

echo "Downloaded result.png"
```

### Video Generation

```bash
# Generate video
curl https://oelala.xyz/api/v1/generate \
  -H "X-API-Key: oelala_your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "type": "text-to-video",
    "prompt": "a flowing river through a forest",
    "duration_seconds": 3
  }'
```

## SDKs

Official SDKs coming soon for:
- Python
- JavaScript/TypeScript
- Go

## Support

- **Documentation:** https://docs.oelala.xyz
- **GitHub Issues:** https://github.com/m0nklabs/oelala/issues
- **Discord:** https://discord.gg/oelala

## Changelog

### v1.0.0 (2024-01-10)
- Initial public release
- Text-to-image generation
- Text-to-video generation
- Image-to-video generation
- API key authentication
- Credit balance checking
