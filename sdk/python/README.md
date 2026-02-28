# Oelala Python SDK

Official Python SDK for the [Oelala](https://oelala.xyz) AI generation API. Generate stunning images and videos with text prompts or source images.

## Installation

```bash
pip install oelala
```

## Quick Start

```python
from oelala import OelalaClient

client = OelalaClient(api_key="oelala_your_key_here")

# Generate an image
job = client.text_to_image("a cat riding a unicorn through space")
print(f"Job {job.job_id} started!")

# Wait for completion and download
result = client.wait_for_job(job.job_id)
if result.succeeded:
    client.download(job.job_id, "space_cat.png")
    print(f"Saved to space_cat.png")
```

## Async Usage

```python
import asyncio
from oelala import AsyncOelalaClient

async def main():
    async with AsyncOelalaClient(api_key="oelala_your_key") as client:
        # Generate a video
        job = await client.text_to_video(
            "a timelapse of flowers blooming in a garden",
            duration_seconds=10,
            width=848,
            height=480,
        )

        # Poll with progress callback
        async def on_progress(status):
            print(f"  Status: {status.status}, Progress: {status.progress}%")

        result = await client.wait_for_job(job.job_id, on_progress=on_progress)
        if result.succeeded:
            await client.download(job.job_id, "flowers.mp4")

asyncio.run(main())
```

## Generation Types

| Type | Method | Description |
|------|--------|-------------|
| Text-to-Image | `client.text_to_image(prompt)` | Generate image from text |
| Text-to-Video | `client.text_to_video(prompt)` | Generate video from text |
| Image-to-Video | `client.image_to_video(prompt, image_url=url)` | Animate an image |

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | str | required | Text description of desired output |
| `negative_prompt` | str | None | What to avoid |
| `width` | int | 1024 | Output width (256-2048) |
| `height` | int | 1024 | Output height (256-2048) |
| `steps` | int | 20 | Diffusion steps (1-100) |
| `cfg` | float | 7.5 | Guidance scale (1.0-20.0) |
| `seed` | int | -1 | Random seed (-1 = random) |
| `duration_seconds` | int | None | Video duration (1-30, video only) |
| `image_url` | str | None | Source image (required for image-to-video) |

## Credits

```python
credits = client.get_credits()
print(f"Balance: {credits.balance}")
print(f"Used: {credits.lifetime_used}")
```

## Webhook Verification

```python
from oelala import verify_webhook_signature
from oelala.webhooks import parse_webhook_event

# Verify the signature
verify_webhook_signature(
    payload=request_body,
    signature=headers["X-Webhook-Signature"],
    secret="whsec_your_secret",
)

# Parse the event
event = parse_webhook_event(json.loads(request_body))
if event.is_completed:
    print(f"Job {event.job_id} finished! Download: {event.output_url}")
elif event.is_failed:
    print(f"Job {event.job_id} failed: {event.error}")
```

## Error Handling

```python
from oelala.exceptions import (
    AuthenticationError,
    InsufficientCreditsError,
    RateLimitError,
    ValidationError,
)

try:
    job = client.text_to_image("hello world")
except AuthenticationError:
    print("Invalid API key")
except InsufficientCreditsError:
    print("Buy more credits at https://oelala.xyz")
except RateLimitError as e:
    print(f"Rate limited, retry after {e.retry_after}s")
except ValidationError as e:
    print(f"Bad request: {e}")
```

## Configuration

```python
client = OelalaClient(
    api_key="oelala_...",
    base_url="http://localhost:7998",  # Local development
    timeout=60.0,                      # Request timeout
)
```

## License

MIT
