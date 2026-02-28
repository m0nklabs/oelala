# Oelala JavaScript/TypeScript SDK

Official SDK for the [Oelala](https://oelala.xyz) AI generation API. Generate stunning images and videos with text prompts or source images.

## Installation

```bash
npm install @oelala/sdk
# or
yarn add @oelala/sdk
# or
pnpm add @oelala/sdk
```

## Quick Start

```typescript
import { OelalaClient } from "@oelala/sdk";

const client = new OelalaClient({ apiKey: "oelala_your_key_here" });

// Generate an image
const job = await client.textToImage("a cat riding a unicorn through space");
console.log(`Job ${job.job_id} started!`);

// Wait for completion
const result = await client.waitForJob(job.job_id, {
  onProgress: (status) => console.log(`Status: ${status.status}`),
});

if (result.status === "completed") {
  const response = await client.download(job.job_id);
  const blob = await response.blob();
  // Save or process the result...
}
```

## Generation Types

```typescript
// Text-to-Image
const job = await client.textToImage("beautiful sunset", {
  width: 1024,
  height: 1024,
  steps: 30,
});

// Text-to-Video
const job = await client.textToVideo("ocean waves crashing", {
  duration_seconds: 10,
  width: 848,
  height: 480,
});

// Image-to-Video
const job = await client.imageToVideo(
  "make it come alive",
  "https://example.com/photo.jpg",
  { duration_seconds: 5 }
);
```

## Credits

```typescript
const credits = await client.getCredits();
console.log(`Balance: ${credits.balance}`);
```

## Webhook Verification

```typescript
import { verifyWebhookSignature, parseWebhookEvent } from "@oelala/sdk";

// Express.js example
app.post("/webhook", express.raw({ type: "application/json" }), (req, res) => {
  try {
    verifyWebhookSignature(
      req.body,
      req.headers["x-webhook-signature"] as string,
      process.env.WEBHOOK_SECRET!
    );

    const event = parseWebhookEvent(JSON.parse(req.body));

    if (event.event === "job.completed") {
      console.log(`Job done! URL: ${event.data.output_url}`);
    }

    res.sendStatus(200);
  } catch (err) {
    res.sendStatus(401);
  }
});
```

## Error Handling

```typescript
import {
  AuthenticationError,
  InsufficientCreditsError,
  RateLimitError,
  ValidationError,
} from "@oelala/sdk";

try {
  const job = await client.textToImage("hello world");
} catch (err) {
  if (err instanceof AuthenticationError) {
    console.error("Invalid API key");
  } else if (err instanceof InsufficientCreditsError) {
    console.error("Buy more credits at https://oelala.xyz");
  } else if (err instanceof RateLimitError) {
    console.error(`Rate limited, retry after ${err.retryAfter}s`);
  } else if (err instanceof ValidationError) {
    console.error("Bad request:", err.message);
  }
}
```

## Configuration

```typescript
const client = new OelalaClient({
  apiKey: "oelala_...",
  baseUrl: "http://localhost:7998", // Local development
  timeout: 60_000,                  // Request timeout (ms)
});
```

## Requirements

- Node.js 18+ (uses native `fetch`)
- TypeScript 5.0+ (optional, full type definitions included)

## License

MIT
