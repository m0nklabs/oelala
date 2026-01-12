### Added

- **Webhook Delivery System** (Issue #62)
  - New `webhooks` and `webhook_deliveries` database tables with RLS policies
  - HMAC-SHA256 signature verification for webhook security
  - Exponential backoff retry logic (5 attempts: 10s, 1m, 5m, 30m, 1h)
  - Event types: `job.queued`, `job.started`, `job.completed`, `job.failed`
  - Full CRUD API for webhook management at `/webhooks/*`
  - Webhook delivery logs with response tracking
  - Test webhook endpoint for verification
  - Background retry worker for failed deliveries
  - Integration with existing WebSocket job events
  
### API Endpoints

- `GET /webhooks` - List user's webhooks
- `POST /webhooks` - Create webhook (returns signing secret once)
- `GET /webhooks/{id}` - Get webhook details
- `PATCH /webhooks/{id}` - Update webhook (with optional secret regeneration)
- `DELETE /webhooks/{id}` - Delete webhook
- `GET /webhooks/{id}/deliveries` - Get delivery history
- `POST /webhooks/{id}/test` - Send test webhook
- `GET /webhooks/events/types` - List available event types

### Webhook Payload Example

```json
{
  "event": "job.completed",
  "event_id": "uuid",
  "timestamp": "2024-01-15T12:05:00Z",
  "data": {
    "job_id": "prompt_id",
    "job_type": "text-to-video",
    "output_url": "/media/generated/video_123.mp4",
    "processing_time_seconds": 180
  }
}
```

### Migration

Run `src/backend/migrations/008_webhooks.sql` in Supabase SQL Editor.
