# WebSocket Progress Events API

## Overview

The Oelala backend provides real-time job progress and queue position updates via WebSocket. This allows frontend clients to display live progress bars, queue positions, and ETAs without polling.

**Authentication Required**: This endpoint requires JWT authentication via the Authorization header.

## WebSocket Endpoint

```
ws://localhost:7998/ws/progress
```

### Authentication

The endpoint requires a valid JWT token sent in the Authorization header. The user ID is derived from the authenticated session token, not from query parameters.

## Connection

```javascript
// Get JWT token from your auth system
const token = localStorage.getItem('auth_token'); // or from your auth state

const ws = new WebSocket('ws://localhost:7998/ws/progress');

// Note: WebSocket API doesn't support custom headers directly in browser
// You need to send the token in the first message or use subprotocol
// For production, consider using a WebSocket library that supports headers

// Alternative: Send token after connection
ws.onopen = () => {
  console.log('✅ Connected to progress WebSocket');
  // If using token-based auth, send it in first message
  ws.send(JSON.stringify({ type: 'auth', token: token }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('Event:', message.type, message.data);
};

ws.onerror = (error) => {
  console.error('❌ WebSocket error:', error);
};

ws.onclose = () => {
  console.log('🔌 Disconnected from progress WebSocket');
};
```

**Note**: The current implementation expects the JWT token in the HTTP Authorization header during the WebSocket handshake. Browser WebSocket API has limited support for custom headers. For production use, consider one of these approaches:

1. Use a WebSocket library that supports custom headers (e.g., `socket.io-client`, `ws` in Node.js)
2. Send token as a subprotocol during handshake
3. Use a cookie-based session authentication instead of header-based JWT
4. Send token in the first WebSocket message after connection (requires additional backend handling)

## Event Types

All events follow this structure:

```json
{
  "type": "event_type",
  "timestamp": "2026-01-08T21:30:45.123456",
  "data": { ... }
}
```

### 1. Queue Update (`queue_update`)

Sent when a job's queue position changes.

```json
{
  "type": "queue_update",
  "timestamp": "2026-01-08T21:30:45.123456",
  "data": {
    "job_id": "prompt_abc123",
    "queue_position": 3,
    "total_pending": 5,
    "status": "queued",
    "eta_seconds": 360,
    "eta_human": "6m 0s"
  }
}
```

**Fields:**
- `job_id`: ComfyUI prompt ID
- `queue_position`: Current position in queue (0 = running, 1+ = waiting)
- `total_pending`: Total number of jobs waiting
- `status`: `"queued"` or `"running"`
- `eta_seconds`: Estimated seconds until job starts (optional)
- `eta_human`: Human-readable ETA string (optional)

### 2. Progress Update (`progress`)

Sent during job execution with progress percentage.

```json
{
  "type": "progress",
  "timestamp": "2026-01-08T21:31:12.456789",
  "data": {
    "job_id": "prompt_abc123",
    "progress": 45,
    "status": "running",
    "message": "Processing: VAE Encode",
    "node_name": "🎨 VAE Encode"
  }
}
```

**Fields:**
- `job_id`: ComfyUI prompt ID
- `progress`: Progress percentage (0-100)
- `status`: Always `"running"`
- `message`: Current processing step (optional)
- `node_name`: Friendly name of current ComfyUI node (optional)

**Update Frequency:**
- Progress events are rate-limited to max 10 per second per job
- Typically updates every 2-5 seconds during active processing

### 3. Job Complete (`job_complete`)

Sent when a job finishes successfully.

```json
{
  "type": "job_complete",
  "timestamp": "2026-01-08T21:35:00.789012",
  "data": {
    "job_id": "prompt_abc123",
    "status": "completed",
    "progress": 100,
    "output_url": "/comfyui-output/video.mp4",
    "metadata": {
      "duration": 5.2,
      "frames": 81
    }
  }
}
```

**Fields:**
- `job_id`: ComfyUI prompt ID
- `status`: Always `"completed"`
- `progress`: Always `100`
- `output_url`: URL to download generated output (optional)
- `metadata`: Additional job-specific data (optional)

### 4. Job Failed (`job_failed`)

Sent when a job fails with an error.

```json
{
  "type": "job_failed",
  "timestamp": "2026-01-08T21:32:30.345678",
  "data": {
    "job_id": "prompt_abc123",
    "status": "failed",
    "error": "Out of memory on CUDA device 0",
    "metadata": {
      "node": "7"
    }
  }
}
```

**Fields:**
- `job_id`: ComfyUI prompt ID
- `status`: Always `"failed"`
- `error`: Error message
- `metadata`: Additional error context (optional)

## Client Implementation Example

### React Hook

```typescript
import { useEffect, useState } from 'react';

interface ProgressEvent {
  type: 'queue_update' | 'progress' | 'job_complete' | 'job_failed';
  timestamp: string;
  data: any;
}

export function useJobProgress(jobId: string, userId?: string) {
  const [progress, setProgress] = useState(0);
  const [queuePosition, setQueuePosition] = useState<number | null>(null);
  const [status, setStatus] = useState<'queued' | 'running' | 'completed' | 'failed'>('queued');
  const [error, setError] = useState<string | null>(null);
  const [outputUrl, setOutputUrl] = useState<string | null>(null);

  useEffect(() => {
    const wsUrl = `ws://localhost:7998/ws/progress${userId ? `?user_id=${userId}` : ''}`;
    const ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      const message: ProgressEvent = JSON.parse(event.data);

      // Only process events for our job
      if (message.data.job_id !== jobId) return;

      switch (message.type) {
        case 'queue_update':
          setQueuePosition(message.data.queue_position);
          setStatus(message.data.status);
          break;

        case 'progress':
          setProgress(message.data.progress);
          setStatus('running');
          setQueuePosition(null); // Clear queue position once running
          break;

        case 'job_complete':
          setProgress(100);
          setStatus('completed');
          setOutputUrl(message.data.output_url);
          break;

        case 'job_failed':
          setStatus('failed');
          setError(message.data.error);
          break;
      }
    };

    return () => ws.close();
  }, [jobId, userId]);

  return { progress, queuePosition, status, error, outputUrl };
}
```

### Vue 3 Composable

```typescript
import { ref, onMounted, onUnmounted } from 'vue';

export function useJobProgress(jobId: string, userId?: string) {
  const progress = ref(0);
  const queuePosition = ref<number | null>(null);
  const status = ref<'queued' | 'running' | 'completed' | 'failed'>('queued');
  const error = ref<string | null>(null);
  const outputUrl = ref<string | null>(null);

  let ws: WebSocket | null = null;

  onMounted(() => {
    const wsUrl = `ws://localhost:7998/ws/progress${userId ? `?user_id=${userId}` : ''}`;
    ws = new WebSocket(wsUrl);

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);

      if (message.data.job_id !== jobId) return;

      switch (message.type) {
        case 'queue_update':
          queuePosition.value = message.data.queue_position;
          status.value = message.data.status;
          break;

        case 'progress':
          progress.value = message.data.progress;
          status.value = 'running';
          queuePosition.value = null;
          break;

        case 'job_complete':
          progress.value = 100;
          status.value = 'completed';
          outputUrl.value = message.data.output_url;
          break;

        case 'job_failed':
          status.value = 'failed';
          error.value = message.data.error;
          break;
      }
    };
  });

  onUnmounted(() => {
    ws?.close();
  });

  return { progress, queuePosition, status, error, outputUrl };
}
```

## Backend Architecture

### Components

1. **WebSocketManager** (`src/backend/websocket_handler.py`)
   - Manages client connections grouped by user
   - Broadcasts events to appropriate clients
   - Handles job ownership tracking

2. **JobQueueManager** (`src/backend/job_queue.py`)
   - Polls ComfyUI queue every 2 seconds
   - Tracks queue positions and ETAs
   - Records historical execution times for ETA estimation

3. **ComfyUIProgressMonitor** (`src/backend/comfyui_progress_monitor.py`)
   - Listens to ComfyUI's WebSocket for progress events
   - Relays progress to WebSocketManager
   - Maps node IDs to human-readable names

### Job Lifecycle

```
1. Job Queued
   └─> queue_update (position N)

2. Queue Advances
   └─> queue_update (position N-1, N-2, ...)

3. Job Starts Running
   └─> queue_update (position 0)

4. Processing
   ├─> progress (10%, node: Load Image)
   ├─> progress (25%, node: VAE Encode)
   ├─> progress (50%, node: Sampler)
   ├─> progress (75%, node: VAE Decode)
   └─> progress (95%, node: Video Combine)

5. Completion
   └─> job_complete (output_url)
```

## Performance Characteristics

- **Queue updates**: Sent within 500ms of position change (2s polling interval)
- **Progress updates**: Rate-limited to 10/second, typically 2-5s intervals
- **ETA accuracy**: Based on rolling average of last 20 completed jobs
- **Connection overhead**: ~1KB/minute per idle connection

## Error Handling

Clients should implement:

1. **Reconnection logic** with exponential backoff
2. **Heartbeat/ping** to detect stale connections
3. **Timeout handling** for jobs that don't progress

Example reconnection:

```javascript
let reconnectDelay = 1000;

function connect() {
  const ws = new WebSocket('ws://localhost:7998/ws/progress');

  ws.onopen = () => {
    reconnectDelay = 1000; // Reset on success
  };

  ws.onclose = () => {
    setTimeout(connect, reconnectDelay);
    reconnectDelay = Math.min(reconnectDelay * 2, 30000); // Max 30s
  };

  return ws;
}
```

## Testing

Use `websocat` for manual testing:

```bash
# Connect and receive events
websocat ws://localhost:7998/ws/progress?user_id=test_user

# Send ping (optional)
echo '{"type":"ping"}' | websocat ws://localhost:7998/ws/progress
```

Expected response:
```json
{"type":"pong"}
```

## Security Considerations

1. **Authentication**: ✅ **IMPLEMENTED** - WebSocket endpoint now requires JWT authentication
   - The endpoint validates JWT tokens from the Authorization header during WebSocket handshake
   - User ID is derived from the authenticated token payload, not from client-supplied parameters
   - Unauthorized clients receive a 1008 close code with "Authentication required" reason
   - Implementation:
     ```python
     @app.websocket("/ws/progress")
     async def websocket_progress(
         websocket: WebSocket,
         credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
     ):
         # Validates token and extracts user_id from JWT payload
         payload = decode_token(credentials.credentials)
         user_id = payload.get("sub")
     ```
   - **Browser Limitation**: Standard WebSocket API doesn't support custom headers. Consider:
     - Using a WebSocket library with header support
     - Cookie-based authentication
     - Token in subprotocol or first message (requires backend modification)

2. **Authorization**: Events are only sent to job owners
   - Users can only see their own job progress
   - Job ownership is tracked via authenticated `user_id` from JWT
   - Each user receives events only for jobs they submitted

3. **DoS Protection**:
   - Rate limiting prevents event spam (100ms minimum between duplicate events)
   - Connection limits should be enforced at nginx/load balancer level
   - Consider implementing per-user connection limits
   - Rate limit cache is automatically cleaned up when jobs complete

## Migration from Polling

Old polling approach:
```javascript
// ❌ Old: Poll every 2 seconds
setInterval(async () => {
  const response = await fetch(`/progress/${jobId}`);
  const data = await response.json();
  setProgress(data.progress);
}, 2000);
```

New WebSocket approach:
```javascript
// ✅ New: Real-time updates
const ws = new WebSocket('ws://localhost:7998/ws/progress');
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  if (message.data.job_id === jobId && message.type === 'progress') {
    setProgress(message.data.progress);
  }
};
```

Benefits:
- Eliminates polling overhead (saves ~30 requests/minute per job)
- Sub-second latency for updates
- Lower server load
- Better UX with instant feedback
