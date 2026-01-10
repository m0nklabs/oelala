# WebSocket Progress Events API

## Overview

The Oelala backend provides real-time job progress and queue position updates via WebSocket. This allows frontend clients to display live progress bars, queue positions, and ETAs without polling.

**Authentication Required**: This endpoint requires JWT authentication sent in the first message after connection.

## WebSocket Endpoint

```
ws://localhost:7998/ws/progress
```

### Authentication

The endpoint requires a valid JWT token sent in the first message after connecting. The user ID is derived from the authenticated session token.

## Connection

```javascript
// Get JWT token from your auth system
const token = localStorage.getItem('auth_token'); // or from your auth state

const ws = new WebSocket('ws://localhost:7998/ws/progress');

ws.onopen = () => {
  console.log('✅ Connected to progress WebSocket');
  
  // REQUIRED: Send authentication as first message
  ws.send(JSON.stringify({
    type: 'auth',
    token: token
  }));
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  // First message will be auth confirmation
  if (message.type === 'auth_success') {
    console.log('✅ Authenticated as user:', message.user_id);
    return;
  }
  
  // Handle progress events
  console.log('Event:', message.type, message.data);
};

ws.onerror = (error) => {
  console.error('❌ WebSocket error:', error);
};

ws.onclose = (event) => {
  console.log('🔌 Disconnected from progress WebSocket');
  if (event.code === 1008) {
    console.error('Authentication failed:', event.reason);
  }
};
```

### Authentication Flow

1. Client connects to WebSocket endpoint
2. Client sends `{"type": "auth", "token": "jwt_token"}` as **first message**
3. Server validates token and responds with `{"type": "auth_success", "user_id": "..."}`
4. Client can now receive progress events
5. If authentication fails or times out (5 seconds), connection is closed with code 1008

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
  type: 'auth_success' | 'queue_update' | 'progress' | 'job_complete' | 'job_failed';
  timestamp?: string;
  data?: any;
  user_id?: string;
}

export function useJobProgress(jobId: string, authToken: string) {
  const [progress, setProgress] = useState(0);
  const [queuePosition, setQueuePosition] = useState<number | null>(null);
  const [status, setStatus] = useState<'connecting' | 'queued' | 'running' | 'completed' | 'failed'>('connecting');
  const [error, setError] = useState<string | null>(null);
  const [outputUrl, setOutputUrl] = useState<string | null>(null);
  const [authenticated, setAuthenticated] = useState(false);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:7998/ws/progress');

    ws.onopen = () => {
      // Send authentication as first message
      ws.send(JSON.stringify({
        type: 'auth',
        token: authToken
      }));
    };

    ws.onmessage = (event) => {
      const message: ProgressEvent = JSON.parse(event.data);

      // Handle auth confirmation
      if (message.type === 'auth_success') {
        setAuthenticated(true);
        setStatus('queued');
        return;
      }

      // Only process events for our job
      if (message.data?.job_id !== jobId) return;

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

    ws.onclose = (event) => {
      if (event.code === 1008) {
        setError(`Authentication failed: ${event.reason}`);
        setStatus('failed');
      }
    };

    return () => ws.close();
  }, [jobId, authToken]);

  return { progress, queuePosition, status, error, outputUrl, authenticated };
}
```

### Vue 3 Composable

```typescript
import { ref, onMounted, onUnmounted } from 'vue';

export function useJobProgress(jobId: string, authToken: string) {
  const progress = ref(0);
  const queuePosition = ref<number | null>(null);
  const status = ref<'connecting' | 'queued' | 'running' | 'completed' | 'failed'>('connecting');
  const error = ref<string | null>(null);
  const outputUrl = ref<string | null>(null);
  const authenticated = ref(false);

  let ws: WebSocket | null = null;

  onMounted(() => {
    ws = new WebSocket('ws://localhost:7998/ws/progress');

    ws.onopen = () => {
      // Send authentication as first message
      ws!.send(JSON.stringify({
        type: 'auth',
        token: authToken
      }));
    };

    ws.onmessage = (event) => {
      const message = JSON.parse(event.data);

      // Handle auth confirmation
      if (message.type === 'auth_success') {
        authenticated.value = true;
        status.value = 'queued';
        return;
      }

      if (message.data?.job_id !== jobId) return;

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

Use `websocat` (or any WebSocket client) for manual testing:

```bash
# Set your JWT token (retrieve this from your auth system)
TOKEN="your_jwt_token_here"

# Connect and authenticate
echo '{"type":"auth","token":"'$TOKEN'"}' | websocat ws://localhost:7998/ws/progress

# Or in interactive mode (send auth message first, then other messages)
websocat ws://localhost:7998/ws/progress
# Then manually type: {"type":"auth","token":"your_jwt_token_here"}
# After auth_success response, you can send: {"type":"ping"}
```

Expected response after authentication:
```json
{"type":"auth_success","user_id":"your_user_id"}
```

Ping/pong test:
```json
{"type":"pong"}
```

## Security Considerations

1. **Authentication**: ✅ **IMPLEMENTED** - WebSocket endpoint requires JWT authentication
   - After connecting, clients must send `{"type":"auth","token":"jwt_token"}` as first message
   - User ID is derived from the authenticated token payload (`sub` claim)
   - Unauthorized clients receive close code 1008 with reason message
   - Authentication must complete within 5 seconds or connection is closed
   - Implementation:
     ```python
     from auth import decode_jwt_with_secret, decode_jwt_with_jwks
     
     @app.websocket("/ws/progress")
     async def websocket_progress(websocket: WebSocket):
         await websocket.accept()
         # Wait for auth message with 5 second timeout
         auth_message = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
         auth_data = json.loads(auth_message)
         token = auth_data.get("token")
         
         # PRODUCTION: Use verified decode methods only
         # Prefer decode_jwt_with_secret (HS256) or decode_jwt_with_jwks (RS256)
         # Do NOT use decode_supabase_jwt in production with untrusted clients
         # as it has an unverified fallback that can be exploited
         payload = decode_jwt_with_secret(token)  # or decode_jwt_with_jwks(token)
         if not payload:
             await websocket.close(code=1008, reason="Invalid token")
             return
         
         user_id = payload.get("sub")
     ```
   - **Browser Compatible**: This approach works with standard browser WebSocket API
   - **Security Warning**: Ensure `SUPABASE_JWT_SECRET` environment variable is set for production deployment to enable cryptographic JWT verification

2. **Authorization**: Events are only sent to job owners
   - Users can only see their own job progress
   - Job ownership is tracked via authenticated `user_id` from JWT
   - Each user receives events only for jobs they submitted

3. **DoS Protection**:
   - Rate limiting prevents event spam (100ms minimum between duplicate events)
   - Connection limits should be enforced at nginx/load balancer level
   - Consider implementing per-user connection limits
   - Rate limit cache is automatically cleaned up when jobs complete
   - Authentication timeout prevents resource exhaustion from unauthenticated connections

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
