# ProgressTracker Component

Real-time progress indicator for ComfyUI job generation.

## Features

- **Queue Position**: Shows current position in the generation queue
- **Progress Bar**: Visual progress indicator with percentage
- **ETA Timer**: Dynamic countdown based on elapsed time and progress
- **Preview Thumbnails**: Displays preview images if available from the backend
- **Smooth Animations**: Animated transitions for progress updates
- **Auto-refresh**: Polls backend every 2-3 seconds for updates

## Usage

### Basic Usage

```jsx
import ProgressTracker from './dashboard/ProgressTracker'

function MyComponent() {
  const [promptId, setPromptId] = useState(null)

  const handleJobComplete = (jobData) => {
    console.log('Job completed:', jobData)
    // Handle completion (e.g., show output, refresh UI)
  }

  return (
    <ProgressTracker
      promptId={promptId}
      onComplete={handleJobComplete}
    />
  )
}
```

### Integrated in QueueIndicator

The `ProgressTracker` is automatically shown for running jobs in the `QueueIndicator` popup. Click on a running job to expand/collapse the detailed progress view.

## Props

| Prop | Type | Required | Description |
|------|------|----------|-------------|
| `promptId` | string | Yes | The ComfyUI prompt ID to track |
| `onComplete` | function | No | Callback when job completes `(jobData) => void` |

## API Requirements

The component requires the following backend endpoints:

- `GET /comfyui/queue` - Returns queue status with running/pending jobs
- `GET /comfyui/job/{prompt_id}` - Returns job status and details

### Expected Response Format

**Queue Status:**
```json
{
  "running": [
    {
      "prompt_id": "abc123",
      "prompt": "sample prompt",
      "resolution": "480p",
      "aspect_ratio": "1:1",
      "num_frames": 41
    }
  ],
  "pending": [...],
  "total_running": 1,
  "total_pending": 0
}
```

**Job Status:**
```json
{
  "status": "running|queued|completed|failed",
  "prompt_id": "abc123",
  "output_video": "/path/to/video.mp4",
  "preview_url": "/path/to/preview.jpg"
}
```

## Styling

The component uses CSS variables from the global theme:

- `--bg-secondary`: Background color
- `--bg-input`: Input/progress bar background
- `--border-color`: Border color
- `--text-primary`: Primary text color
- `--text-secondary`: Secondary text color
- `--text-muted`: Muted text color

Colors are dynamically set based on job status:
- **Running**: Green (#22c55e)
- **Queued**: Yellow (#fbbf24)
- **Completed**: Blue (#3b82f6)

## Implementation Notes

- Progress percentage is estimated based on job status and elapsed time
- ETA calculation uses linear interpolation based on current progress
- The component auto-hides when `promptId` is null or job is completed
- Polling intervals: job status (2s), queue position (3s)
- Preview thumbnails display when `preview_url` is provided in job status

## Future Enhancements

- [ ] WebSocket support for real-time updates (instead of polling)
- [ ] Detailed node-by-node progress from ComfyUI
- [ ] Progress history/graph visualization
- [ ] Configurable polling intervals
- [ ] Pause/resume functionality
