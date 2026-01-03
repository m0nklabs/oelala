# Oelala Web Interface

## AI Video Generation Web Application

This web interface provides a modern dashboard for the Oelala AI video generation platform, powered by ComfyUI with Wan2.2 workflows.

## Features

### Video Generation
- 🎬 **Image to Video**: Transform images into videos with AI (Wan2.2 DisTorch2)
- 📝 **Text Prompts**: Positive and negative prompts for guidance
- ⚙️ **Advanced Controls**: Resolution, duration, FPS, aspect ratio
- 🎛️ **Model Selection**: GGUF model pairs (high/low noise)
- 🎨 **LoRA Support**: Apply LoRA models with adjustable strength
- 📋 **Presets**: Save and load workflow presets

### Media Management (My Media)
- 📁 **Gallery View**: Grid layout with thumbnails
- 🖼️ **Filter by Type**: All, Images, Videos, Favorites
- 💬 **Prompts Section**: Browse generation history with full metadata
- ⭐ **Favorites**: Mark and filter favorite items
- 🗑️ **Multi-select**: Bulk delete with keyboard shortcuts (Shift/Ctrl+click)
- 🔍 **Sort Options**: By date, name, size

### Prompt Viewing (NEW)
- 💬 **Prompt Bubble**: Hover over thumbnails to see prompt indicator
- 📋 **Popup Modal**: Click to view full prompt details:
  - ✨ Positive prompt with copy button
  - 🚫 Negative prompt
  - ⚙️ Generation settings (steps, CFG, seed, sampler, scheduler)
  - 🎨 LoRAs used with strength percentages
  - 📐 Resolution and video duration
  - 🤖 Model/checkpoint name

## Architecture

### Backend (FastAPI)
- **Framework**: FastAPI with automatic API documentation
- **ComfyUI Client**: WebSocket-based workflow execution
- **Static Serving**: Built frontend served from `/`
- **CORS**: Configured for development frontend
- **Endpoints**: REST API for generation, media listing, presets

### Frontend (React + Vite)
- **Framework**: React 18 with modern hooks
- **Build Tool**: Vite for fast development
- **Styling**: CSS Variables with dark theme
- **Icons**: Lucide React icons
- **State**: Local component state + localStorage persistence

## Quick Start

### Prerequisites
- Python 3.10+ with GPU venv (`/home/flip/venvs/gpu`)
- Node.js 18+ and npm
- CUDA-compatible GPU (RTX 3060+ recommended)
- ComfyUI running on port 8188

### Running the Application

**Option 1: Production (recommended)**
```bash
# Build frontend
cd /home/flip/oelala/src/frontend
npm run build

# Start backend (serves built frontend)
source /home/flip/venvs/gpu/bin/activate
cd /home/flip/oelala/src/backend
uvicorn app:app --host 0.0.0.0 --port 7998
```

**Option 2: Development (hot reload)**
```bash
# Terminal 1 - Backend
source /home/flip/venvs/gpu/bin/activate
cd /home/flip/oelala/src/backend
uvicorn app:app --host 0.0.0.0 --port 7998 --reload

# Terminal 2 - Frontend dev server
cd /home/flip/oelala/src/frontend
npm run dev
```

### Access the Application

- **Web Interface**: http://localhost:7998 (production) or http://localhost:5174 (dev)
- **API Documentation**: http://localhost:7998/docs
- **ComfyUI**: http://localhost:8188

## API Endpoints

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Backend + ComfyUI health status |
| `/loras` | GET | List available LoRA models |
| `/unet-models` | GET | List GGUF unet model pairs |
| `/api/presets` | GET | List workflow presets |
| `/extract-metadata` | POST | Extract prompt from image |
| `/wan22/image-to-video` | POST | Generate video via ComfyUI |

### Media Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/list-comfyui-media` | GET | List media with metadata |
| `/comfyui-output/{file}` | GET | Serve generated files |
| `/delete-comfyui-media` | DELETE | Delete selected files |

### Media Listing Parameters

```
GET /list-comfyui-media?type=video&include_metadata=true&hide_start_images=true
```

- `type`: `all`, `image`, `video`
- `include_metadata`: Include prompt/generation info
- `hide_start_images`: Hide source images for videos
- `grouped`: Group by timestamp (for video/image pairs)

### Metadata Response

When `include_metadata=true`, each item includes:

```json
{
  "filename": "video_00001.mp4",
  "type": "video",
  "metadata": {
    "has_metadata": true,
    "positive_prompt": "a beautiful scene...",
    "negative_prompt": "worst quality, blurry",
    "steps": 6,
    "cfg": 1.0,
    "seed": 123456,
    "sampler": "uni_pc",
    "scheduler": "normal",
    "width": 576,
    "height": 1024,
    "model": "wan2.2_i2v_14B_Q6_K.gguf",
    "loras": [
      {"name": "style_lora.safetensors", "strength": 1.5}
    ]
  }
}
```

## File Structure

```
/home/flip/oelala/
├── src/
│   ├── backend/
│   │   ├── app.py                    # FastAPI application
│   │   ├── comfyui_client.py         # ComfyUI WebSocket client
│   │   └── requirements.txt          # Python dependencies
│   └── frontend/
│       ├── package.json              # Node dependencies
│       ├── vite.config.js            # Vite configuration
│       ├── dist/                     # Built frontend (served by backend)
│       └── src/
│           ├── App.jsx               # Main React app
│           ├── main.jsx              # React entry point
│           ├── dashboard/
│           │   ├── Dashboard.jsx     # Dashboard layout
│           │   ├── nav.js            # Navigation config
│           │   └── tools/
│           │       ├── ImageToVideoTool.jsx  # I2V interface
│           │       └── MyMediaTool.jsx       # Media gallery + prompts
│           └── components/
│               ├── PresetSelector.jsx        # Preset dropdown
│               └── VideoGenerator.jsx        # Legacy generator
├── ComfyUI/
│   ├── output/                       # Generated media
│   └── models/                       # AI models
├── workflows/
│   ├── registry.json                 # Preset definitions
│   └── ImageToVideo/                 # I2V workflow templates
└── docs/                             # Documentation
```

## Configuration

### Backend Configuration
- **Host**: 0.0.0.0 (all interfaces)
- **Port**: 7998
- **ComfyUI**: http://localhost:8188
- **Output Directory**: `/home/flip/oelala/ComfyUI/output/`

### Frontend Configuration
- **Dev Port**: 5174
- **API Proxy**: Proxied to backend in dev mode
- **Theme**: Dark mode with CSS variables
- **Host**: 192.168.1.2
- **Port**: 3000
- **API proxy**: `/api` routes proxied to backend

## Troubleshooting

### Common Issues

1. **backend not starting**:
   - Ensure Python virtual environment is activated
   - Check if port 7999 is available
   - Verify Wan2.2 model is accessible

2. **frontend not loading**:
   - Check if port 3000 is available
   - Ensure npm dependencies are installed
   - Verify backend is running for API calls

3. **Video generation fails**:
   - Check GPU memory availability (16GB+ recommended)
   - Verify input image format and size
   - Check backend logs for detailed error messages

4. **CORS errors**:
   - Ensure backend CORS settings allow frontend origin
   - Check network connectivity between services

### Logs and Debugging

- **backend Logs**: Check terminal output when running `Python app.py`
- **frontend Logs**: Check browser developer console
- **Health Check**: Visit `/health` endpoint for system Status

## development

### Adding New Features

1. **backend**: Add new endpoints in `app.py` or create separate API modules
2. **frontend**: Create new components in `src/components/`
3. **Styling**: Add styles to component CSS files or global styles

### Building for Production

```bash
# Build frontend
cd src/frontend
npm run build

# The built files will be in src/frontend/dist/
```

## Security Notes

- file uploads are validated for image types only
- CORS is configured for specific origins in production
- No authentication implemented (add as needed)
- file paths are sanitized to prevent directory traversal

## Future Enhancements

- [ ] User authentication and session Management
- [ ] Batch video generation
- [ ] Video editing and post-processing
- [ ] Integration with OpenPose for pose-guided generation
- [ ] Real-time progress updates
- [ ] Video gallery and Management
- [ ] Advanced parameter controls
- [ ] Mobile-responsive improvements

## Recent Updates

- **January 3, 2026**: Added Prompts section with full metadata display
  - Prompt bubble (💬) on thumbnails
  - Popup modal with prompts, settings, LoRAs, model info
  - Dedicated prompts list view
  - Extended metadata extraction (sampler, scheduler, resolution, LoRAs)
- **January 2, 2026**: Added LoRA support, model pair selection, presets
- **December 2025**: ComfyUI integration with DisTorch2 workflows
