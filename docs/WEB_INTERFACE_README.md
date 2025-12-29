# Oelala Web Interface

## AI Video Generation Web Application

This web interface provides an easy-to-use frontend for the Oelala AI video generation pipeline, powered by Wan2.2 and OpenPose.

## Features

- 🎨 **Image upload**: Drag & drop or click to upload images
- 🎬 **Video Generation**: Transform images into videos with AI
- 📝 **Text Prompts**: Add custom prompts to guide video generation
- ⚙️ **Parameter Control**: Adjust number of frames (8-32)
- 📥 **Download Videos**: Download generated videos directly
- 🔄 **Real-time Status**: Live backend health monitoring

## Architecture

### backend (FastAPI)
- **Framework**: FastAPI with automatic API documentation
- **AI Engine**: Wan2.2 Image-to-Video generation
- **file Handling**: Secure upload/download with validation
- **CORS**: Configured for frontend communication
- **Health Checks**: Real-time Status monitoring

### frontend (React + Vite)
- **Framework**: React 18 with modern hooks
- **Build tool**: Vite for fast development
- **Styling**: Custom CSS with responsive design
- **Icons**: Lucide React icons
- **HTTP Client**: Axios for API communication

## Quick Start

### Prerequisites
- Python 3.10 with virtual environment
- Node.js 16+ and npm
- CUDA-compatible GPU (recommended)

### Installation

1. **backend dependencies**:
```bash
cd /home/flip/openpose_py310
source bin/activate
pip install -r /home/flip/oelala/src/backend/requirements.txt
```

2. **frontend dependencies**:
```bash
cd /home/flip/oelala/src/frontend
npm install
```

### Running the Application

**Option 1: Automated Startup**
```bash
cd /home/flip/oelala
./start_web.sh
```

**Option 2: Manual Startup**

Terminal 1 - backend:
```bash
source /home/flip/venvs/gpu/bin/activate
cd /home/flip/oelala/src/backend
python app.py
```

Terminal 2 - frontend:
```bash
cd /home/flip/oelala/src/frontend
npm run dev
```

### Access the Application

- **frontend**: http://192.168.1.2:5174
- **backend API**: http://192.168.1.2:7998
- **API Documentation**: http://192.168.1.2:7998/docs

## API endpoints

### Core endpoints

- `GET /` - API information and available endpoints
- `GET /health` - backend health and model Status
- `POST /generate` - generate video from image
  - **Parameters**:
    - `file`: Image file (required)
    - `prompt`: Text prompt (optional)
    - `num_frames`: Number of frames (8-32, default: 16)
    - `output_filename`: Custom output filename (optional)
- `GET /videos/{filename}` - Download generated video
- `GET /images/{filename}` - Download uploaded image
- `GET /list-videos` - list all generated videos

### Example API Usage

```Python
import requests

# upload and generate video
files = {'file': open('person.jpg', 'rb')}
data = {'prompt': 'dancing gracefully', 'num_frames': 16}
response = requests.post('http://192.168.1.2:7998/generate', files=files, data=data)
result = response.json()

# Download video
video_url = f"http://192.168.1.2:7998{result['video_url']}"
```

## file Structure

```
/home/flip/oelala/
├── src/
│   ├── backend/
│   │   ├── app.py              # FastAPI application
│   │   ├── requirements.txt    # Python dependencies
│   │   └── api/                # API endpoints directory
│   └── frontend/
│       ├── package.json        # Node dependencies
│       ├── vite.config.js      # Vite configuration
│       ├── index.html          # HTML template
│       └── src/
│           ├── App.jsx         # Main React app
│           ├── main.jsx        # React entry point
│           ├── App.CSS         # App styles
│           ├── index.CSS       # Global styles
│           └── components/
│               ├── VideoGenerator.jsx    # Main component
│               └── VideoGenerator.CSS    # component styles
├── uploads/                    # Uploaded images
├── generated/                  # Generated videos
├── scripts/start_web.sh        # Startup script
└── WAN2_README.md            # Wan2.2 documentation
```

## Configuration

### backend Configuration
- **Host**: 192.168.1.2
- **Port**: 7999
- **upload Directory**: `/home/flip/oelala/uploads/`
- **Output Directory**: `/home/flip/oelala/generated/`
- **Max file Size**: Limited by FastAPI defaults

### frontend Configuration
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

## Recent edits and provenance

- 2025-09-07: Updated project network IPs to use 192.168.1.2 and added notes about the 7000-7999 port convention.
- 2025-09-07: Fixed example API usage URLs to use 192.168.1.2.

Edits performed by: GitHub Copilot

Notes: Edited to align documentation examples and network notes with the project's LAN convention; see `DOCS_CHANGELOG.md` for a full record.
