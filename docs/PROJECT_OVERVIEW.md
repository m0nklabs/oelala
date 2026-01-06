# Oelala Project Documentation

## Overview

**Oelala** is an AI-driven video generation platform with a modern web dashboard. The project integrates ComfyUI with Wan2.2 workflows for high-quality image-to-video generation through an intuitive web interface.

### Project Goals
- Develop an integrated pipeline for AI video generation via ComfyUI
- Build a modern dashboard UI for video generation control
- Provide comprehensive media management with prompt history
- Support multiple LoRA models and GGUF model pairs
- Enable workflow presets for quick configuration

### Core Technologies
- **Wan2.2**: Advanced image-to-video generation (14B parameter models)
- **ComfyUI**: Workflow engine with DisTorch2 dual-pass sampling
- **FastAPI**: High-performance web API backend
- **React + Vite**: Modern frontend framework
- **GGUF Models**: Quantized models for efficient VRAM usage
- **LoRA**: Fine-tuning support for style customization

---

## Project Status & Voortgang

### ✅ Completed Tasks

#### 1. Project Setup (Completed)
- **Workspace**: `/home/flip/oelala/` created and configured
- **Git Repository**: GitHub repo at m0nklabs/oelala
- **Dependencies**: CUDA 12.9, cuDNN 8.9.7, PyTorch 2.5.1+cu121
- **Python Environment**: `/home/flip/venvs/gpu` (canonical GPU venv)

#### 2. ComfyUI Integration (Completed)
- **Port**: 8188 with WebSocket support
- **Workflows**: DisTorch2 dual-pass I2V workflow
- **Models**: GGUF quantized models (Q6_K) for 14B parameters
- **LoRA Support**: 53+ LoRA files with category organization
- **Output**: `/home/flip/oelala/ComfyUI/output/`

#### 3. Web Dashboard (Completed)
- **Backend**: FastAPI on port 7998, serves built frontend
- **Frontend**: React + Vite with sidebar navigation
- **Features**:
  - Image to Video generation
  - Model pair selection (high/low noise)
  - LoRA selection with strength control
  - Preset system for workflow configurations
  - My Media gallery with filters
  - **Prompts section** with full metadata display (NEW)

#### 4. My Media Features (Completed)
- **Gallery View**: Grid layout with thumbnails
- **Filters**: All, Images, Videos, Favorites, Prompts
- **Prompt Bubble**: 💬 icon on thumbnails with prompts
- **Prompt Popup**: Full generation details modal
  - Positive/negative prompts with copy
  - Steps, CFG, seed, sampler, scheduler
  - LoRAs used with strength percentages
  - Model name and resolution
  - Video duration
- **Metadata Extraction**: From PNG workflow JSON (images + videos)

**Technical Details:**
```Python
import openpose.pyopenpose as on

# Initialize wrapper
wrapper = on.WrapperPython()

# Configure parameters
params = {
    "model_folder": "/home/flip/oelala/openpose/models/",
    "net_resolution": "320x176",
    "face": False,
    "hand": False
}
wrapper.configure(params)
wrapper.start()

# Pose estimation uitvoeren
datum = on.Datum()
datum.cvInputData = image
datumVector = on.VectorDatum()
datumVector.append(datum)
wrapper.emplaceAndPop(datumVector)

# Resultaten ophalen
keypoints = datum.poseKeypoints
```

### 🔄 Current Status (January 2026)

#### Dashboard UI
- **Status**: Fully functional with sidebar navigation
- **Tools Available**:
  - ✅ Image to Video (Wan2.2 14B DisTorch2)
  - ✅ Text to Video (Wan2.2)
  - ✅ Text to Image (SDXL, Flux)
  - ✅ Video to Video (style transfer)
  - ✅ Image Upscaler (RealESRGAN)
  - ✅ Video Upscaler
  - ✅ Frame Interpolation (RIFE)
  - ✅ Face Swap (InSwapper)
  - ✅ Lip Sync (LatentSync)
  - ✅ Voice Cloning (F5-TTS)
  - ✅ Speech to Video (TTS + Lip Sync)
  - ✅ My Media (gallery with filters)
  - ✅ Public Gallery (community content)
  - ✅ LoRA Training (placeholder)

#### User System
- **Status**: Fully implemented
- **Auth**: Supabase (Google, GitHub OAuth)
- **Credits**: Stripe payments, per-generation costs
- **Access Levels**: Guest, User, Admin

#### Storage System
- **Status**: oelala-storage running (Go service)
- **User Buckets**: Each user has isolated storage
- **Quota**: Planned (not yet enforced)

#### ComfyUI Integration
- **Status**: Fully integrated with WebSocket progress
- **Workflow**: DisTorch2 dual-pass sampling
- **Multi-GPU**: RTX 5060 Ti (16GB) + RTX 3060 (12GB)
- **Models**: GGUF Q6_K quantized (14B parameters)
- **LoRAs**: 50+ organized by category

#### Gallery & Publishing
- **Status**: Fully implemented
- **Features**: Publish, like, view counts
- **NSFW Control**: SFW-only for guests

### 📋 Next Priority Tasks

1. **Auto-Upload** (#7, #15): Generated content → user storage
2. **Storage Quota** (#33): Track and display usage
3. **Retention Policy** (#71): 30-day free tier cleanup

#### 8. Production Enhancements
- ❌ User authentication and session Management (pending)
- ❌ Video gallery and Management system (pending)
- ❌ Advanced parameter controls and presets (pending)
- ❌ Mobile-responsive improvements (pending)
- ❌ API rate limiting and security features (pending)

---

## technical Architectuur

### Directory Structure
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
│       ├── dist/                     # Built frontend
│       └── src/
│           ├── App.jsx               # Main React app
│           ├── dashboard/
│           │   ├── Dashboard.jsx     # Dashboard layout
│           │   ├── nav.js            # Navigation config
│           │   └── tools/
│           │       ├── ImageToVideoTool.jsx
│           │       └── MyMediaTool.jsx
│           └── components/
│               ├── PresetSelector.jsx
│               └── VideoGenerator.jsx
├── ComfyUI/
│   ├── output/                       # Generated media
│   └── models/
│       ├── diffusion_models/         # GGUF unet models
│       ├── loras/                    # LoRA files
│       ├── clip/                     # Text encoders
│       └── vae/                      # VAE models
├── workflows/
│   ├── registry.json                 # Preset definitions
│   └── ImageToVideo/                 # Workflow templates
├── docs/                             # Documentation
└── README.md                         # Project readme
```

### Dependencies & Requirements

#### System Requirements
- **OS**: Linux (Ubuntu 24.04)
- **GPU**: NVIDIA RTX 3060+ with 12GB+ VRAM
- **CUDA**: 12.1+ with cuDNN 8.9+
- **Python**: 3.10+ with venv
- **Node.js**: 18+ with npm

#### Python Environment
- **Virtual Environment**: `/home/flip/venvs/gpu/`
- **Key Packages**:
  - fastapi, uvicorn[standard], python-multipart
  - pillow, aiofiles, websockets
  - torch (for metadata extraction)

#### Node.js Environment (Frontend)
- **Node Version**: 18+ (v22.x recommended)
- **Package Manager**: npm
- **Key Packages**:
  - react ^18.2.0, react-dom ^18.2.0
  - lucide-react (icons)
  - @vitejs/plugin-react (Vite React plugin)

### Build & Installatie

#### OpenPose Build
```bash
cd /home/flip/oelala/openpose/build
cmake .. -DBUILD_PYTHON=ON
make -j$(nproc)
```

#### Python Environment Setup
```bash
# Python 3.10 installeren
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt install python3.10 python3.10-dev python3.10-venv

# Virtual environment aanmaken
python3.10 -m venv /home/flip/openpose_py310
source /home/flip/openpose_py310/bin/activate
pip install setuptools pybind11 numpy opencv-Python
```

#### Libraries Installation
```bash
# OpenPose libraries kopiëren to virtual environment
cp /home/flip/oelala/openpose/build/src/openpose/*/*.so* /home/flip/openpose_py310/lib/python3.10/site-packages/
cp /home/flip/oelala/openpose/build/caffe/lib/libcaffe.so* /home/flip/openpose_py310/lib/python3.10/site-packages/
```

---

## Web Interface Architecture

### backend (FastAPI)
- **Framework**: FastAPI with automatic OpenAPI documentation
- **Server**: Uvicorn ASGI server with auto-reload
- **endpoints**:
  - `POST /generate`: Video generation from image upload
  - `GET /videos/{filename}`: Download generated videos
  - `GET /images/{filename}`: Download uploaded images
  - `GET /health`: System health and Status
  - `GET /list-videos`: list all generated videos
- **Features**: file upload, validation, error handling, CORS support
- **Integration**: Direct connection to Wan2.2 generator

### frontend (React + Vite)
- **Framework**: React 18 with modern hooks and functional components
- **Build tool**: Vite for fast development and optimized production builds
- **Components**:
  - `App.jsx`: Main application component
  - `VideoGenerator.jsx`: Core video generation interface
- **Features**: Drag-and-drop upload, real-time preview, progress tracking
- **Styling**: Custom CSS with responsive design and modern UI
- **HTTP Client**: Axios for API communication

### API Integration
```javascript
// Example: generate video from image
const FormData = new FormData();
FormData.append('file', imageFile);
FormData.append('prompt', 'dancing gracefully');
FormData.append('num_frames', 16);

const response = await axios.post('/api/generate', FormData);
const videoUrl = response.data.video_url;
```

### Deployment
- **development**: `npm run dev` (port 5174) + `python app.py` (port 7998)
- **Production**: `npm run build` + serve static files
- **URLs**:
  - frontend: http://192.168.1.2:5174
  - backend: http://192.168.1.2:7998
  - API Docs: http://192.168.1.2:7998/docs

## OpenPose technical Details

### Model Specificaties
- **Model**: BODY_25
- **Keypoints**: 25 lichaamsdelen
- **Output Format**: (x, y, confidence) per keypoint
- **Resolution**: Configureerbaar (standaard 320x176)
- **Performance**: GPU-accelerated

### BODY_25 Keypoints Mapping
```
0: Nose
1: Neck
2: RShoulder
3: RElbow
4: RWrist
5: LShoulder
6: LElbow
7: LWrist
8: MidHip
9: RHip
10: RKnee
11: RAnkle
12: LHip
13: LKnee
14: LAnkle
15: REye
16: LEye
17: REar
18: LEar
19: LBigToe
20: LSmallToe
21: LHeel
22: RBigToe
23: RSmallToe
24: RHeel
```

### use Voorbeelden

#### Basis Pose Estimation
```Python
import cv2
import openpose.pyopenpose as on

# Afbeelding laden
image = cv2.imread('person.jpg')

# OpenPose initialiseren
wrapper = on.WrapperPython()
params = {"model_folder": "/home/flip/oelala/openpose/models/"}
wrapper.configure(params)
wrapper.start()

# Pose estimation
datum = on.Datum()
datum.cvInputData = image
datumVector = on.VectorDatum()
datumVector.append(datum)
wrapper.emplaceAndPop(datumVector)

# Resultaten
keypoints = datum.poseKeypoints
print(f"Gedetecteerd: {keypoints.shape[0]} personen")
```

#### Visualisatie
```Python
# Keypoints tekenen
for person in keypoints:
    for i, keypoint in enumerate(person):
        x, y, confidence = keypoint
        if confidence > 0.1:
            cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)

cv2.imwrite('result.jpg', image)
```

---

## Project Roadmap

### ✅ Phase 1: Core Components (Completed)
- ✅ OpenPose pose estimation with Python bindings
- ✅ Wan2.2 image-to-video generation integration
- ✅ Web interface with FastAPI backend and React frontend
- ✅ Complete pipeline from image upload to video generation
- ✅ GPU acceleration and CUDA support
- ✅ Production-ready deployment scripts

### 🔄 Phase 2: Integration & Enhancement (Current - Q4 2025)
- 🔄 Pose-guided video generation (OpenPose + Wan2.2)
- 🔄 LoRA fine-tuning for consistent avatars
- 🔄 Batch processing capabilities
- 🔄 Real-time pose estimation
- 🔄 Advanced video post-processing

### 📋 Phase 3: Production & Scale (Q1 2026)
- ❌ User authentication and session Management
- ❌ Video gallery and Management system
- 📋 Mobile-responsive optimizations
- ❌ API rate limiting and security
- 📋 Performance monitoring and analytics

### 🎯 Phase 4: Advanced Features (Q2 2026)
- 🎯 Multi-workspace support
- 🎯 Custom model training pipelines
- 🎯 Real-time webcam processing
- 🎯 Multi-language support
- 🎯 Plugin architecture for extensibility

## Quick Start Guide

### Production Setup
```bash
# 1. Build frontend
cd /home/flip/oelala/src/frontend
npm install && npm run build

# 2. Start backend (serves frontend)
source /home/flip/venvs/gpu/bin/activate
cd /home/flip/oelala/src/backend
uvicorn app:app --host 0.0.0.0 --port 7998

# 3. Access at http://localhost:7998
```

### Development Setup
```bash
# Terminal 1 - Backend with hot reload
source /home/flip/venvs/gpu/bin/activate
cd /home/flip/oelala/src/backend
uvicorn app:app --host 0.0.0.0 --port 7998 --reload

# Terminal 2 - Frontend dev server
cd /home/flip/oelala/src/frontend
npm run dev
# Access at http://localhost:5174
```

### Prerequisites
- ComfyUI running on port 8188
- GGUF models in `ComfyUI/models/diffusion_models/`
- LoRAs in `ComfyUI/models/loras/`

---

## development & Contribution

### development Setup
1. Clone and setup the complete environment
2. Install Python 3.10 and Node.js dependencies
3. Build OpenPose with Python bindings
4. Run validation tests for all components
5. Start development servers for backend and frontend

### Coding Standards
- **Python**: PEP 8 compliant with type hints
- **JavaScript/React**: ESLint configuration with modern standards
- **C++**: Google C++ Style Guide for OpenPose components
- **Documentation**: Comprehensive Markdown documentation
- **Version Control**: Git with descriptive commit messages

### testing Strategy
- **Unit Tests**: Individual component testing
- **Integration Tests**: Pipeline and API testing
- **Performance Tests**: GPU utilization and processing speed
- **Visual Tests**: Output quality validation
- **End-to-End Tests**: Complete user workflow testing

### component Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   React UI      │───▶│   FastAPI       │───▶│   Wan2.2 AI     │
│   (frontend)    │    │   backend       │    │   Generator     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  file upload    │    │   API routes    │    │   Video Gen     │
│  & preview      │    │   & Validation  │    │   Pipeline      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

---

## Contact & Resources

### Documentation
- **Main README**: `/home/flip/oelala/README.md`
- **Web Interface**: `/home/flip/oelala/WEB_INTERFACE_README.md`
- **Wan2.2 Guide**: `/home/flip/oelala/WAN2_README.md`
- **OpenPose Guide**: `/home/flip/oelala/OPENPOSE_TECHNICAL_GUIDE.md`
- **Project Plan**: `/home/flip/oelala/PROJECT_PLAN.md`

### Key Files
- **backend App**: `src/backend/app.py`
- **frontend App**: `src/frontend/src/App.jsx`
- **Video Generator**: `wan2_generator.py`
- **Startup Script**: `start_web.sh`
- **test Suite**: `test_web_interface.py`

### support
- **Repository**: `/home/flip/oelala/`
- **Issues**: use GitHub issues for bug reports
- **Wiki**: Project documentation in `/docs/`
- **Logs**: Check terminal output for debugging

---

## System Requirements Summary

### Hardware
- **GPU**: NVIDIA RTX 3060 or better (11.8GB+ VRAM)
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ free space for models and generated content
- **Network**: Stable internet for model downloads

### Software
- **OS**: Linux (Ubuntu 24.04 recommended)
- **Python**: 3.10.18 with virtual environment
- **Node.js**: 16+ (18+ recommended)
- **CUDA**: 12.1+ with cuDNN 8.9+
- **Git**: Latest version for version control

---

*Last Updated: January 3, 2026*
*Status: Dashboard fully operational with ComfyUI integration*
*Version: 2.0.0 - Dashboard UI with Prompts feature*
*Progress: Dashboard UI ✅ | Image-to-Video ✅ | My Media ✅ | Prompts ✅*
