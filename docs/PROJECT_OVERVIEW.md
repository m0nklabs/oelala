# Oelala Project Documentation

## Overview

**Oelala** is an AI-driven project for video generation, pose estimation, and realistic avatar creation. The project combines state-of-the-art AI models to generate consistent and realistic AI avatars in videos through an intuitive web interface.

### Project Goals
- Develop an integrated pipeline for AI video generation
- Implement pose estimation for improved movements
- Create realistic and consistent AI avatars
- Build a standalone, portable web application
- Provide user-friendly interface for AI video creation

### Core Technologies
- **Wan2.2**: Advanced image-to-video generation (Wan-AI/Wan2.2-I2V-A14B)
- **OpenPose**: Pose estimation and keypoint detection
- **FastAPI**: High-performance web API backend
- **React**: Modern frontend framework with Vite
- **PyTorch**: Deep learning framework with CUDA acceleration
- **Python/C++**: Programming languages for prototyping and production

---

## Project Status & Voortgang

### ✅ Completed Tasks

#### 1. Project Setup (Completed)
- **Workspace**: `/home/flip/oelala/` created and configured
- **Git Repository**: Initialized with proper structure
- **dependencies**: CUDA 12.9, cuDNN 8.9.7, PyTorch 2.5.1+cu121
- **Build System**: CMake configuration for OpenPose

#### 2. OpenPose Python Bindings (Completed)
- **Python Environment**: Python 3.10.18 virtual environment (`/home/flip/openpose_py310/`)
- **Build**: Successful compilation with Python bindings enabled
- **Libraries**: All shared libraries properly installed
- **testing**: Functional demo scripts and validation tests
- **API**: Fully working Python interface for pose estimation

#### 3. Wan2.2 Integration (Completed)
- **Model**: Wan-AI/Wan2.2-I2V-A14B (1.3B parameters, 14B text encoder)
- **Pipeline**: Image-to-video generation with optional text prompts
- **Environment**: PyTorch 2.5.1 with CUDA 12.1 support
- **dependencies**: diffusers, transformers, accelerate, safetensors, opencv-Python
- **testing**: Validation scripts, demo implementation, and error handling
- **Performance**: GPU-accelerated with RTX 3060 (11.8GB VRAM)

#### 4. Web Interface (Completed)
- **backend**: FastAPI server with comprehensive REST API
- **frontend**: React 18 + Vite application with modern UI
- **Features**: Image upload, video generation, real-time progress, download
- **API endpoints**: `/generate`, `/videos/{filename}`, `/health`, `/list-videos`
- **Integration**: Seamless connection between frontend and Wan2.2 pipeline
- **Deployment**: Ready for production with automated startup script

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

### 🔄 Current Status

#### OpenPose component
- **Status**: Fully functional with Python 3.10 bindings
- **test Results**: All demo scripts working correctly
- **Performance**: GPU-accelerated with CUDA support
- **Models**: BODY_25 model loaded and operational
- **Output**: 25 keypoints per person (x, y, confidence scores)
- **Integration**: Ready for pose-guided video generation

#### Wan2.2 component
- **Status**: Fully integrated and operational
- **Model**: Wan-AI/Wan2.2-I2V-A14B loaded on-demand
- **Capabilities**: Image-to-video generation with text prompts
- **Performance**: ~16GB VRAM recommended, working with 11.8GB RTX 3060
- **Features**: 8-32 frame videos, customizable prompts, MP4 output
- **Error Handling**: Graceful fallback and user-friendly error messages

#### Web Interface
- **backend**: FastAPI server running on port 7999
- **frontend**: React application running on port 3000
- **Features**: Drag-and-drop upload, real-time generation, video download
- **API**: RESTful endpoints with automatic documentation
- **Status**: Production-ready with comprehensive error handling
- **testing**: All components validated and working

#### Demo Scripts
- `demo_openpose.py`: OpenPose pose estimation demo
- `test_real_image.py`: Real image testing with OpenPose
- `test_openpose.py`: Comprehensive OpenPose validation
- `wan2_generator.py`: Wan2.2 video generation class
- `demo_wan2.py`: Wan2.2 demo with placeholder image
- `test_wan2_setup.py`: Wan2.2 environment validation
- `test_web_interface.py`: Complete web interface testing

### 📋 Planned Tasks

#### 5. Pose-Video Integration
- 🔄 Combine OpenPose keypoints with Wan2.2 video generation (basis ready)
- 🔄 Implement pose-guided animation for more realistic movements (pending)
- 🔄 test avatar consistency across video frames (pending)
- 🔄 Develop joint pose estimation and video generation pipeline (pending)

#### 6. LoRA Fine-tuning
- 🔄 Implement LoRA (Low-Rank Adaptation) for model customization (placeholder aanwezig)
- 🔄 Create consistent avatar styles and characteristics (pending)
- 🔄 Fine-tune Wan2.2 for specific use cases (pending)
- 🔄 Optimize model performance and memory usage (pending)

#### 7. Advanced Pipeline Features
- ❌ Batch processing for multiple images simultaneously (pending)
- ❌ Real-time pose estimation with webcam input (pending)
- 🔄 Quality optimization and performance improvements (lopend)
- ❌ Advanced video post-processing and effects (pending)

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
├── src/                          # Main source code
│   ├── backend/                  # FastAPI backend
│   │   ├── app.py               # Main FastAPI application
│   │   └── requirements.txt     # Python dependencies
│   └── frontend/                # React frontend
│       ├── src/
│       │   ├── App.jsx          # Main React application
│       │   ├── main.jsx         # React entry point
│       │   ├── App.CSS          # Application styles
│       │   ├── index.CSS        # Global styles
│       │   └── components/
│       │       └── VideoGenerator.jsx  # Main video generation component
│       ├── package.json         # Node.js dependencies
│       ├── vite.config.js       # Vite configuration
│       └── index.html           # HTML template
├── uploads/                     # Uploaded images directory
├── generated/                   # Generated videos directory
├── CMakeLists.txt              # CMake build configuration
├── README.md                   # Project overview
├── PROJECT_PLAN.md            # Detailed project plan
├── KEYWORDS.md                # Project keywords
├── OPENPOSE_TECHNICAL_GUIDE.md # OpenPose documentation
├── WAN2_README.md             # Wan2.2 integration guide
├── WEB_INTERFACE_README.md    # Web interface documentation
├── wan2_generator.py          # Wan2.2 video generator
├── demo_wan2.py              # Wan2.2 demo script
├── test_wan2_setup.py        # Wan2.2 validation
├── test_web_interface.py     # Web interface testing
├── start_web.sh              # Automated startup script
├── docs/                      # Project documentation
├── src/                       # C++ source code (legacy)
│   └── main.cpp              # Basic OpenPose demo
├── Python/                    # Python scripts
│   ├── demo_openpose.py      # Pose estimation demo
│   ├── test_real_image.py    # Real image testing
│   └── test_openpose.py      # Comprehensive validation
├── build/                     # Build artifacts
├── models/                    # AI models directory
└── openpose/                  # OpenPose source code
    ├── build/                # OpenPose build directory
    ├── models/               # Pose estimation models
    └── 3rdparty/             # Third-party dependencies
```

### dependencies & Requirements

#### Systeem Requirements
- **OS**: Linux (Ubuntu 24.04)
- **CUDA**: 12.9 with cuDNN 8.9.7
- **Python**: 3.10.18 (dead snakes PPA)
- **Compiler**: GCC 13.3.0
- **Build System**: CMake 3.28+

#### Python Environment
- **Virtual Environment**: `/home/flip/openpose_py310/`
- **Python Version**: 3.10.18 (via deadsnakes PPA)
- **Key Packages**:
  - setuptools, pybind11, numpy
  - torch 2.5.1+cu121, torchvision
  - diffusers, transformers, accelerate, safetensors
  - opencv-Python, Pillow
  - fastapi, uvicorn[standard], Python-multipart, pydantic
  - openpose (compiled Python bindings)

#### Node.js Environment (frontend)
- **Node Version**: 16+ (recommended 18+)
- **Package Manager**: npm
- **Key Packages**:
  - react ^18.2.0, react-dom ^18.2.0
  - axios ^1.6.0 (HTTP client)
  - lucide-react ^0.294.0 (icons)
  - @vitejs/plugin-react ^4.2.1 (Vite React plugin)
  - eslint (code linting)

#### C++ dependencies
- **OpenPose**: v1.7.0 with Python bindings
- **Boost**: 1.83 (system libraries)
- **OpenCV**: 4.6.0 with CUDA support
- **HDF5**: System libraries for data storage
- **Protobuf**: 3.21.12 for serialization

#### C++ dependencies
- **OpenPose**: v1.7.0
- **Boost**: 1.83
- **OpenCV**: 4.6.0
- **HDF5**: Systeem libraries
- **Protobuf**: 3.21.12

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

### Automated Setup
```bash
cd /home/flip/oelala
./start_web.sh
```
This will start both the backend (port 7999) and frontend (port 3000).

### Manual Setup

#### 1. backend Setup
```bash
# Activate Python environment
cd /home/flip/openpose_py310
source bin/activate

# Install dependencies
pip install -r /home/flip/oelala/src/backend/requirements.txt

# Start backend
cd /home/flip/oelala/src/backend
Python app.py
```

#### 2. frontend Setup
```bash
# Install dependencies
cd /home/flip/oelala/src/frontend
npm install

# Start development server
npm run dev
```

#### 3. Access the Application
- **Web Interface**: http://192.168.1.2:5174
- **API Documentation**: http://192.168.1.2:7998/docs
- **Health Check**: http://192.168.1.2:7998/health

### testing
```bash
# Run comprehensive test
cd /home/flip/oelala
python tests/test_web_interface.py

# test individual components
python tests/test_wan2_setup.py
python examples/demo_wan2.py
```

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

*Last Updated: September 7, 2025*
*Status: Web interface fully operational, ready for pose-video integration*
*Version: 1.0.0 - Production Ready*
*Progress: ~75-80% Complete*
