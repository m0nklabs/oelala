# Wan2.2 Image-to-Video Integration
## Oelala Project - AI Video Generation Pipeline

This directory contains the Wan2.2 Image-to-Video generation integration for the Oelala AI pipeline.

## Overview

Wan2.2 is a state-of-the-art image-to-video generation model that can create smooth, realistic videos from input images. This integration allows you to:

- generate videos from images with optional text prompts
- Control video length and quality parameters
- use GPU acceleration for fast generation
- Integrate with pose estimation for avatar generation

## Files

- `wan2_generator.py` - Main Wan2.2 generator class
- `demo_wan2.py` - Demo script for testing
- `test_wan2_setup.py` - Setup validation script

## Requirements

- Python 3.10
- PyTorch 2.5.1+ (with CUDA support)
- CUDA-compatible GPU (recommended 16GB+ VRAM)
- dependencies: diffusers, transformers, accelerate, safetensors, opencv-Python, PIL, numpy

## Installation

The dependencies are already installed in the `/home/flip/openpose_py310` virtual environment.

## Usage

### Basic Usage

```Python
from wan2_generator import Wan2VideoGenerator

# Initialize generator
generator = Wan2VideoGenerator()

# Load model (first time will download ~14GB)
generator.load_model()

# generate video from image
generator.generate_video_from_image(
    image_path="person.jpg",
    prompt="A person dancing gracefully",
    output_path="output.mp4",
    num_frames=16
)
```

### Command Line Demo

```bash
# Activate virtual environment
cd /home/flip/openpose_py310
source bin/activate

# Run demo
cd /home/flip/oelala
Python demo_wan2.py
```

### Advanced Parameters

```Python
generator.generate_video_from_image(
    image_path="input.jpg",
    prompt="Dynamic movement and animation",
    output_path="result.mp4",
    num_frames=32,  # More frames = longer video
    # num_inference_steps=20,  # Quality vs speed tradeoff
    # guidance_scale=6.0      # How closely to follow prompt
)
```

## Model Details

- **Model**: Wan-AI/Wan2.2-I2V-A14B
- **Type**: Image-to-Video (I2V)
- **Parameters**: 1.3B
- **Input**: RGB images (any resolution, will be resized)
- **Output**: MP4 videos (default 832x480, 16 frames)
- **Memory**: ~16GB VRAM recommended

## Integration with OpenPose

This Wan2.2 integration is designed to work with the OpenPose pose estimation system:

1. **Pose Estimation**: use OpenPose to detect keypoints from input images
2. **Video Generation**: generate videos with natural movement
3. **Avatar Creation**: Combine pose data with video generation for realistic avatars

## Performance Tips

- **GPU Memory**: Ensure at least 16GB VRAM for best results
- **Inference Steps**: Lower values (20) for faster generation, higher (50+) for better quality
- **Frame Count**: 16 frames = ~2 seconds at 8 FPS
- **Resolution**: Default 832x480, can be adjusted but affects memory usage

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `num_frames` or `num_inference_steps`
   - use smaller input images
   - Restart Python session to clear GPU memory

2. **Model Download Issues**
   - Ensure stable internet connection
   - Model will be cached after first download
   - Total download size: ~14GB

3. **Import Errors**
   - Make sure you're using the correct virtual environment
   - Run: `cd /home/flip/openpose_py310 && source bin/activate`

### Validation

Run the setup test to verify everything is working:

```bash
cd /home/flip/openpose_py310
source bin/activate
cd /home/flip/oelala
Python test_wan2_setup.py
```

## Next Steps

1. **Pose Integration**: Combine with OpenPose for pose-guided video generation
2. **LoRA Fine-tuning**: Train custom models for specific styles
3. **Batch Processing**: generate multiple videos in parallel
4. **Web Interface**: Integrate with FastAPI backend for web access

## Network and LAN notes

- Project IP: This Oelala instance uses the LAN IP 192.168.1.2 for frontend and backend services. Update any local shortcuts or scripts to use this IP.
- Ports: services for this project should use ports in the range 7000-7999 (default backend port: 7998). This keeps ports consistent across related projects on the same host.

Example addresses:

- frontend: http://192.168.1.2:5174
- backend API: http://192.168.1.2:7998

If you need to change the IP, update the frontend `src/config.js` and the backend start scripts accordingly.

## License

This integration follows the same license as the Oelala project. Wan2.2 model usage follows HuggingFace model licenses.
