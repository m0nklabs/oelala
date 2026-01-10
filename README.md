# 🎬 OELALA - AI Media Creation Platform

A modern web platform for AI-powered image and video generation. Create, customize, and publish your media content with state-of-the-art AI models.

## ✨ Features

### 🎨 Media Creation
- **Text-to-Image**: Generate images from text prompts using Flux, SDXL, and more
- **Image-to-Video**: Transform images into dynamic video clips (WAN 2.2, LTX-2)
- **Text-to-Video**: Create videos directly from text descriptions
- **Video with Audio**: Generate synchronized audio tracks (LTX-2 Audio)
- **Frame Interpolation**: Smooth your videos with AI-powered frame generation
- **Video Upscaling**: Enhance resolution with RealESRGAN

### 💰 Credits System
- Pay-as-you-go model - only pay for what you generate
- Welcome bonus for new users
- Credit packages from €5 to €500
- Transparent pricing per generation type

### 🖼️ Gallery & Publishing
- Publish your creations to the public gallery
- Browse and like other creators' work
- Tag and categorize your content
- View counts and engagement stats

### 👤 User Features
- Google OAuth authentication
- Personal media library (MyMedia)
- Generation history with workflow export
- User storage with quota management

## 🚀 Quick Start

### Web Access
Visit [oelala.xyz](https://oelala.xyz) to start creating!

### Self-Hosting

```bash
# Clone the repository
git clone https://github.com/m0nklabs/oelala.git
cd oelala

# Install dependencies
pip install -r requirements.txt
cd src/frontend && npm install

# Start services
sudo systemctl start comfyui
sudo systemctl start oelala-backend
npm run dev  # Frontend development server
```

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Frontend     │────▶│    Backend      │────▶│    ComfyUI      │
│   (React/Vite)  │     │   (FastAPI)     │     │  (AI Workflows) │
│   Port: 5174    │     │   Port: 7998    │     │   Port: 8188    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌─────────────────┐
                        │  oelala-storage │
                        │   (Go/gRPC)     │
                        │   Port: 7990    │
                        └─────────────────┘
```

## 🎯 Supported Models

### Image Generation
| Model | Description | VRAM |
|-------|-------------|------|
| Flux Dev | High-quality general purpose | 16GB |
| SDXL | Stable Diffusion XL | 8GB |
| Pony Diffusion | Stylized art generation | 8GB |

### Video Generation
| Model | Description | VRAM |
|-------|-------------|------|
| WAN 2.2 14B | High-quality video (I2V, T2V) | 24GB+ |
| LTX-2 | Fast video with audio support | 16GB |

### Audio Generation
| Model | Description |
|-------|-------------|
| LTX-2 Audio VAE | Synchronized video audio |
| MMAudio | Sound effects and music |

## 🔧 Configuration

### Environment Variables

```bash
# Backend (.env)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key
STRIPE_SECRET_KEY=sk_...
CREDITS_ENABLED=1  # Set to 0 to disable credits

# ComfyUI
COMFYUI_URL=http://localhost:8188
```

### Multi-GPU Setup

OELALA supports multi-GPU configurations with DisTorch2:

```
# Example: RTX 5060 Ti (16GB) + RTX 3060 (12GB)
cuda:0,12gb;cuda:1,16gb
```

## 🔞 Age Restriction

**Users must be 18+ to access mature content.**

- NSFW tools and components require age verification
- Adult content is clearly labeled and filtered by default
- Gallery has separate SFW/NSFW filters
- Guests can only view SFW content

## 📊 Credit Costs

| Generation Type | Credits |
|----------------|---------|
| Image (standard) | 1-2 |
| Image (HD) | 2-3 |
| Video (3 sec) | 5-8 |
| Video (5 sec) | 8-12 |
| Video HD | 10-15 |
| Audio sync | 3-5 |

## 🖥️ System Requirements

### Minimum
- **GPU**: NVIDIA RTX 3060 12GB
- **RAM**: 32GB
- **Storage**: 100GB SSD

### Recommended
- **GPU**: NVIDIA RTX 4090 24GB (or multi-GPU setup)
- **RAM**: 64GB
- **Storage**: 500GB NVMe SSD

## 🛠️ Development

```bash
# Run tests
pytest tests/

# Lint code
ruff check src/

# Build frontend
cd src/frontend && npm run build
```

## 📝 License

- **Platform code**: MIT License
- **AI Models**: Various (check individual model licenses)
- **Flux models**: FLUX.1-dev license
- **WAN models**: Apache 2.0

## 🤝 Contributing

Contributions welcome! Please read our contributing guidelines and submit PRs.

1. Fork the repository
2. Create a feature branch
3. Add changelog fragment in `changelog/`
4. Submit a pull request

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/m0nklabs/oelala/issues)
- **Discussions**: [GitHub Discussions](https://github.com/m0nklabs/oelala/discussions)

---

Built with ❤️ by m0nk111 | Powered by ComfyUI
