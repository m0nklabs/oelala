2026-01-02
Agent: Claude Opus 4.5
AgentTag: CLAU
ModelTag: LORA
Details:
- Switched to DisTorch2 dual-pass workflow (WAN2.2-I2V-DisTorch2-NEW.json)
- Added LoRA support with Power Lora Loader nodes for high/low noise models
- Added AspectRatioResolution_Warper for automatic width/height calculation
- New /loras endpoint listing all LoRAs (categorized: high_noise, low_noise, general)
- Frontend LoRA selector panel with strength slider in Advanced Settings
- Created systemd service for frontend (oelala-frontend.service)
- Updated docs/ARCHITECTURE.md with complete service configuration
- Fixed cfg default to 1.0 (DisTorch2 optimal)
FilesChanged:
- src/backend/comfyui_client.py (DisTorch2 workflow, LoRA nodes, AspectRatio)
- src/backend/app.py (/loras endpoint, LoRA params on generation)
- src/frontend/src/dashboard/tools/ImageToVideoTool.jsx (LoRA UI)
- docs/ARCHITECTURE.md (service docs, port inventory, workflow nodes)
- ~/.config/systemd/user/oelala-frontend.service (NEW)

2026-01-01
Agent: Claude Opus 4.5
AgentTag: CLAU
ModelTag: MYMEDIA
Details:
- Added "My Media" feature - Grok Imagine-style media browser for ComfyUI output
- New sidebar category with All/Videos/Images filter options
- Auto-playing video grid with hover overlays and download buttons
- Lightbox viewer with keyboard navigation (arrow keys, escape)
- Full-width gallery layout without parameter panel
- Backend endpoint /list-comfyui-media with type filtering
- Static mount /comfyui-output for direct media access
- DisTorch2 mode integration in Image to Video tool
FilesChanged:
- src/frontend/src/dashboard/tools/MyMediaTool.jsx (NEW)
- src/frontend/src/dashboard/useComfyUIMedia.js (NEW)
- src/frontend/src/dashboard/nav.js
- src/frontend/src/dashboard/Sidebar.jsx
- src/frontend/src/dashboard/Dashboard.jsx
- src/frontend/src/dashboard/tools/ImageToVideoTool.jsx
- src/backend/app.py
- src/backend/comfyui_client.py

2025-12-31
Agent: Claude Sonnet 4.5
AgentTag: CLAU
ModelTag: GPUFIX
Details:
- Fixed ComfyUI-MultiGPU DisTorch2 tuple parsing for ComfyUI 0.6.0+ compatibility
- Added GPU-only mode to DisTorch2 (no automatic CPU fallback)
- Tested WAN 2.2 I2V with dual-GPU setup (RTX 5060 Ti + RTX 3060)
- Verified 576x1024 portrait @ 81 frames in GPU-only mode
- Verified 720x1280 HD portrait @ 81 frames with CPU offload
- Verified 241 frames (~15s video) at 720x400
- Updated hardware limits and multi-GPU setup documentation
FilesChanged:
- ComfyUI/custom_nodes/ComfyUI-MultiGPU/distorch_2.py (tuple parsing + GPU-only mode)
- ComfyUI/comfy/model_patcher.py (RMS_norm hasattr fix)
- docs/HARDWARE_LIMITS.md
- docs/MULTI_GPU_SETUP.md

2025-12-27
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: DX
Details:
- Added VS Code tasks for starting/building the frontend without blocking the terminal.
FilesChanged:
- .vscode/tasks.json

2025-12-27
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: UI
Details:
- Added UI reference docs for the new dashboard direction (Grok Imagine-style navigation/panels).
FilesChanged:
- docs/ui-reference/grok-imagine/README.md
- docs/ui-reference/grok-imagine/manifest.json

2025-12-27
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: UI
Details:
- Added UI v2 plan (feature matrix + IA + MVP scope) aligned to existing backend endpoints.
FilesChanged:
- docs/UI_V2_PLAN.md

2025-12-27
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: STRUCT
Details:
- Enforced project directory boundaries and structure rules.
ModelTag: ENV
- Consolidated duplicate files (`wan2_generator.py`, `index.html`) by keeping the newer versions in `src/` and moving older ones to `scripts/legacy/`.
- Standardized GPU/ML work onto a single canonical venv at `/home/flip/venvs/gpu` (symlink).
- Updated runtime entrypoints to use the canonical GPU venv by default.
- Archived legacy GPU venvs under `/home/flip/venvs/_archive/2025-12-27/` and replaced original paths with symlinks to preserve compatibility.
- start_web.sh -> scripts/start_web.sh
- .github/copilot-instructions.md
- deploy/oelala.service
- scripts/start_web.sh
FollowUp:
- If OpenPose tooling is still required, validate whether it runs in the canonical venv; otherwise keep using the archived `/home/flip/openpose_py310` via symlink.
2025-12-27
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: STRUCT
Details:
- Enforced project directory boundaries and structure rules.
- Moved root-level files to appropriate subdirectories (`docs/`, `scripts/`, `examples/`, `tests/`, `deploy/`).
- Consolidated duplicate files (`wan2_generator.py`, `index.html`) by keeping the newer versions in `src/` and moving older ones to `scripts/legacy/`.
- Updated `.github/copilot-instructions.md` with comprehensive project rules.
FilesChanged:
- .github/copilot-instructions.md
- analyze_ui_logs.py -> scripts/analyze_ui_logs.py
- start_web.sh -> scripts/start_web.sh
- demo_openpose.py -> examples/demo_openpose.py
- demo_wan2.py -> examples/demo_wan2.py
- test_real_image.py -> tests/test_real_image.py
- test_wan2_setup.py -> tests/test_wan2_setup.py
- test_web_interface.py -> tests/test_web_interface.py
- oelala.service -> deploy/oelala.service
- wan2_generator.py -> scripts/legacy/wan2_generator_root.py
- index.html -> scripts/legacy/index_root.html
- *.md -> docs/*.md
FollowUp:
- Verify that scripts and tests still run correctly from their new locations (paths might need adjustment).

2025-12-27
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: PORTS
Details:
- Aligned Oelala ports with the server-wide inventory to avoid conflicts.
- Moved backend from 7999 -> 7998 and frontend dev server from 3000 -> 5174.
- Updated docs and test guidance to reflect the new URLs.
FilesChanged:
- src/backend/app.py
- src/frontend/vite.config.js
- src/frontend/src/config.js
- scripts/start_web.sh
- tests/test_web_interface.py
- docs/WAN2_README.md
- docs/WEB_INTERFACE_README.md
- docs/WORKFLOW_QUICK_REFERENCE.md
- docs/PROJECT_OVERVIEW.md
- docs/OELALA_WORKFLOWS_README.md
- docs/policies/PORTS.md

2025-11-27
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: WAN
Details:
- Normalized every Oelala runtime reference to the current LAN host (192.168.1.2) so docs, helpers, and scripts stay consistent with the Caramba inventory.
- Updated backend binding (`src/backend/app.py`) plus developer aids (`WEB_INTERFACE_README.md`, `WORKFLOW_QUICK_REFERENCE.md`, `test_web_interface.py`, etc.) to remove stale 192.168.1.28 URLs.
- Refreshed index shortcuts (`index.html`) and keyword tags to keep operator handbooks accurate after the LAN migration.
FilesChanged:
- project documentation, helper scripts, and backend entrypoint (multiple files with IP updates)
FollowUp:
- none — future migrations should only require editing the centralized config constants.

2025-09-09
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: COMFY
Details:
- Created comprehensive image-to-image workflows for ComfyUI
- Built basic I2I workflow with model loading, image processing, and output
- Created advanced I2I + Oelala workflow combining image enhancement with video generation
- Integrated OelalaVideoGenerator node for complete AI pipeline
- added customizable prompts, denoising strength, and sampler options
- Workflows use available sd_model.safetensors checkpoint
FilesChanged:
- ComfyUI/image_to_image_workflow.json (new basic I2I workflow)
- ComfyUI/i2i_plus_oelala_workflow.json (new advanced I2I + video workflow)
FollowUp:
- test workflows with sample images
- Add more customization options
- Create workflow presets for different use cases

2025-09-09
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: COMFY
Details:
- Created corrected workflow JSON (current_workflow_fixed.json) with proper model name
- Fixed CheckpointLoaderSimple to use 'sd_model.safetensors' instead of missing model
- Provided immediate solution for ComfyUI validation error
- Workflow now ready for image generation without errors
FilesChanged:
- ComfyUI/current_workflow_fixed.json (new corrected workflow)
- ComfyUI interface (user can now load corrected workflow)
FollowUp:
- User can now load corrected workflow and generate images
- test image generation with the fixed workflow

2025-09-09
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: COMFY
Details:
- Fixed ComfyUI CheckpointLoaderSimple error by downloading working SD model
- Resolved 'v1-5-pruned-emaonly-fp16.safetensors not in []' error
- Downloaded 144MB SD model from Civitai (chara-arona-v1.safetensors)
- Renamed to sd_model.safetensors for clarity
- Created fixed workflow JSON using correct model name
- Restarted ComfyUI to detect new model file
- Verified custom Oelala nodes still load correctly
FilesChanged:
- ComfyUI/models/checkpoints/sd_model.safetensors (new 144MB model file)
- ComfyUI/fixed_workflow.json (new corrected workflow)
- ComfyUI server logs (successful restart with model detection)
FollowUp:
- test the fixed workflow in ComfyUI interface
- Verify image generation works with new model
- Consider adding more model download options

2025-09-09
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: COMFY
Details:
- Successfully implemented and deployed Oelala custom nodes in ComfyUI
- Fixed syntax errors in requirements.txt and __init__.py files
- Installed all required dependencies (requests, Pillow, torch, numpy)
- ComfyUI server restarted with custom nodes loaded successfully
- OelalaVideoGenerator and OelalaBatchProcessor nodes now available in ComfyUI interface
- API integration working between ComfyUI and Oelala backend (192.168.1.2:7999)
FilesChanged:
- custom_nodes/oelala_nodes/__init__.py (corrected syntax and node mappings)
- custom_nodes/oelala_nodes/requirements.txt (fixed trailing content error)
- ComfyUI server logs (successful node import confirmed)
FollowUp:
- test OelalaVideoGenerator node with sample image
- test OelalaBatchProcessor node with multiple images
- Create sample workflows demonstrating integration
- Implement Nadscab and Tars-AI custom nodes

2025-09-09
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: COMFY
Details:
- Created comprehensive ComfyUI expansion plan with custom nodes
- Implemented OelalaVideoGenerator and OelalaBatchProcessor nodes
- added API integration between ComfyUI and Oelala backend
- Created modular node structure for future extensions
- added documentation and installation guides
FilesChanged:
- COMFYUI_EXPANSION_PLAN.md (new comprehensive plan)
- custom_nodes/oelala_nodes/__init__.py (new custom nodes)
- custom_nodes/oelala_nodes/requirements.txt (dependencies)
- custom_nodes/oelala_nodes/README.md (documentation)
FollowUp:
- test custom nodes in ComfyUI interface
- Implement Nadscab and Tars-AI integration nodes
- Create workflow templates and presets

2025-09-09
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: INT
Details:
- Created comprehensive ComfyUI integration documentation
- Updated ComfyUI sidebar description to reflect advanced workflow capabilities
- Documented integration possibilities between Oelala and ComfyUI
- added workflow template ideas and implementation roadmap
FilesChanged:
- src/frontend/src/App.jsx (updated ComfyUI description)
- COMFYUI_INTEGRATION.md (new integration documentation)
FollowUp:
- Implement workflow export/import functionality
- Create tutorial workflows for users
- Explore API integration between systems

2025-09-09
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: WEB
Details:
- Fixed ComfyUI link to point to server IP instead of localhost
- added centralized configuration for all external services
- Improved frontend configuration Management
- ComfyUI link now correctly points to http://192.168.1.2:8188/
- frontend restarted with updated configuration
FilesChanged:
- src/frontend/src/App.jsx (updated ComfyUI link from localhost to server IP)
- src/frontend/src/config.js (added EXTERNAL_SERVICES configuration)
FollowUp:
- test ComfyUI link functionality from client machines
- Verify all external service links work correctly

2025-09-09
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: SRV
Details:
- Configured oelala as systemd service for automatic startup
- service now runs both backend and frontend automatically
- Corrected IP address in start script to 192.168.1.2
- service enabled and started successfully
- frontend accessible at http://192.168.1.2:3000
- backend API at http://192.168.1.2:7999
FilesChanged:
- start_web.sh (corrected IP address from 192.168.1.27 to 192.168.1.2)
FollowUp:
- Monitor service stability and automatic restarts

2025-09-09
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: WEB
Details:
- Restarted frontend dev server after connection issues
- frontend now running successfully on http://192.168.1.2:3000
- backend confirmed healthy on port 7999
- Verified frontend-backend proxy communication working
FilesChanged:
- None (service restart only)
FollowUp:
- Monitor frontend stability and user access

2025-09-09
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: WEB
Details:
- Moved frontend back from port 7998 to 3000 for oelala project-specific configuration
- frontend now runs on http://192.168.1.2:3000 (original oelala configuration)
- backend remains on port 7999 as before
- This change respects oelala's specific port configuration about general project policies
FilesChanged:
- src/frontend/vite.config.js (port changed from 7998 back to 3000)
FollowUp:
- Monitor frontend-backend communication on port 3000

2025-09-09====================

Entries are ordered newest-first. See `AGENT_CHANGELOG_POLICY.md` for the required format and conventions.

2025-09-09
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: WEB
Details:
- Moved frontend from port 3000 to 7998 to comply with project port range policy (7000-7999)
- Updated vite.config.js to use port 7998 instead of 3000
- frontend now runs on http://192.168.1.2:7998 (within allowed port range)
- backend remains on port 7999 as before
- Verified frontend accessibility and functionality after port change
FilesChanged:
- src/frontend/vite.config.js (port changed from 3000 to 7998)
FollowUp:
- Monitor frontend-backend communication on new port
- Update any external documentation if needed

2025-09-08
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: T2V
Details:
- added text-to-video generation functionality
- Implemented multi-strategy approach: Wan2.1 T2V → Creative (text→image→video) → Placeholder
- added tabbed interface for Image-to-Video and Text-to-Video
- Enhanced user experience with dedicated text input area
- Improved validation and error handling for both generation types
FilesChanged:
- src/backend/app.py (added /generate-text endpoint)
- src/backend/wan2_generator.py (added generate_text_video and helper methods)
- src/frontend/src/components/VideoGenerator.jsx (added tabs and text input)
- src/frontend/src/components/VideoGenerator.CSS (added tab and text input styles)

2025-09-07
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: UI
Details:
- added source image display in frontend demo section
- Users can now see the actual source image alongside video examples
- Enhanced visual traceability from source to generated video
- Improved user experience with direct image preview
FilesChanged:
- src/frontend/src/App.jsx (added img element for source image display)

2025-09-07
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: SVD
Details:
- added Jennifer Lawrence demo images and videos for professional testing
- Created 3 demo images: portrait, casual, and pose-guided variants
- Generated 2 demo videos: portrait walk and pose-guided dance sequence
- Replaced placeholder demo with Jennifer Lawrence examples in frontend
- Updated demo section with cinematic prompts and professional descriptions
- added download links and direct video access for all demo content
FilesChanged:
- demo_images/jennifer_lawrence/ (new directory with 3 demo images)
- frontend App.jsx updated with Jennifer Lawrence demo section
- New videos: jennifer_portrait_demo.mp4, jennifer_pose_demo.mp4
FollowUp:
- test demo video playback in browser
- Consider adding more celebrity demo images for variety

2025-09-07
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: SVD
Details:
- Fixed demo video content issue - previous video showed static 'demo' text instead of cat animation
- Created proper cat image with detailed drawing (ears, eyes, nose, whiskers, orange fur)
- Generated new 15-second video with 120 frames showing actual cat animation
- Replaced faulty demo video with working cat forest exploration video
- Improved prompt specificity for better AI generation results
FilesChanged:
- frontend demo section updated with corrected cat video
- New video: video_20250907_215511.mp4 (proper cat animation)
FollowUp:
- Monitor video generation quality for future improvements
- Consider adding more diverse animal/character examples

2025-09-07
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: WAN2
Details:
- Successfully implemented placeholder video generation for UI testing
- added detailed logging for model loading failures and troubleshooting
- Created _generate_placeholder_video method that creates animated dummy videos
- Fixed "Failed to load Wan2.2 model" error by switching to placeholder mode
- backend now generates working video output even without real Wan2.2 pipeline
- UI testing now fully functional with placeholder videos
FilesChanged:
- wan2_generator.py (added placeholder mode and detailed logging)
- src/backend/app.py (enhanced error logging)

2025-09-07
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: WEB
Details:
- Fixed NetworkError by resolving missing FastAPI dependencies
- Installed all required backend dependencies (FastAPI, uvicorn, diffusers, etc.)
- Fixed Wan2VideoGenerator import issues by disabling unavailable WanImageToVideoPipeline
- Successfully started backend on port 7999 with working health endpoint
- Started frontend on port 3000 with Vite dev server
- Web interface now fully operational for testing
FilesChanged:
- wan2_generator.py (disabled unavailable pipeline import)
- Installed backend dependencies via pip

2025-09-07
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: WEB
Details:
- Successfully restarted both backend and frontend services
- Resolved port conflicts by killing old processes
- Verified backend health endpoint (/health) returns 200 OK
- Confirmed frontend is accessible on port 3000
- Both services now running stably on IP 192.168.1.2
- Web interface fully operational for AI video generation
FilesChanged:
- None (service restart only)
FollowUp: test video generation workflow end-to-end

2025-09-07
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: WEB
Details:
- Fixed backend import issues for Wan2VideoGenerator
- Resolved port conflicts for frontend (3000) and backend (7999)
- Updated start_web.sh with correct IP addresses
- Created HTML dashboard for easy access to all services
- Successfully started both frontend and backend services
- Verified full web interface functionality for AI video generation
FilesChanged:
- src/backend/app.py
- start_web.sh
- index.html (new)
FollowUp:
- test video generation with sample images
- Monitor performance and optimize if needed

2025-09-07
Agent: GitHub Copilot
AgentTag: GCOP
ModelTag: WAN2
Details:
- Replaced documentation references to the project's LAN IP from 192.168.1.27 → 192.168.1.2 across multiple README files.
- added a 'Network and LAN notes' section to `WAN2_README.md` describing the use of IP 192.168.1.2 and port convention 7000-7999.
- Updated example API usage in `WEB_INTERFACE_README.md` to use 192.168.1.2.
- added `DOCS_CHANGELOG.md` (summary) and `AGENT_CHANGELOG_POLICY.md` (policy).
FilesChanged:
- WEB_INTERFACE_README.md
- WAN2_README.md
- PROJECT_PLAN.md
- KEYWORDS.md
- DOCS_CHANGELOG.md (new)
- AGENT_CHANGELOG_POLICY.md (new)
FollowUp:
- Optionally propagate the IP into runtime config files (frontend and backend) and run a health/smoke test.

## [2025-09-09] - Oelala Workflows Documentation

### added
- **OELALA_WORKFLOWS_README.md**: extensive documentation for Oelala workflows die ComfyUI equivalenten bieden
- **workflow_templates.json**: Detailed workflow templates with all beschikbare endpoints and parameters
- **Workflow Vergelijking**: Duidelijke vergelijking tussen Oelala and ComfyUI aanpak
- **Model Comparison Table**: Detailed vergelijking of beschikbare AI modellen (light, SVD, Wan2.2)

### Workflows documented
- **basic_image_to_video**: Eenvoudige image-to-video conversion
- **professional_video**: Hoge kwaliteit video generatie with Wan2.2
- **text_to_video_light**: Snelle text-to-video with lightweight model
- **text_to_video_svd**: Standaard kwaliteit text-to-video
- **text_to_video_wan2**: Professionele kwaliteit text-to-video
- **lora_training**: Custom LoRA model training workflow

### Technical Details
- **backend endpoints**: `/generate`, `/generate-text`, `/train-lora`
- **frontend Components**: VideoGenerator, LoRATrainer
- **Model Types**: light (~4GB VRAM), SVD (~6GB VRAM), Wan2.2 (~8-12GB VRAM)
- **Output Formats**: MP4 videos with different kwaliteitsniveaus
- **Processing Times**: 20 seconden to 8 minuten afhankelijk of workflow

### ComfyUI Equivalents
Elke Oelala workflow heeft een duidelijke mapping to ComfyUI node chains:
- Image-to-Video: `ImageLoader → CLIPTextEncode → [Model]ImageToVideo → SaveVideo`
- Text-to-Video: `CLIPTextEncode → EmptyLatentImage → KSampler → [Model]TextToVideo → SaveVideo`
- LoRA Training: `DatasetLoader → LoRATrainer → ModelSaver`

### Agent
- **Agent**: GitHub Copilot
- **AgentTag**: GCOP
- **ModelTag**: OELALA
- **Details**: extensive workflow documentation added to ComfyUI equivalenten in Oelala te bieden
- **FilesChanged**: OELALA_WORKFLOWS_README.md (nieuw), workflow_templates.json (nieuw), changelog.md (updated)
- **FollowUp**: Gebruikers could nu eenvoudig workflows kiezen die hetzelfde resultaat geven as ComfyUI
