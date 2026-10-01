# Oelala UI v2 Plan (Grok-Imagine inspired)

> **Retired 2026-10-01:** Wan 2.2 was retired from the product on 2026-10-01; references to it below are historical. See `docs/LEGACY.md`.

## Status: IMPLEMENTED ✅

The dashboard UI has been implemented with the following features:

### Completed Features

#### Dashboard Layout ✅
- Left sidebar navigation with grouped tools
- Main content area with controls + output panels
- Collapsible sidebar sections
- Dark theme with CSS variables

#### Video Tools
- ✅ **Image to Video** - Full ComfyUI integration with DisTorch2
  - Image upload (drag & drop, URL, from gallery)
  - Positive/negative prompts with persistence
  - Model pair selection (high/low noise GGUF)
  - LoRA selection with strength control
  - Resolution, duration, FPS controls
  - Preset system for workflow configurations
- ⏳ Text to Video - Planned
- ⏳ Video to Video - Planned

#### Image Tools
- ⏳ Text to Image - Planned
- ⏳ Image to Image - Planned
- ⏳ Face Swap - Planned
- ⏳ Upscaler - Planned

#### My Media ✅
- ✅ **Gallery View** - Grid layout with thumbnails
- ✅ **Filters** - All, Images, Videos, Favorites
- ✅ **Prompts Section** - NEW! Browse generation history
  - Prompt bubble (💬) on thumbnails
  - Popup modal with full generation details
  - Copy prompts to clipboard
  - View LoRAs, sampler, model, resolution
- ✅ **Favorites** - Star items, filter by favorites
- ✅ **Multi-select** - Shift/Ctrl+click for bulk operations
- ✅ **Sorting** - By date, name, size

#### Training
- ⏳ Train LoRA - Placeholder ready

---

## Current Backend Capabilities

### Generation Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| \`/wan22/image-to-video\` — **retired 2026-10-01, endpoint removed** | POST | ComfyUI-based I2V with DisTorch2 |
| \`/health\` | GET | Backend + ComfyUI status |

### Media Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| \`/list-comfyui-media\` | GET | List media with metadata |
| \`/comfyui-output/{file}\` | GET | Serve generated files |
| \`/delete-comfyui-media\` | DELETE | Bulk delete files |
| \`/extract-metadata\` | POST | Extract prompt from image |

### Model Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| \`/loras\` | GET | List LoRAs by category |
| \`/unet-models\` | GET | List GGUF model pairs |
| \`/api/presets\` | GET | List workflow presets |

---

## Frontend Architecture

### File Structure
\`\`\`
src/frontend/src/
├── dashboard/
│   ├── Dashboard.jsx       # Main layout with sidebar
│   ├── Dashboard.css       # Dashboard styles
│   ├── OutputPanel.jsx     # Right panel for output
│   ├── nav.js              # Navigation configuration
│   └── tools/
│       ├── ImageToVideoTool.jsx  # I2V generation
│       ├── TextToVideoTool.jsx   # T2V (planned)
│       └── MyMediaTool.jsx       # Gallery + prompts
└── components/
    ├── PresetSelector.jsx  # Preset dropdown
    └── VideoGenerator.jsx  # Legacy component
\`\`\`

### Navigation Structure (nav.js)

\`\`\`javascript
{
  id: 'video-tools',
  label: 'Video Tools',
  items: [
    { id: 'text-to-video', label: 'Text to Video' },
    { id: 'image-to-video', label: 'Image to Video' },
    { id: 'video-to-video', label: 'Video to Video', status: 'soon' }
  ]
},
{
  id: 'my-media',
  label: 'My Media',
  items: [
    { id: 'my-media-all', label: 'All' },
    { id: 'my-media-images', label: 'Images' },
    { id: 'my-media-videos', label: 'Videos' },
    { id: 'my-media-favorites', label: 'Favorites' },
    { id: 'my-media-prompts', label: 'Prompts', status: 'new' }
  ]
}
\`\`\`

---

## Next Steps (v3)

### Planned Features
1. **Text to Video** - Direct text prompt to video generation
2. **Video to Video** - Style transfer / video editing
3. **Text to Image** - ComfyUI T2I workflows
4. **LoRA Training** - Fine-tune models on custom images
5. **Batch Processing** - Queue multiple generations

### Technical Improvements
- TypeScript migration
- React Query for data fetching
- Zustand for state management
- Improved error handling
- WebSocket progress updates

---

*Last Updated: January 3, 2026*
