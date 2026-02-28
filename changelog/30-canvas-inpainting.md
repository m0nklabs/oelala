### Added

- Canvas-based inpainting tool with HTML5 Canvas brush/eraser
- Brush size, opacity, and feathering controls
- Undo/redo with 30-state history (Ctrl+Z / Ctrl+Y)
- Keyboard shortcuts: B (brush), E (eraser), [ ] (size)
- Zoom controls for detailed mask editing
- Red overlay visualization for painted mask areas
- Touch support for tablet/mobile drawing
- Backend `/inpaint` endpoint using SDXL SetLatentNoiseMask approach
- Auth, credit deduction (8 credits), and WebSocket progress tracking
- Model selection from available SDXL checkpoints
- Advanced settings: steps, CFG, denoise strength, negative prompt
- Tool registered in dashboard navigation under Image Tools
