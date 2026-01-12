# ComfyUI Custom Node Patches

This directory contains patches for ComfyUI custom nodes that fix bugs or add features needed by Oelala.

## ComfyUI-MultiGPU.patch

Fixes for the ComfyUI-MultiGPU DisTorch2 nodes:

- Fix device allocation string parsing
- Fix VRAM tracking for multi-GPU setups
- Improve memory management for RTX 5060 Ti + RTX 3060 combo
- Fix VAE/CLIP loader device assignment

### Applying the patch

```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-MultiGPU
git apply /path/to/oelala/external/comfyui-patches/ComfyUI-MultiGPU.patch
```

### Reverting the patch

```bash
cd /path/to/ComfyUI/custom_nodes/ComfyUI-MultiGPU
git checkout .
```

## When to update patches

After updating ComfyUI-MultiGPU from upstream, check if the patch still applies cleanly:

```bash
git apply --check ComfyUI-MultiGPU.patch
```

If conflicts occur, the patch needs to be regenerated from the working local changes.
