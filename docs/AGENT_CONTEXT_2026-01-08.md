# Agent Context - Session 2026-01-08

**Agent**: Rick Sanchez (Claude Opus 4.5)
**Last Updated**: 2026-01-08 ~23:00 UTC

---

## 🎯 Current State

### Completed This Session
1. **PR #81 Rebased** - Fixed CHANGELOG.md merge conflict, force-pushed
   - Branch: `copilot/auto-upload-generated-content`
   - Ready to merge after CI passes

2. **LTX-2 Installation**
   - ✅ ComfyUI-LTXVideo nodes installed: `/home/flip/oelala/ComfyUI/custom_nodes/ComfyUI-LTXVideo/`
   - ✅ LTX-2 model downloaded: `models/checkpoints/ltx-2-19b-distilled-fp8.safetensors` (26GB)
   - ✅ Dependencies installed in gpu venv

3. **Disk Cleanup** - Freed 33GB
   - Removed SmolLM2 .git folder (22GB)
   - Removed pip cache (11GB)

4. **GitHub Issues Triage**
   - Closed 9 stale issues: #13, #14, #19, #21, #22, #52, #53, #66, #78
   - Assigned #16, #17 to Copilot coding agent

### In Progress
- **NVMe 500GB Installation** - 🔴 CRITICAL
  - Disk at 100% used (1.8GB free)
  - Server shutdown required

### Completed (Just Now)
- ✅ **Gemma 3 Text Encoder** - DOWNLOADED
  - Location: `models/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized/`
  - Size: 23GB (5 shards complete)

### Copilot PRs (Background)
| PR | Branch | Status | Issue |
|---|---|---|---|
| #81 | copilot/auto-upload-generated-content | Ready | #7 Auto-upload |
| #82 | copilot/add-progress-tracker-component | Draft | #16 Progress UI |
| #83 | copilot/add-websocket-progress-events | Draft | #17 WebSocket |

---

## 🔧 Technical Notes

### LTX-2 Requirements
- **VRAM**: 32GB+ recommended (we have 28GB - may need CPU offload)
- **Text Encoder**: Gemma 3 ONLY (not T5) - embedding dim mismatch
- **Model**: `ltx-2-19b-distilled-fp8.safetensors` - faster than full model

### LTX-2 vs T5 Issue
Tried using T5XXL with LTX-2, got:
```
RuntimeError: mat1 and mat2 shapes cannot be multiplied (256x4096 and 3840x4096)
```
LTX-2 expects Gemma 3's 3840 embedding dimension, T5 has 4096.

### Disk Space
- 🔴 **CRITICAL**: 1.8GB free (100% used)
- NVMe 500GB ready for installation

---

## 📋 Post-NVMe Checklist

1. **Mount NVMe** (suggest `/mnt/nvme` or `/data`)

2. **Move Large Models to NVMe**:
   ```bash
   # Example moves (adjust mount point)
   sudo mv /home/flip/oelala/ComfyUI/models/checkpoints/ltx-2-*.safetensors /mnt/nvme/models/
   sudo mv /home/flip/oelala/ComfyUI/models/text_encoders/gemma-3-12b-it* /mnt/nvme/models/
   # Symlink back
   ln -s /mnt/nvme/models/ltx-2-19b-distilled-fp8.safetensors /home/flip/oelala/ComfyUI/models/checkpoints/
   ```

3. **Restart ComfyUI**: `sudo systemctl restart comfyui`

4. **Test LTX-2 Workflow**
   - Use example from `ComfyUI-LTXVideo/example_workflows/`
   - Start with 512x320, 17 frames

5. **Merge PR #81**

---

## 🗂️ File Locations

| Item | Path |
|------|------|
| LTX-2 Model | `ComfyUI/models/checkpoints/ltx-2-19b-distilled-fp8.safetensors` |
| LTX-2 Nodes | `ComfyUI/custom_nodes/ComfyUI-LTXVideo/` |
| Gemma (to download) | `ComfyUI/models/text_encoders/gemma-3-12b-it-qat-q4_0-unquantized/` |
| T5XXL (existing) | `ComfyUI/models/clip/t5xxl_fp8_e4m3fn.safetensors` |
| GPU venv | `/home/flip/venvs/gpu` |
