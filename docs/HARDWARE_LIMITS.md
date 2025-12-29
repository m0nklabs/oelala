# Hardware Limits Testing - RTX 5060 Ti (16GB) + RTX 3060 (12GB)

## Current Working Configuration

| Parameter | Value | Status |
|-----------|-------|--------|
| Model | Q5_K_S (10GB) | ✅ Works |
| Resolution | 480×480 | ✅ Works |
| Frames | 41 | ✅ Works |
| Steps | 6 | ✅ Works |
| CFG | 5.0 | ✅ Works |
| Attention | SageAttention | ✅ Works |
| Offload | CPU (DisTorch2) | ✅ Works |

## Test Matrix

### Resolution Tests (41 frames, 6 steps)
| Resolution | Aspect | Pixels | Status | VRAM Peak | Time |
|------------|--------|--------|--------|-----------|------|
| 256×256 | 1:1 | 65K | ⏳ Pending | | |
| 320×320 | 1:1 | 102K | ⏳ Pending | | |
| 480×480 | 1:1 | 230K | ✅ Works | ~14GB | ~30s |
| 512×512 | 1:1 | 262K | ⏳ Pending | | |
| 576×576 | 1:1 | 331K | ⏳ Pending | | |
| 640×640 | 1:1 | 410K | ⏳ Pending | | |
| 720×720 | 1:1 | 518K | ⏳ Pending | | |
| 480×720 | 2:3 | 346K | ⏳ Pending | | |
| 720×480 | 3:2 | 346K | ⏳ Pending | | |
| 480×848 | 9:16 | 407K | ⏳ Pending | | |
| 848×480 | 16:9 | 407K | ⏳ Pending | | |

### Frame Count Tests (480×480, 6 steps)
| Frames | Duration @16fps | Status | VRAM Peak | Time |
|--------|-----------------|--------|-----------|------|
| 17 | 1.1s | ⏳ Pending | | |
| 25 | 1.6s | ⏳ Pending | | |
| 33 | 2.1s | ⏳ Pending | | |
| 41 | 2.6s | ✅ Works | ~14GB | ~30s |
| 49 | 3.1s | ⏳ Pending | | |
| 57 | 3.6s | ⏳ Pending | | |
| 65 | 4.1s | ⏳ Pending | | |
| 81 | 5.1s | ⏳ Pending | | |
| 97 | 6.1s | ⏳ Pending | | |
| 121 | 7.6s | ⏳ Pending | | |

### Step Count Tests (480×480, 41 frames)
| Steps | Quality Trade-off | Status | Time |
|-------|-------------------|--------|------|
| 4 | Fast, lower quality | ⏳ Pending | |
| 6 | Balanced | ✅ Works | ~30s |
| 8 | Better quality | ⏳ Pending | |
| 10 | High quality | ⏳ Pending | |
| 12 | Maximum quality | ⏳ Pending | |
| 20 | Overkill? | ⏳ Pending | |

### Model Quantization Tests (480×480, 41 frames, 6 steps)
| Model | Size | Status | Quality | Speed |
|-------|------|--------|---------|-------|
| Q4_K_M | 9GB | ✅ Works | Baseline | Fast |
| Q5_K_S | 10GB | ✅ Works | Better | ~Same |
| Q6_K | 12GB | ❌ OOM | N/A | N/A |
| Q8_0 | 15GB | ❌ Too large | N/A | N/A |

## VRAM Usage Breakdown (Estimated)

```
Component                    VRAM Usage
─────────────────────────────────────────
T5 Text Encoder (bf16)       ~4 GB
CLIP Vision                  ~1 GB
VAE                          ~1 GB
GGUF Model (Q5_K_S)          ~6 GB (with CPU offload)
Latents/Activations          ~4 GB (varies with resolution)
─────────────────────────────────────────
Total                        ~16 GB (at limit)
```

## Recommended Presets for Frontend

### Fast Preset
```json
{
  "resolution": {"width": 480, "height": 480},
  "frames": 25,
  "steps": 4,
  "estimated_time": "15-20s",
  "quality": "draft"
}
```

### Balanced Preset
```json
{
  "resolution": {"width": 480, "height": 480},
  "frames": 41,
  "steps": 6,
  "estimated_time": "25-35s",
  "quality": "good"
}
```

### Quality Preset
```json
{
  "resolution": {"width": 480, "height": 480},
  "frames": 41,
  "steps": 10,
  "estimated_time": "45-60s",
  "quality": "high"
}
```

### Long Video Preset
```json
{
  "resolution": {"width": 320, "height": 320},
  "frames": 81,
  "steps": 6,
  "estimated_time": "60-90s",
  "quality": "medium"
}
```

## Testing Script (TBD)

```bash
#!/bin/bash
# hardware_test.sh - Automated hardware limit testing

RESOLUTIONS=("256x256" "480x480" "512x512" "640x640" "720x720")
FRAME_COUNTS=(17 25 41 57 81)
STEP_COUNTS=(4 6 8 10 12)

for res in "${RESOLUTIONS[@]}"; do
  for frames in "${FRAME_COUNTS[@]}"; do
    echo "Testing: ${res} @ ${frames} frames..."
    # Submit workflow via API
    # Measure VRAM usage
    # Record time
    # Check for OOM
  done
done
```

## Notes

- **CPU Offload**: DisTorch2 offloads ~8-10GB to RAM, essential for Q5_K_S
- **SageAttention**: Reduces memory usage and speeds up attention
- **Multi-GPU**: RTX 3060 can assist but has limited VRAM
- **Batch Processing**: Not tested yet, likely needs lower settings

---

**Last Updated**: December 28, 2025
**Next**: Run systematic tests to fill in the matrix
