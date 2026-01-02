# GPU Device Mapping Reference

## Physical Hardware (nvidia-smi order)
| nvidia-smi Index | GPU Model | VRAM |
|-----------------|-----------|------|
| 0 | NVIDIA GeForce RTX 3060 | 12 GB |
| 1 | NVIDIA GeForce RTX 5060 Ti | 16 GB |

## PyTorch/ComfyUI Order (via CUDA_VISIBLE_DEVICES)
The systemd service (`~/.config/systemd/user/comfyui.service`) sets:
```
CUDA_VISIBLE_DEVICES=GPU-d7034f73-...,GPU-90e13f28-...
```

This remaps devices so PyTorch sees:
| cuda:X | GPU Model | VRAM | Role |
|--------|-----------|------|------|
| cuda:0 | RTX 5060 Ti | 16 GB | **Primary Compute** |
| cuda:1 | RTX 3060 | 12 GB | Donor/Secondary |

## Workflow Settings

### For DisTorch2 MultiGPU nodes:
```json
{
  "compute_device": "cuda:0",
  "donor_device": "cuda:1",
  "virtual_vram_gb": 16,
  "expert_mode_allocations": "cuda:0,12gb;cuda:1,10gb;cpu,*"
}
```

### Allocation Strategy (34GB GGUF model):
- **cuda:0 (5060 Ti)**: Up to 14GB allocation
- **cuda:1 (3060)**: Up to 10GB allocation  
- **CPU**: Remainder (~10-14GB)

### Safe Working Config:
```
expert_mode_allocations: "cuda:0,8gb;cuda:1,12gb;cpu,*"
```
Total: 20GB GPU + ~14GB CPU = 34GB model

## Verification Commands

Check PyTorch device order:
```bash
python3 -c "import torch; [print(f'cuda:{i} = {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

Check nvidia-smi order:
```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

## Troubleshooting

If device order is wrong:
1. Check `CUDA_VISIBLE_DEVICES` in systemd service
2. Reload: `systemctl --user daemon-reload && systemctl --user restart comfyui`
3. Verify with PyTorch command above

## Last Updated
2025-12-31 - GPU ordering permanently fixed
