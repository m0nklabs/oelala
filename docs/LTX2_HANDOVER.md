# LTX-2 GGUF CPU Gemma - WERKENDE OPLOSSING 🎉

## Status: OPGELOST ✅

LTX-2 GGUF + Gemma-3 CPU encoding werkt nu volledig!

## Het Oorspronkelijke Probleem

LTX-2 GGUF + Gemma-3 encoder = OOM omdat:
1. **LTX-2 Q4_K_M**: ~12.7GB VRAM
2. **Gemma-3 FP4**: ~14GB VRAM (met activations)
3. **Totaal**: ~27GB, maar ComfyUI laadde beide op DEZELFDE GPU

## De Oplossing

CPU Gemma encoder in `ComfyUI/custom_nodes/ComfyUI-LTXVideo/cpu_gemma_encoder.py`:

### Kern fixes

1. **Folder path resolution** - `get_full_path()` geeft `None` voor folders, handmatige path constructie toegevoegd
2. **Tokenizer met left padding** - `padding_side="left"` + `model_max_length` + `pad_token`
3. **Max length padding** - `padding="max_length"` ipv `padding=True`
4. **Audio embeddings connector** - LTX-2 is een AV model, verwacht video+audio embeddings geconcateneerd
5. **Attention mask handling** - Juiste format voor conditioning dict

### Output shape

De encoder produceert nu:
- Input: text string
- Output: `[1, 256, 7680]` tensor (256 tokens, 3840 video + 3840 audio embeddings)

## Werkende Workflow

Bestand: `/home/flip/oelala/workflows/ltx2_cpu_gemma_api.json`

```
LTXVCPUGemmaEncode ─┬─→ LTXVConditioning ─→ BasicGuider ─→ SamplerCustomAdvanced
LTXVCPUGemmaNegativeEncode ─┘                     ↑
                                                  │
UnetLoaderGGUFAdvancedDisTorch2MultiGPU ──────────┘
VAELoader ─→ VAEDecode
```

## Performance

| Stap | Tijd |
|------|------|
| Gemma laden (eenmalig, cached) | ~9 sec |
| Text encoding | ~55 sec per prompt |
| Model laden | ~7 sec |
| Sampling (8 steps, 17 frames) | ~30 sec |
| VAE decode | ~1 sec |
| **Totaal** | ~2.5 min |

## VRAM Usage

| Component | Device | VRAM |
|-----------|--------|------|
| Gemma-3 12B | CPU | 0 GB VRAM, ~24GB RAM |
| LTX-2 UNET | cuda:1 | ~11 GB |
| LTX-2 UNET | cuda:0 | ~1 GB |
| VAE | cuda:1 | ~2.4 GB |

## Bestanden

- `/home/flip/oelala/ComfyUI/custom_nodes/ComfyUI-LTXVideo/cpu_gemma_encoder.py` - CPU Gemma encoder
- `/home/flip/oelala/workflows/ltx2_cpu_gemma_api.json` - Werkende workflow

## GPU Setup

| GPU | VRAM | CUDA |
|-----|------|------|
| RTX 5060 Ti | 16GB | cuda:1 |
| RTX 3060 | 12GB | cuda:0 |

DisTorch2 allocation: `cuda:1,11gb;cuda:0,14gb;cpu,2gb`

## Test Commando

```bash
cd /home/flip/oelala && python3 -c "
import requests, json
with open('workflows/ltx2_cpu_gemma_api.json') as f:
    workflow = json.load(f)
resp = requests.post('http://localhost:8188/prompt', json={'prompt': workflow})
print(resp.json())
"
```
