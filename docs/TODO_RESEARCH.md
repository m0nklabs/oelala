# Research Todo List

> Lopende en geplande research taken voor oelala

---

## 🔄 In Progress

### LTX-2 Video Model
- **Status**: Research phase
- **Doc**: [LTX2_RESEARCH.md](LTX2_RESEARCH.md)
- **Findings**:
  - ✅ Native audio + video in één pass
  - ✅ NSFW niet expliciet verboden in license
  - ⚠️ 3GB VRAM claim - moet zelf testen
  - ✅ ComfyUI nodes beschikbaar
- **Next**:
  - [ ] Wacht op 15-sec benchmark completion
  - [ ] Installeer ComfyUI-LTXVideo nodes
  - [ ] Download ltx-2-19b-distilled-fp8 (~19GB)
  - [ ] Test met minimal I2V workflow
  - [ ] Test audio generation

### 15-Second Video Benchmark
- **Status**: Running (3/21 complete)
- **Script**: `scripts/benchmark_15sec.py`
- **Results so far**:
  - ❌ 384x680 @ 241f - Failed
  - ✅ 432x768 @ 241f - Success!
  - ✅ 480x848 @ 241f - Success!
- **Next**:
  - [ ] Wacht tot benchmark klaar is
  - [ ] Update MULTI_GPU_SETUP.md met resultaten

---

## 📋 Todo

### Legal & Licensing (HIGH PRIORITY)
- **Doc**: [LEGAL_RESEARCH.md](LEGAL_RESEARCH.md)
- [ ] Check of oelala content policy nodig heeft
- [ ] Audit alle LoRAs voor licenties
- [ ] Audit alle base models voor licenties
- [ ] Check WAN 2.2 license
- [ ] Check Flux license
- [ ] Bepaal of ToS nodig is voor launch

### CivitAI LoRA Licenties
- **Question**: Mogen we CivitAI LoRAs commercieel gebruiken?
- **Answer**: Per LoRA verschillend, check individuele licenties
- [ ] Maak lijst van alle gebruikte LoRAs
- [ ] Check licentie van elke LoRA
- [ ] Documenteer in LICENSE_AUDIT.md
- [ ] Vervang non-commercial LoRAs indien nodig

### Alternative Video Models
- [ ] Onderzoek Mochi
- [ ] Onderzoek CogVideoX
- [ ] Onderzoek Pika-style models
- [ ] Vergelijk kwaliteit/snelheid/VRAM

### Audio Generation
- [ ] Onderzoek standalone audio models (als LTX-2 niet werkt)
- [ ] MMAudio integratie checken
- [ ] Music generation opties

### Upscaling & Enhancement
- [ ] Onderzoek video upscalers
- [ ] Test RealESRGAN voor video
- [ ] Frame interpolation opties (RIFE, etc.)

---

## ✅ Completed

### Benchmark v3 (2026-01-08)
- Gefixt: Targeting wrong node (`WanImageToVideo` i.p.v. `EmptyWanLatentVideo`)
- Resultaten gedocumenteerd in [MULTI_GPU_SETUP.md](MULTI_GPU_SETUP.md)
- Key finding: ~160 frame limiet ongeacht resolutie

---

## 📚 Reference Docs

| Topic | Document |
|-------|----------|
| Multi-GPU Setup | [MULTI_GPU_SETUP.md](MULTI_GPU_SETUP.md) |
| LTX-2 Research | [LTX2_RESEARCH.md](LTX2_RESEARCH.md) |
| Legal Research | [LEGAL_RESEARCH.md](LEGAL_RESEARCH.md) |
| ComfyUI Inventory | [COMFYUI_INVENTORY.md](COMFYUI_INVENTORY.md) |
| Project Overview | [PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md) |

---

*Last updated: 2026-01-08*
