# Workflow Quick Reference: Oelala ↔ ComfyUI

## Snelle Workflow Selectie

### 🎯 for Beginners (Eenvoudig & Snel)
| Oelala Workflow | ComfyUI Equivalent | Wanneer use |
|----------------|-------------------|-------------------|
| **basic_image_to_video** | `simple_working_workflow.json` | Eerste keer proberen |
| **text_to_video_light** | `quick_fix_workflow.json` | Snelle text-to-video |

### 🎬 for Professionele Resultaten
| Oelala Workflow | ComfyUI Equivalent | Wanneer use |
|----------------|-------------------|-------------------|
| **professional_video** | `corrected_workflow.json` | Hoge kwaliteit nodig |
| **text_to_video_wan2** | `user_workflow_fixed.json` | best kwaliteit |

## Workflow Switching Guide

### of ComfyUI to Oelala
1. **Identificeer je ComfyUI workflow type:**
   - Heeft het een ImageLoader? → use `/generate` endpoint
   - Heeft het CLIPTextEncode zonder image? → use `/generate-text` endpoint

2. **Map parameters:**
   - `CLIPTextEncode.text` → `prompt` parameter
   - `EmptyLatentImage.width/height` → use 512x512 (aanbevolen)
   - `KSampler.steps` → `num_frames` (16, 25, of 49)

3. **Selecteer model type:**
   - for snelheid: `model_type: "light"`
   - for balans: `model_type: "svd"`
   - for kwaliteit: `model_type: "wan2.2"`

### of Oelala to ComfyUI
1. **Ga to ComfyUI:** http://192.168.1.2:8188
2. **Laad overeenkomstige workflow:**
   - Oelala basic → `simple_working_workflow.json`
   - Oelala professional → `corrected_workflow.json`
   - Oelala text-to-video → `workflow_corrected_components.json`

## Praktische Voorbeelden

### Voorbeeld 1: Kat Video Maken
**Oelala Weg:**
```bash
# upload afbeelding via web interface
# Prompt: "majestic orange tabby cat exploring enchanted forest"
# Model: wan2.2, Frames: 49
# Resultaat: Professionele video
```

**ComfyUI Weg:**
```json
// Laad corrected_workflow.json
// upload dezelfde afbeelding to ImageLoader
// Stel dezelfde prompt in CLIPTextEncode
// Klik Queue Prompt
// Resultaat: Vergelijkbare professionele video
```

### Voorbeeld 2: Snelle test
**Oelala Weg:**
```bash
# use text-to-video light
# Prompt: "simple animation test"
# Model: light, Frames: 16
# Resultaat: Binnen 30 seconden
```

**ComfyUI Weg:**
```json
// Laad simple_working_workflow.json
// Stel prompt in CLIPTextEncode
// Klik Queue Prompt
// Resultaat: Binnen 30 seconden
```

## Workflow Troubleshooting

### as Oelala not works:
1. **Controleer backend:** `curl http://192.168.1.2:7998/health`
2. **Bekijk logs:** `tail -f src/backend/backend.log`
3. **Restart service:** `./scripts/start_web.sh`

### as ComfyUI not works:
1. **Controleer ComfyUI:** `ps aux | grep comfy`
2. **Restart indien nodig:** `Python main.py --listen 192.168.1.2 --port 8188`
3. **Laad simple workflow:** `simple_working_workflow.json`

## Performance Vergelijking

| Aspect | Oelala | ComfyUI |
|--------|--------|---------|
| **Setup Tijd** | Instant | 2-5 minuten |
| **Gebruiksgemak** | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Flexibiliteit** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Snelheid** | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Kwaliteit** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

## Aanbevelingen

### 🎯 Start with Oelala as:
- Je nieuw bent with AI video generatie
- Je snel resultaten wilt
- Je een gebruiksvriendelijke interface wilt

### 🔧 use ComfyUI as:
- Je geavanceerde workflows nodig hebt
- Je custom nodes wilt maken
- Je maximale controle wilt about the pipeline

### 🔄 Switch tussen beide:
- **Ontwikkeling:** use Oelala for snelle iteraties
- **Productie:** use ComfyUI for consistente, herhaalbare resultaten
- **Experimenten:** use ComfyUI for new technieken
- **Deployment:** use Oelala for gebruikersvriendelijke applicaties

## Workflow Templates Locaties

### Oelala Templates
- **Web Interface:** http://192.168.1.2:5174
- **Config file:** `/home/flip/oelala/workflow_templates.json`
- **Documentation:** `/home/flip/oelala/docs/OELALA_WORKFLOWS_README.md`

### ComfyUI Templates
- **Web Interface:** http://192.168.1.2:8188
- **Workflow Files:** `/home/flip/external-code/ComfyUI/*.json`
- **Documentation:** `/home/flip/external-code/ComfyUI/changelog.md`

Deze setup geeft je het best of beide werelden: the snelheid and eenvoud of Oelala, gecombineerd with the kracht and flexibiliteit of ComfyUI.
