# Oelala Workflows - ComfyUI Equivalent Setup

This document beschrijft hoe je workflows kunt opzetten in Oelala to dezelfde resultaten te krijgen as in ComfyUI.

## Overzicht

Oelala is ontworpen as een vereenvoudigde, gebruiksvriendelijke interface for AI video generatie. Terwijl ComfyUI een node-based workflow editor is, gebruikt Oelala een meer gestructureerde aanpak with:

- **backend API endpoints**: for different soorten video generatie
- **frontend components**: for gebruikersinteractie
- **Workflow Templates**: for consistente resultaten

## backend API endpoints

### 1. Image-to-Video Generatie (`/generate`)
```Python
POST /generate
- file: UploadFile (afbeelding)
- prompt: str (tekst prompt)
- num_frames: int (aantal frames, default: 16)
- output_filename: str (optionele custom naam)
```

**ComfyUI Equivalent:**
```
ImageLoader → CLIPTextEncode → SVDImageToVideo → SaveVideo
```

### 2. Text-to-Video Generatie (`/generate-text`)
```Python
POST /generate-text
- prompt: str (vereist - tekst beschrijving)
- num_frames: int (default: 16)
- model_type: str ("light", "svd", "wan2.2")
- output_filename: str (optioneel)
```

**ComfyUI Equivalent:**
```
CLIPTextEncode → EmptyLatentImage → KSampler → SVDTextToVideo → SaveVideo
```

### 3. Health Check (`/health`)
```Python
GET /health
```
Retourneert systeem Status and model beschikbaarheid.

## frontend Workflow components

### VideoGenerator component
**Locatie:** `src/frontend/src/components/VideoGenerator.jsx`

**functionality:**
- Afbeelding upload
- Prompt invoer
- Video parameters instellen
- Realtime voortgang tonen
- Resultaat downloaden

**Workflow Equivalent:**
```jsx
// upload afbeelding
// → Voer prompt in
// → Stel parameters in
// → Genereer video
// → Download resultaat
```

### LoRATrainer component
**Locatie:** `src/frontend/src/components/LoRATrainer.jsx`

**functionality:**
- Dataset upload
- Training parameters
- LoRA model generatie
- Training voortgang monitoren

## Workflow Templates

### Template 1: Basis Image-to-Video
```json
{
  "workflow_name": "basic_image_to_video",
  "description": "Eenvoudige conversion of afbeelding to video",
  "backend_endpoint": "/generate",
  "parameters": {
    "prompt": "cinematic video of uploaded image",
    "num_frames": 16,
    "model_type": "svd"
  },
  "comfyui_equivalent": "ImageToVideo_SVD.json"
}
```

### Template 2: Professionele Video Generatie
```json
{
  "workflow_name": "professional_video",
  "description": "Hoge kwaliteit video with custom instellingen",
  "backend_endpoint": "/generate",
  "parameters": {
    "prompt": "professional 4K cinematic video, high quality",
    "num_frames": 49,
    "model_type": "wan2.2"
  },
  "comfyui_equivalent": "Professional_Wan2_Video.json"
}
```

### Template 3: Text-to-Video
```json
{
  "workflow_name": "text_to_video",
  "description": "Genereer video alleen vanuit tekst",
  "backend_endpoint": "/generate-text",
  "parameters": {
    "prompt": "A majestic cat exploring enchanted forest",
    "num_frames": 25,
    "model_type": "light"
  },
  "comfyui_equivalent": "TextToVideo_Lightweight.json"
}
```

## Workflow execution Stappen

### Stap 1: backend Initialisatie
```Python
# in app.py startup event
generator = Wan2VideoGenerator(model_type="light")
generator.load_model()
```

### Stap 2: frontend Setup
```jsx
// in App.jsx
const [activeWorkflow] = useState('generate');

// Render juiste component gebaseerd on workflow
{activeWorkflow === 'generate' && <VideoGenerator />}
```

### Stap 3: API Call execution
```javascript
// in api.js
const response = await fetch('/generate', {
  method: 'POST',
  body: FormData
});
```

### Stap 4: Resultaat Verwerking
```javascript
// Resultaat tonen and download aanbieden
const videoUrl = response.video_url;
const downloadLink = `${BACKEND_BASE}/videos/${response.output_video}`;
```

## Workflow Vergelijking: Oelala vs ComfyUI

| Aspect | Oelala | ComfyUI |
|--------|--------|---------|
| **Gebruikersinterface** | Web-based GUI | Node editor |
| **Workflow Definitie** | Code-based templates | Visual node connections |
| **Aanpassing** | Parameter tuning | Node modifications |
| **Complexiteit** | Laag (beginner friendly) | Hoog (expert level) |
| **Snelheid setup** | Snel | Langzaam |
| **Flexibiliteit** | Beperkt | Uitgebreid |

## Workflow Debugging

### backend Logs
```bash
# Bekijk backend logs
tail -f /home/flip/oelala/src/backend/backend.log
```

### frontend Logs
```bash
# Bekijk frontend logs
tail -f /home/flip/oelala/src/frontend/frontend.log
```

### Health Check
```bash
# Controleer systeem Status
curl http://192.168.1.2:7998/health
```

## Workflow Optimalisatie

### 1. Model Selectie
- **Light**: Snelste, minste VRAM (4GB)
- **SVD**: Balans kwaliteit/snelheid
- **Wan2.2**: highest kwaliteit, meeste VRAM (8GB+)

### 2. Frame Optimalisatie
- **16 frames**: Snel, korte video's
- **25 frames**: Standaard kwaliteit
- **49 frames**: Hoge kwaliteit, langere videos

### 3. GPU Memory Management
```Python
# in wan2_generator.py
torch.cuda.empty_cache()  # Memory vrijmaken
```

## Workflow Uitbreiding

### new Workflow Toevoegen

1. **backend endpoint** toevoegen in `app.py`
2. **frontend component** maken in `components/`
3. **Menu Item** toevoegen in `App.jsx`
4. **API Call** toevoegen in `api.js`
5. **Workflow Template** documenteren

### Voorbeeld: Batch Processing Workflow
```Python
# backend: nieuw endpoint
@app.post("/generate-batch")
async def generate_batch_videos(files: list[UploadFile]):
    # Batch processing logic
    pass
```

```jsx
// frontend: new component
function BatchVideoGenerator() {
    // Batch upload and processing UI
}
```

## Troubleshooting Workflows

### Veelvoorkomende Problemen

1. **Model not geladen**
   - Controleer GPU memory
   - Restart backend service
   - Bekijk logs for errors

2. **Video generatie faalt**
   - Controleer input afbeelding formaat
   - Verlaag frame count
   - use lichtere model

3. **frontend errors**
   - Clear browser cache
   - Controleer network connectivity
   - Bekijk browser console

### Performance Monitoring
```Python
# in backend logs
logger.info(f"Generation time: {end_time - start_time}s")
logger.info(f"GPU Memory used: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
```

## Conclusie

Oelala workflows bieden een vereenvoudigde maar krachtige manier to AI video generatie out te voeren, vergelijkbaar with ComfyUI maar with minder complexiteit. by the gestructureerde aanpak could gebruikers snel resultaten behalen zonder diepgaande kennis of AI pipelines nodig te hebben.

for geavanceerde gebruikers blijft ComfyUI available via the sidebar link for meer complex workflows and fine-tuning mogelijkheden.
