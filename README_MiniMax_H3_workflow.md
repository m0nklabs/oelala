# MiniMax H3 T2V workflow — ComfyUI / Comfy Desktop

Bestand: `MiniMax_H3_T2V_workflow.json`
Gebouwd op de **officiële ComfyUI 0.33.1-template** `video_minimax_h3_t2v`, mét de
text-encoder aangepast naar de **int8_convrot** variant die op de server staat
(download uit de Comfy-Org repack).

## Wat zit erin
- MiniMax H3 **text-to-video** (ook primed voor audio: native stereo uit het model)
- 5s video, 864×480 (0.4 MP, aanpasbaar via de Resolution Selector: 16:9 / andere
  aspecten; max 768×1344 korte kant)
- Model: `minimax_h3_fl2va_pruned_int8_convrot.safetensors`
- Text encoder: `qwen3vl_32b_minimax_h3_int8_convrot.safetensors`
- Video VAE `minimax_h3_video_vae_fp16.safetensors` + Audio VAE `minimax_h3_audio_vae_fp32.safetensors`
- Uitvoer: `SaveVideo` → `output/video/`

## Openen

- **In de browser (aanbevolen)** — server draait al met alle modellen op schijf:
  `http://COMPUTE_NODE_2_HOST:8188` (de remote ComfyUI node die minimax_h3 draait) → sleep
  het .json op het canvas (of Workflow → Open).
- **In Comfy Desktop:** Workflow-menu → Open → dit bestand.

## Belangrijk — waar Desktop z'n modellen vandaan haalt

De 5 model-bestanden staan in de **portable** install:
`C:\PROGRAMME\ComfyUI_windows_portable\ComfyUI\models\`  (`diffusion_models/`, `text_encoders/`, `vae/`, `loras/`).

Comfy **Desktop** gebruikt z'n **eigen** modellen-map en ziet die dus niet
automatisch. Twee opties:

1. **Wijs Desktop op de portable-modellen** (aanbevolen, geen 32 GB kopiëren):
   ComfyUI Desktop → Settings → zoek "Model Directory" / "extra model paths"
   → voeg `C:\PROGRAMME\ComfyUI_windows_portable\ComfyUI\models` toe.
2. of **kopieer** de 4 bestanden naar Desktop z'n map:
   `diffusion_models\minimax_h3_fl2va_pruned_int8_convrot.safetensors`
   `text_encoders\qwen3vl_32b_minimax_h3_int8_convrot.safetensors`
   `vae\minimax_h3_video_vae_fp16.safetensors`
   `vae\minimax_h3_audio_vae_fp32.safetensors`

Meestal zit de Desktop-modellenmap op `%USERPROFILE%\Documents\ComfyUI\models`.

## Server-omgeving (draait al)

- ComfyUI 0.33.1, RTX 5060 Ti 16GB, `--listen 0.0.0.0 --port 8188 --fast-disk`
  (disk-streaming vanaf de NVMe, CPU/RAM offload automatisch via DynamicVRAM)
- Poort 8188 staat open voor in het LAN (firewall-regel "ComfyUI 8188").
- Herstart bij inloggen via taak `ComfyUIServer` (log: `comfy_server.log`).

## VRAM-tip die bij het on-thread hoort

Gebruikt de 16GB-kaart, en de int8-gewichten stromen via NVMe binnen. Mocht je
toch OOM krijgen: eerste zet `--disable-pinned-memory` aan in
`start_comfy_server.bat`, daarna `--vram-headroom 1`.

---

Gemaakt: 2026-08-20. Vragen? Zeg het gerust.
