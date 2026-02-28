### Added
- Face LoRA training pipeline using ai-toolkit (ostris) with JuggernautXL SDXL base model
- `face_train_service.py`: training job management, Dreambooth captions, subprocess launcher, progress tracking
- Backend endpoints: `POST /api/face-train`, `GET /api/face-train`, `GET /api/face-train/loras`, `GET|DELETE /api/face-train/{job_id}`
- Frontend: 3-tab FaceSwapTool (Swap / Profiles / Train LoRA) with job progress bar, trained LoRA list, trigger word copy
- Pre-downloaded `buffalo_l` insightface detection model (cold-start prevention)
- All previous face swap commits cleanly cherry-picked onto `main` (removed from dependabot branch)
