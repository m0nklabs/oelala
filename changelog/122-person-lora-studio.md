### Changed
- Switched person LoRA training to default to the official `stabilityai/stable-diffusion-xl-base-1.0` model reference instead of a hard-coded Juggernaut XL checkpoint.
- Raised the minimum face LoRA dataset size to 5 reference images in both backend validation and frontend UI.
- Replaced the old placeholder LoRA Training dashboard tool with a dedicated Person LoRA Studio backed by `/api/face-train`.

### Added
- Added a saved face LoRA library view inside the LoRA Training tool, including trigger copy actions and browser-index refresh.
- Documented the dedicated LoRA training UI flow and updated face-system notes for SDXL-base training.
