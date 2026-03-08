### Fixed
- RunPod Cloud Max worker startup no longer treats I2V-specific models as mandatory for every boot.
- T2V jobs now avoid unnecessary I2V model preparation during startup; mode-specific models are resolved on demand from cached-model storage or Hugging Face.

### Changed
- Documented Cloud Max startup policy: preload only shared core assets at boot, defer I2V/T2V-specific UNETs and CLIP Vision to workflow demand.