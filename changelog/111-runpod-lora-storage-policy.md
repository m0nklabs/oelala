### Changed

- Replaced the old RunPod network volume with a fresh 50GB `EU-CZ-1` volume dedicated to LoRAs and hard-to-replace private/custom assets.
- Updated the RunPod worker policy so public/general Hugging Face models always stay on container disk or cached-model storage, never on the RunPod network volume.
- Documented the LoRA-only RunPod storage policy in the repository instructions, agent file, and RunPod deployment guide.
- Added an on-demand uploader for selected local LoRAs/private assets using the RunPod S3-compatible API, with guardrails that block general/public model uploads.
