### Added
- Cloud Max jobs now appear transparently in QueueIndicator alongside local jobs
- Cloud job completion automatically decodes base64 output, saves to `media/generated/cloud-max/`, and uploads to oelala-storage
- New `/media/generated/cloud-max/{filename}` endpoint serves Cloud Max output with proper CORS headers
- Cloud job status caching prevents redundant RunPod API calls on repeated polls
- I2V input images are now properly sent as base64 to RunPod handler (fixes I2V on cloud)

### Fixed
- Cloud Max I2V mode was missing base64 image data in RunPod submission (images were uploaded to local ComfyUI only)
- Cloud jobs were invisible in the queue panel (QueueIndicator only polled local ComfyUI)
- Cloud job completion had no output processing (base64 decode, file save, storage upload)
