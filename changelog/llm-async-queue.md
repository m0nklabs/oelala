### Added
- **Async LLM Queue System**: All LLM prompt enhancement requests now go through a FIFO queue with background worker processing, preventing race conditions and ensuring serialized access to the Guardian LLM
- `LLMQueueManager` backend module (`llm_queue.py`) with job tracking, queue position reporting, and automatic cleanup of completed jobs (5 min TTL)
- `/generate-prompt` endpoint now returns immediately with `job_id` for async polling
- `/llm-job/{job_id}` polling endpoint for job status/result retrieval
- `useLLMEnhance` React hook for frontend tools — handles submit → poll → result flow with abort/cleanup support
- `LLMQueueIndicator` component showing queue position and processing state inline next to enhance buttons
- All 5 prompt-capable tools updated: TextToVideo, ImageToVideo, TextToImage, PromptGenerator, ImageToText

### Changed
- Backend systemd service reduced from 2 to 1 uvicorn worker (async I/O handles concurrency; fixes in-memory state sharing issues)
