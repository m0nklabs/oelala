### Changed

- **Phase out Ollama, migrate to Guardian proxy** (`app.py`, `admin_api.py`):
  - Renamed `OLLAMA_BASE`/`OLLAMA_MODEL` constants to `GUARDIAN_BASE`/`GUARDIAN_MODEL`
  - Env var priority: `GUARDIAN_BASE_URL` > `GUARDIAN_BASE` > `OLLAMA_BASE` (backward compat)
  - All LLM API calls now use OpenAI `/v1/chat/completions` format instead of Ollama `/api/generate`
  - All calls now use Bearer token auth (`GUARDIAN_API_KEY`) instead of HTTP Basic auth
  - Added `_guardian_headers()` helper for consistent auth headers
  - Settings key `ollama_model` → `llm_model` (legacy key still read for migration)
  - Available models fetched from `/v1/models` instead of `/api/tags`

### Fixed

- **Vision analysis (Image-to-Text caption)** was broken: switched from Ollama multimodal format
  (`images: [base64]` in `/api/generate`) to OpenAI multimodal format (`image_url` content in
  `/v1/chat/completions`). This is required by llama.cpp's vision implementation.
- **Reasoning model support**: `max_tokens` bumped to 2048 and added fallback to `reasoning_content`
  when `content` is empty (reasoning models like GLM-4.7-Flash use separate fields for CoT)

### Added

- `GET /guardian/status` endpoint — replaces `/ollama/status` (kept as deprecated alias)
- `GUARDIAN_MODEL=GLM-4.7-Flash` and `VISION_MODEL=Step3-VL-10B` added to `.env`
