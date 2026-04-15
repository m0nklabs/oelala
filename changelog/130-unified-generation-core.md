### Added
- Unified Generation Core foundation (`src/backend/generation/` package)
  - `types.py`: `MediaType`, `Operation`, `ComputeTarget`, `LoraFormat`, `AdapterConstraints`, `GenerationRequest`, `GenerationResult`, `LoraStackItem` data models
  - `adapter.py`: `GenerationAdapter` abstract base class for all generation adapters
  - `registry.py`: `AdapterRegistry` for registering, finding, and listing adapters
  - `router.py`: `GenerationRouter` for dispatching requests with credit checking, LoRA filtering, and control validation
  - `lora_utils.py`: Extracted LoRA helpers (resolve, sanitize, filter, download list)
- Qwen Edit cloud adapter (`adapters/cloud/qwen_edit.py`) — first adapter implementation
- 54 unit tests covering types, registry, router, LoRA utils, and Qwen Edit adapter
