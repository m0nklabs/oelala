### Fixed
- **Cloud gen polling deadlock**: QueueIndicator now polls pending cloud jobs (not just running), fixing the deadlock where cloud jobs were submitted but never progressed past IN_QUEUE.

### Changed
- **handler.py `_progress()` log delta**: Now sends only NEW log lines (delta) on each heartbeat instead of the entire accumulated buffer. Prevents exponentially growing payloads on 20+ minute jobs.
- **handler.py `wait_for_completion()` adaptive polling**: 2s during startup (first 30s), 5s during generation. Reduces unnecessary polls from ~400 to ~250 for 20-min jobs.
- **handler.py `encode_outputs()` streaming base64**: Encodes files in 3MB chunks instead of loading entire file + base64 copy in memory. Reduces peak memory from 3x to ~1.3x file size for large videos.
- **handler.py `wait_for_cuda()` in-process check**: Uses in-process `torch.cuda` check instead of spawning a full Python subprocess per attempt (~2-5s overhead eliminated per retry).
- **handler.py `_candidate_cached_model_roots()` cached**: Results cached after first call since env vars don't change during container lifetime.
- **handler.py cleanup between jobs**: Output directory cleaned of stale files from previous jobs on warm workers.
- **handler.py `handler()` try/finally**: Log handler cleanup uses try/finally instead of scattered removeHandler calls, preventing handler leaks on unexpected exceptions.

### Added
- **Multi-endpoint RunPod client**: `RUNPOD_ENDPOINT_IDS` env var support for comma-separated endpoint failover with `_candidate_endpoint_ids()` ordering.
