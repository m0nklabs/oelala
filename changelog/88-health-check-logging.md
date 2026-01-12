### Added
- **Debug Logging for Health Check Endpoint**: Added debug logging to `/api/v1/health` endpoint
  - Logs timestamp and client IP address on each health check request
  - Uses existing `debug_log` pattern (controlled by `OELALA_DEBUG` environment variable)
  - Added test to verify logging behavior with mocking
