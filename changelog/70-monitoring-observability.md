### Added
- Sentry SDK integration for backend (FastAPI) with automatic error capture, performance tracing, and profiling
- Sentry SDK integration for frontend (React) with browser tracing and session replay
- Global `ErrorBoundary` in React app — catches render crashes with a user-friendly fallback UI
- Request metrics middleware tracking total requests, error rates, status code distribution, and per-endpoint latency (p50/p95/max)
- `GET /health/deep` endpoint — tests ComfyUI, oelala-storage, Supabase, and disk space connectivity
- `GET /api/admin/metrics` endpoint — admin-only dashboard with request counts, latency summaries, uptime, and error rates
- `X-Response-Time` header on all API responses for debugging
- Sentry Vite plugin for source map upload (activated when `SENTRY_AUTH_TOKEN` is set)
- New environment variables in `.env.example`: `SENTRY_DSN`, `SENTRY_ENVIRONMENT`, `SENTRY_TRACES_SAMPLE_RATE`, `VITE_SENTRY_DSN`

### Changed
- Source maps now generated in production builds (required for Sentry stack traces)
