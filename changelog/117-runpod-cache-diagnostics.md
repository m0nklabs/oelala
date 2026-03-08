### Added
- RunPod Cloud Max worker startup now logs a clear cache/readiness summary for shared startup models versus deferred workflow-specific models.
- Per-job workflow diagnostics now log whether a request is T2V, I2V, mixed, or shared-core only, plus which referenced models are cache hits versus download fallbacks.