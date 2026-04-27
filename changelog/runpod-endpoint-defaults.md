### Changed
- Centralized RunPod endpoint defaults and per-job execution policies for Wan2.2, LTX-2.3, and I2I workers.
- Aligned LTX-2.3 cloud defaults with the distilled 8-step, cfg=1.0 workflow.
- Updated RunPod deployment scripts and documentation to preserve current endpoint scaling defaults.

### Fixed
- Restored the `generation.adapters.cloud.qwen_edit` compatibility exports used by Qwen edit tests.
