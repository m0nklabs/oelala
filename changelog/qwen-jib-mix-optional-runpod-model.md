### Added
- Added optional Qwen Image Edit model selection for cloud edit mode.
- Added support for selecting JIB Mix Qwen V6 as an alternative RunPod Qwen edit model.

### Changed
- Qwen cloud workflows now pass the selected model variant through frontend, backend, and RunPod worker execution.
- RunPod Qwen worker now lazy-downloads optional model variants only when a workflow references them.

### Fixed
- Avoided unconditional download of optional Qwen model variants during worker startup by resolving model requirements from the submitted workflow.
