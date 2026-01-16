### Added
- Documentation: `docs/DISTORCH2_MULTI_GPU_SETTINGS.md` - Comprehensive DisTorch2 multi-GPU configuration guide
- Updated `docs/GENERATION_MODES_TREE.md` with tested frame limits and VRAM budgets

### Changed
- All DisTorch2 workflows updated with optimal allocation: `cuda:1,11gb;cuda:0,15gb;cpu,*`
  - Putting cuda:1 (3060) first makes it hold 97% of model weights (~11GB)
  - cuda:0 (5060 Ti) keeps only 3% model but has ~15GB free for activations
  - Result: 5060 Ti runs at 100% utilization while 3060 holds model
- Fixed text encoder filename in workflows: `umt5-xxl-enc-bf16.safetensors`

### Fixed
- Workflow GPU allocation order - cuda:1 must come FIRST to maximize 3060 VRAM usage
- Production frame limit set to 321 (safe) instead of 353-355 (OOM risk)

### Tested
- 480×848 @ 321 frames: ✅ Stable production max (~20 seconds)
- 480×848 @ 341 frames: ✅ Tight but works
- 480×848 @ 351-355 frames: ⚠️ Works sometimes, OOM risk
- 480×848 @ 357+ frames: ❌ OOM
