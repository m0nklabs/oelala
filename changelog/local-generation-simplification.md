### Changed
- Simplified promoted local video generation modes into stable local, quality local, and explicit cloud-only modes.
- Documented local generation defaults and the mode-to-adapter policy.
- Backend validation now clamps requested frames and FPS against adapter constraints before queueing local work.

### Fixed
- Fixed T2V LTX-2.3 routing so the `ltx2` mode uses the `ltx23-cloud-t2v` adapter instead of falling back to local Wan2.2.
- Prevented cloud-only video modes from being submitted with `compute_target=local`.
- Restored Wan2.2 I2V extend/unet parameters by checking the actual stable local mode instead of a never-empty mode value.