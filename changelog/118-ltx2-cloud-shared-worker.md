### Changed

- Enabled `ltx2` text-to-video cloud routing through the existing RunPod worker path instead of forcing LTX-2 to local-only execution.
- Expanded the shared RunPod worker image to include `ComfyUI-GGUF` and `ComfyUI-LTXVideo`, so Wan and LTX workflows can run from the same on-demand Docker image.
- Updated RunPod deployment docs to reflect the single-image, multi-model cloud worker strategy.