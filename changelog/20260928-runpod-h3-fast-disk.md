### Changed
- RunPod MiniMax-H3 worker: ComfyUI now starts with an explicit `--fast-disk` flag
  - v0.37.2's auto-detection only enables disk-backed weight streaming when the
    model file's backing device is a real NVMe (>= Gen4 x4) and every file agrees
  - On RunPod hosts where the container disk is not exposed as an NVMe device in
    /sys/class/block, the auto-detection silently disables streaming, pinning
    42.5 GB of H3 weights in host RAM (only ~50 GB on A6000/A40 hosts) and
    slowing large jobs toward the workflow timeout
  - RunPod GPU hosts use local NVMe container disk, so forced fast-disk matches
    the hardware and makes streaming behaviour deterministic per host
