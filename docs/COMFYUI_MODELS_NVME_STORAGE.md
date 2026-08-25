# Storage Migration Documentation - Feb 14, 2026

## Overview
Due to disk space constraints on the root filesystem (`/`), large ComfyUI model directories were migrated to a dedicated high-speed NVMe drive (`/mnt/ali_nvme_500gb`).

## Migrated Directories
The following directories in `/home/flip/oelala/ComfyUI/models/` are now **Symlinks** pointing to `/mnt/ali_nvme_500gb/comfy_models/`:

1.  `unet` (~155 GB)
2.  `checkpoints` (~104 GB)
3.  `text_encoders` (~97 GB)
4.  `diffusion_models` (~58 GB)

**Total Freed Space:** ~414 GB

## File System Details
- **New Device:** `/dev/vdd1` (Label: `AI_NVME`)
- **Mount Point:** `/mnt/ali_nvme_500gb`
- **File System:** ext4
- **fstab Entry:** Added to `/etc/fstab` for auto-mount at boot.

## How to Verify
Run `ls -l /home/flip/oelala/ComfyUI/models/` and look for arrows `->` pointing to the new location.

## How to Revert (Rollback)
If you need to move everything back to the root drive (ensure you have 450GB+ free space first!):

1.  **Remove the symlink:**
    ```bash
    rm /home/flip/oelala/ComfyUI/models/<directory_name>
    ```
    *(Note: minimal risk, this only deletes the pointer, not the data)*

2.  **Move data back:**
    ```bash
    mv /mnt/ali_nvme_500gb/comfy_models/<directory_name> /home/flip/oelala/ComfyUI/models/
    ```

3.  **Repeat** for all affected directories.
