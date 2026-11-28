### Changed

- The LoRA library location is configuration, not a literal. `LORA_SSD_DIR` names an extra root;
  unset, the checkout's own `ComfyUI/models/loras` is the only source. `project_paths` grows
  `lora_ssd_dir()` and `lora_roots()` (checkout first, mount second) and every reader uses them:
  the admin/listing endpoint, the file-serving root, `lora_scanner`, `lora_utils`, and the two
  scripts that walk the library.
- **Migration on the production box:** add one line to `.env` to keep today's behaviour exactly:

  ```bash
  LORA_SSD_DIR=/mnt/ssd/loras
  ```

  Without it, LoRA browsing lists only what the checkout holds. This is the only reason it is not a
  silent default change: the path is real, it is just not the code's business where it lives.
- `lora_utils.LORA_SSD_DIR` is `None` when nothing is configured, and the resolve loop filters it
  instead of iterating over a `None`.

Slice 2c of #226, and the last absolute path of this class outside the checkout.
