### Added

- LoRA support for MiniMax-H3 cloud video generation (T2V + I2V): the RunPod
  adapters now pass the requested LoRA stack into the workflow
  (`LoraLoaderModelOnly` chain on the FL2VA model) and ship signed download
  URLs so the worker fetches the files before sampling — same mechanism as
  Wan2.2/LTX-2.3 cloud jobs.

### Changed

- Frontend T2V/I2V tools: the LoRA panel is now available for MiniMax-H3
  modes (cloud and local) with a single-stage selector limited to the
  `minimax-h3` LoRA category; the "LoRAs not supported yet" banners are
  replaced by notes that files are uploaded/downloaded automatically.
