### Fixed
- "Use in tool" from My Media was image-only for EVERY cloud generation: the
  frontend fetched workflow metadata from the local ComfyUI output dir
  (`/comfyui-metadata`), where cloud files never exist, and the backend
  workflow extractor only read the `comment` ffprobe tag while ComfyUI/VHS
  embeds the graph in the `prompt` tag. My Media now uses the user-bucket
  workflow endpoint (`/user/media/{type}/{file}/workflow`, local fallback
  kept) and the extractor tries `prompt`/`comment`/`workflow` tags.
- Cloud output filenames no longer hardcode "wan22": MiniMax-H3 / LTX-2.3
  generations were saved as `..._cloud_wan22_...` because the save path
  predated the multi-model cloud support. Filenames and staging bucket dirs
  are now model-family aware (`cloud_minimax_h3_...`, bucket
  `generated/cloud-minimaxh3/`); a generic serving route
  `/media/generated/cloud-{family}/...` serves the new buckets while the
  legacy `cloud-wan22` / `cloud-max` routes keep working for old files.
- "Use in tool" now prefills MiniMax-H3 settings from a generation's embedded
  workflow metadata: the frontend parser did not know the H3 graph shape
  (`SamplerCustomAdvanced`/`BasicScheduler`/`RandomNoise`/`MiniMaxH3ImageToVideo`),
  so steps, seed, fps, resolution, frame count and user LoRAs were missing
  from the import modal (prompt worked via the generic fallback). Known turbo
  LoRAs are skipped on import — they come back via the quality-mode preset.
  Wan2.2 parsing behaviour is unchanged (regression-tested).
