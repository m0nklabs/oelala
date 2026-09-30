### Fixed
- Image-to-video failed for MiniMax-H3 with `resolved is not defined`: the H3 variant
  payload lines were copied from the Text-to-Video tool, which merges a settings profile
  into a `resolved` object that the Image-to-Video tool does not build. The tool now reads
  its own state (`h3Variant` / `h3QualityMode`) directly.
