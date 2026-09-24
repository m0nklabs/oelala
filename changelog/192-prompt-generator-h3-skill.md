### Added

- Prompt Generator: dedicated MiniMax-H3 target-model section. Selecting it
  activates the official H3-Context-IR prompt-writing skill (distilled from
  MiniMax-AI/MiniMax-H3 `skills/h3-prompt-writing/`): one prompt drives video
  and synchronized soundtrack, three fixed sections, no negative prompt, plus
  an optional image-to-video (first keyframe) variant.
- The H3 skill runs on `Huihui-Qwen3.5-9B-abliterated` via the Guardian proxy
  (env-overridable via `H3_PROMPT_MODEL`); model added to the SFW/NSFW model
  lists.

### Changed

- Guardian LLM calls no longer send a fixed `max_tokens` cap (prompt
  generation, AI suggest, vision analysis, text-to-text): reasoning models
  were truncated mid-think and never reached their answer. `max_tokens` is now
  an explicit opt-in parameter only (`analyze_image_with_vision`).
- Prompt-generation JSON parsing is robust to reasoning models: it extracts
  the last complete JSON object with a `prompt` key from thinking output.
