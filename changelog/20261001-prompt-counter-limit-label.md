### Fixed

- Prompt character counters no longer claim a limit that does not exist: the
  hardcoded `/2048` suffix is gone from the positive and negative prompt fields in
  `TextToVideoTool` and `ImageToVideoTool` (4 places). Nothing enforced 2048 —
  the textarea had no `maxLength`, the submit guard only checks for non-empty
  input, and the backend request models carry no prompt `max_length`, so long
  prompts were always accepted in full. The plain character count stays.

### Changed

- The `2048` value only exists as a caption-generation token cap in
  `workflows/registry.json` (`max_tokens`, min 64 / max 2048 / default 512), which
  is unrelated to the prompt fields; the counter was copied in with the tool
  alignment commit `ab63a26` without the matching limit.
