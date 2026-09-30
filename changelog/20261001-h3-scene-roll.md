### Added
- **H3 scene roll** for the prompt generator: `randomize` (+ `scene_seed`, `duration`) on
  `/generate-prompt` rolls a concrete scene — cast, place, lighting, wardrobe, mood, style
  pack, story format, act arc with per-shot camera — and hands it to the prompt model as a
  constraint block, so shots follow a real structure instead of generic filler. The rolled
  scene comes back in the response and is shown in the tool.
- `src/backend/generation/prompt_scene.py` + `src/backend/h3_scene_vocab.json`: scene
  vocabulary and H3 stage knowledge (772 options across 46 dimensions, 7 stage foley
  profiles, 15 style packs, 10 story formats) extracted from the Ref2Vid randomizer page by
  `scripts/extract_h3_scene_vocab.py` (brace-matching parser, committed JSON, no runtime
  dependency on the page).
- 🎲 "Roll a scene" toggle in the Prompt Generator (H3 target), with a card showing the
  rolled setting and shot list.

### Changed
- The MiniMax-H3 prompt skill now carries **sound and body guards** learned from that page:
  wet/close-mic sound must never read as chewing, crunching or eating; bodies keep two arms,
  two legs, one head and five separate fingers without melting into each other or furniture;
  continuity carries across cuts; one camera move per shot and a locked reference camera for
  image-to-video inputs.
